"""Acquire a city-sized OSM region in reusable tiles, preserving source IDs."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import numpy as np

API = "https://api.openstreetmap.org/api/0.6"


def fetch(url, path):
    if not path.exists():
        request = Request(url, headers={"User-Agent": "MapBuilder-research-map-acquisition"})
        with urlopen(request, timeout=120) as response:
            payload = response.read()
        ET.fromstring(payload)
        path.write_bytes(payload)
    return ET.parse(path).getroot()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--request-file", type=Path, help="Prepared requests for a separate download host")
    parser.add_argument("--prepare-only", type=Path, help="Write source requests without downloading")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tile-m", type=float, default=1200)
    args = parser.parse_args()
    if args.request_file:
        prepared = json.loads(args.request_file.read_text())
    else:
        from map_generation.coordinates import LocalFrame
        config = json.loads(args.config.read_text())
        frame = LocalFrame(config["city"], config["coordinates"]["origin_city_xyz"])
        bounds = np.asarray(config["bbox_city_xy"]) + np.array([-1,-1,1,1]) * config["coarse"]["buffer_m"]
        x, y = [np.linspace(bounds[k], bounds[k+2], int(np.ceil((bounds[k+2]-bounds[k])/args.tile_m))+1)
                for k in (0, 1)]
        prepared = {"osm": config["sources"]["osm"], "tiles": []}
        for i, (a, b) in enumerate(zip(x[:-1], x[1:])):
            for j, (c, d) in enumerate(zip(y[:-1], y[1:])):
                corners = frame.local_to_lonlat(frame.city_to_local([[a,c],[a,d],[b,c],[b,d]]))
                bbox = [*corners.min(0), *corners.max(0)]
                prepared["tiles"].append({"url": API+"/map?bbox="+",".join(map(str,bbox)),
                                          "file": f"tile_{i}_{j}.osm", "bbox_lonlat": bbox})
    if args.prepare_only:
        args.prepare_only.parent.mkdir(parents=True, exist_ok=True)
        args.prepare_only.write_text(json.dumps(prepared, indent=2)+"\n")
        return
    path = args.data_root / prepared["osm"]
    cache = path.parent / (path.stem + "_tiles")
    cache.mkdir(parents=True, exist_ok=True)
    elements, requests, relations = {}, [], []

    def merge(root):
        for element in root:
            if element.tag in ("node", "way", "relation"):
                key = (element.tag, element.get("id"))
                old = elements.get(key)
                if old is None or int(element.get("version", 0)) >= int(old.get("version", 0)):
                    elements[key] = element

    for request in prepared["tiles"]:
        tile = cache / request["file"]
        merge(fetch(request["url"], tile))
        requests.append({**request, "cache": str(tile)})
        print(f"OSM tile {len(requests)}/{len(prepared['tiles'])}; elements {len(elements)}", flush=True)
    # Complete building and restriction relations at tile boundaries.
    for (kind, identity), relation in list(elements.items()):
        if kind != "relation":
            continue
        tags = {t.get("k"): t.get("v") for t in relation.findall("tag")}
        if not ("building" in tags or "building:part" in tags or tags.get("type") == "restriction"):
            continue
        missing = [(m.get("type"), m.get("ref")) for m in relation.findall("member")
                   if (m.get("type"), m.get("ref")) not in elements]
        if missing:
            url = API + f"/relation/{identity}/full"
            merge(fetch(url, cache / f"relation_{identity}.osm"))
            relations.append({"relation_id": identity, "url": url, "missing_members": missing})
    root = ET.Element("osm", version="0.6", generator="MapBuilder tiled source acquisition")
    for key in sorted(elements, key=lambda key: ({"node":0,"way":1,"relation":2}[key[0]],int(key[1]))):
        root.append(elements[key])
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    report = {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "source": API,
              "license": "OpenStreetMap contributors, ODbL", "tiles": requests,
              "completed_relations": relations, "elements": len(elements), "osm": str(path)}
    path.with_suffix(".request.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"osm": str(path), "tiles": len(requests), "elements": len(elements)}))


if __name__ == "__main__":
    main()
