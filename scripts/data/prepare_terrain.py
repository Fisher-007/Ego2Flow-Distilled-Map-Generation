#!/usr/bin/env python3
"""Export a small USGS 3DEP bare-earth DEM for the configured coarse bounds."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
SERVICE = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"


def request_json(operation, parameters):
    with urlopen(f"{SERVICE}/{operation}?{urlencode(parameters)}", timeout=60) as response:
        result = json.load(response)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--request-file", type=Path, help="Prepared EPSG and UTM bounds for a separate download host")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--raster-id", type=int,
                        help="USGS catalog raster ID; use the source raster ID for the requested region")
    parser.add_argument("--raster-name", help="Spatially resolve a USGS source name; catalog object IDs can change")
    parser.add_argument("--resolution-m", type=float, default=1)
    args = parser.parse_args()
    if args.request_file:
        request = json.loads(args.request_file.read_text())
        epsg, bounds, relative = request["epsg"], np.asarray(request["bounds_utm_xy"]), request["output"]
    else:
        from map_generation.coordinates import LocalFrame
        config = json.loads(args.config.read_text())
        frame = LocalFrame(config["city"], config["coordinates"]["origin_city_xyz"])
        local_bounds = np.asarray(config["coarse"]["terrain_bounds_local_xy"], dtype=float)
        bounds = local_bounds + np.tile(frame.origin_utm, 2)
        epsg, relative = frame.epsg, config["sources"]["terrain_dem"]
    size = np.ceil((bounds[2:]-bounds[:2])/args.resolution_m).astype(int)
    query = {"f": "json", "returnGeometry": "false",
             "outFields": "Name,ProductName,Source,VerticalDatum,title,URL,AcquisitionDate"}
    if args.raster_name:
        query.update(where="Name='"+args.raster_name.replace("'", "''")+"'",
                     geometry=",".join(map(str,bounds)), geometryType="esriGeometryEnvelope",
                     inSR=epsg, spatialRel="esriSpatialRelIntersects")
    elif args.raster_id is not None:
        query["objectIds"] = args.raster_id
    else:
        parser.error("Supply --raster-name or --raster-id")
    datasets = [f["attributes"] for f in request_json("query", query)["features"]]
    if not datasets:
        raise ValueError("No USGS source raster intersects the requested map bounds")
    dataset = datasets[0]
    raster_ids = [d["OBJECTID"] for d in datasets]
    parameters = {"f": "json", "bbox": ",".join(map(str, bounds)), "bboxSR": epsg,
                  "imageSR": epsg, "size": ",".join(map(str, size)), "format": "tiff",
                  "pixelType": "F32", "noData": -999999, "interpolation": "RSP_BilinearInterpolation",
                  "renderingRule": json.dumps({"rasterFunction": "None"}),
                  "mosaicRule": json.dumps({"mosaicMethod": "esriMosaicLockRaster", "lockRasterIds": raster_ids})}
    result = request_json("exportImage", parameters)
    path = args.data_root/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(result["href"], timeout=60) as response:
        contents = response.read()
    path.write_bytes(contents)
    extent = result["extent"]
    metadata = {"service": SERVICE, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "epsg": epsg, "extent_utm_xy": [extent[k] for k in ("xmin", "ymin", "xmax", "ymax")],
                "vertical_datum": dataset["VerticalDatum"], "dataset": dataset, "datasets": datasets,
                "request": parameters, "source_query": query,
                "response": {key: value for key, value in result.items() if key != "href"},
                "bytes": len(contents), "license": "USGS public domain; credit USGS 3DEP",
                "selection": "Configured coarse terrain extent covers complete selected roads and building footprints."}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({"dem": str(path), "bytes": len(contents), "raster_ids": raster_ids,
                      "extent_utm_xy": metadata["extent_utm_xy"]}, indent=2))


if __name__ == "__main__":
    main()
