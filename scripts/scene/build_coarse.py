"""Generate the buffered OSM road network, calibrated terrain and building masses."""

import argparse
import json
from pathlib import Path

from map_generation.buildings import build_buildings
from map_generation.coarse import convert_osm, read_roads, route_check
from map_generation.scene import build_scene, save_scene
from map_generation.terrain import Terrain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--netconvert", type=Path, required=True)
    parser.add_argument("--sumo-home", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-scene", type=Path, help="Reuse completed terrain and building meshes after a road-source revision")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    scene = build_scene(config, args.data_root)
    output = args.output_dir.resolve()
    network = convert_osm(config, args.data_root, output / "network", args.netconvert, args.sumo_home)
    terrain = Terrain(scene, config, args.data_root, bounds=config["coarse"]["terrain_bounds_local_xy"])
    if args.reuse_scene:
        previous = json.loads(args.reuse_scene.read_text())
        scene["terrain"] = previous["terrain"]
        scene["buildings"] = previous["buildings"]
        scene["reused_static_geometry"] = str(args.reuse_scene)
        del previous
    else:
        scene["terrain"] = terrain.to_mesh()
        scene["buildings"] = build_buildings(scene, config, args.data_root, terrain)
    scene["coarse_map"] = read_roads(config, network, terrain)
    scene["coarse_map"]["check_route"] = route_check(scene, config)
    scene["purpose"] = "Coarse scene for AV2 fusion and native asset construction"
    save_scene(scene, output / "scene.json")
    summary = {"map_id": scene["map_id"], "output": str(output / "scene.json"),
               "lanes": len(scene["coarse_map"]["lanes"]), "buildings": len(scene["buildings"]),
               "terrain_vertices": len(scene["terrain"]["vertices"]),
               "route_passed": scene["coarse_map"]["check_route"]["passed"]}
    print(json.dumps(summary, indent=2))
    if not summary["route_passed"]:
        raise SystemExit("Coarse cross-junction route failed; inspect saved SceneIR before proceeding")


if __name__ == "__main__":
    main()
