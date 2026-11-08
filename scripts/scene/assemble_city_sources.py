"""Prepare an explicitly reviewed AV2 source assembly for the shared scene reader."""

import argparse
import json
from pathlib import Path

from map_generation.city_sources import combine_exact_observations
from map_generation.city_registration import register_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--log-id", action="append", help="Current construction subset; repeat for each actual log")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", type=Path, help="Agent-reviewed identical-boundary alias and marking source choices")
    parser.add_argument("--geometry-review", type=Path, help="Reviewed log registrations and geometric lane identities")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    sources = []
    for relative in config["sources"]["av2_logs"]:
        if args.log_id and Path(relative).name not in args.log_id:
            continue
        path = next((args.data_root/relative/"map").glob("log_map_archive_*.json"))
        sources.append((str(path.relative_to(args.data_root)),json.loads(path.read_text())))
    review = json.loads(args.review.read_text()) if args.review else {}
    geometry_review=json.loads(args.geometry_review.read_text()) if args.geometry_review else {}
    sources,registrations=register_sources(sources,geometry_review)
    assembly = combine_exact_observations(sources, review.get("marking_aliases"),geometry_review.get('geometry_aliases'),geometry_review.get('ownership_review'))
    result = {"lane_segments": {str(lane["id"]):lane for lane in assembly["lanes"]},
              "drivable_areas": {}, "pedestrian_crossings": {}}
    asset_conflicts = []
    asset_observations = {}
    asset_resolutions = []
    selections = geometry_review.get('asset_ownership',{})
    for category in ('drivable_areas','pedestrian_crossings'):
        observations = {}
        for source,data in sources:
            for key,feature in data[category].items():
                observations.setdefault(key,[]).append(dict(source=source,feature=feature))
        asset_observations[category] = observations
        for key,rows in observations.items():
            unequal = any(row['feature'] != rows[0]['feature'] for row in rows[1:])
            choice = selections.get(category,{}).get(key)
            if unequal and choice:
                owner = next((row for row in rows if Path(row['source']).parts[-3] == choice['source_log']),None)
                if owner is None or not choice.get('reason'):
                    raise ValueError(f'Asset source selection has no observed owner: {category} {key}')
                result[category][key] = owner['feature']
                asset_resolutions.append(dict(kind=category,source_id=key,selected_source=owner['source'],
                    reason=choice['reason'],alternative_sources=[row['source']for row in rows if row is not owner]))
            else:
                result[category][key] = rows[0]['feature']
                if unequal:
                    asset_conflicts.extend(dict(kind=category,source_id=key,source=row['source'])for row in rows[1:]if row['feature']!=rows[0]['feature'])
    report = {"source_maps": [s for s,_ in sources],
              "source_review": str(args.review) if args.review else None,
              "geometry_review": str(args.geometry_review) if args.geometry_review else None,
              "log_registrations": registrations,
              **{key:value for key,value in assembly.items() if key!="lanes"},
              "asset_conflicts": asset_conflicts,
              "asset_source_observations": asset_observations,
              "asset_ownership_resolutions": asset_resolutions,
              "lane_count": len(result["lane_segments"]),
              "purpose": "Current source assembly; dynamic and physical city-map validation follows compilation."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix(".review.json").write_text(json.dumps(report,indent=2)+"\n")
    if assembly["source_id_geometry_conflicts"] or assembly["neighbor_conflicts"] or asset_conflicts:
        raise SystemExit("Source assembly needs the recorded source ownership review before compilation.")
    result["source_assembly"] = report
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print({"sources":len(sources),"lanes":len(assembly["lanes"]),"output":str(args.output)})


if __name__ == "__main__":
    main()
