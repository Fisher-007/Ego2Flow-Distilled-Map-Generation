"""Cut measured facade openings into selected building solids."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import shapely


def polygon_parts(geometry):
    return [p for p in shapely.get_parts(geometry)
            if p.geom_type == "Polygon" and p.area > 1e-10]


def cut_solid(solid, openings):
    base, roof = solid["base_z_m"], solid["roof_z_m"]
    shell = shapely.Polygon(solid["exterior"], solid.get("holes", []))
    levels = sorted({base, roof} | {
        max(base, min(roof, opening[key]))
        for opening in openings for key in ("bottom_z_m", "top_z_m")})
    parts, removed = [], 0.
    for low, high in zip(levels, levels[1:]):
        active = [shapely.Polygon(o["footprint_xy_m"]) for o in openings
                  if o["bottom_z_m"] < high and o["top_z_m"] > low]
        retained = shell.difference(shapely.union_all(active)) if active else shell
        removed += (shell.area - retained.area) * (high - low)
        for polygon in polygon_parts(retained):
            part = deepcopy(solid)
            part.update(exterior=list(map(list, polygon.exterior.coords)),
                        holes=[list(map(list, ring.coords)) for ring in polygon.interiors],
                        base_z_m=low, roof_z_m=high, height_m=high-low)
            parts.append(part)
    return parts, removed


def build(plan, conditions):
    selected = {r["building_id"]: r for r in conditions["buildings"]}
    result = {k: deepcopy(v) for k, v in plan.items()
              if k not in ("buildings", "retained_buildings", "graph_condition")}
    result.update(buildings=[], retained_buildings=[], replace_complete_building_set=False,
                  previous_native_import=conditions["previous_native_import"],
                  scope=conditions["scope"], observed_opening_conditions=conditions)
    reports = []
    for original in plan["buildings"]:
        if original["id"] not in selected:
            continue
        condition = selected[original["id"]]
        row = deepcopy(original)
        parts, removed = [], 0.
        for solid in original.get("solid_parts", [original]):
            fragments, volume = cut_solid(solid, condition["openings"])
            parts.extend(fragments)
            removed += volume
        row.update(solid_parts=parts, previous_source_ids=[original["id"]],
                   detail_policy="Explicit source opening solids; decorative panels cannot close the observed access.")
        result["buildings"].append(row)
        reports.append(dict(building_id=original["id"], solid_parts=len(parts),
                            removed_volume_m3=removed,
                            source_footprint_and_height_outside_cut_retained=True,
                            navigation_and_permissions_unchanged=True,
                            native_validation_pending=True))
    return result, reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "conditions", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    plan, conditions = [json.loads(p.read_text()) for p in (args.plan, args.conditions)]
    output, rows = build(plan, conditions)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"frontage_plan.json").write_text(json.dumps(output, separators=(",", ":"))+"\n")
    report = dict(buildings=rows, selected=len(rows), unselected_reused=len(plan["buildings"])-len(rows),
                  source_plan=str(args.plan), source_conditions=str(args.conditions),
                  scope=conditions["scope"])
    (args.output/"opening_review.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
