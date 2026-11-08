"""Combine exact AV2 observations while retaining evidence for source conflicts."""

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path


GEOMETRY_FIELDS = ("lane_type", "is_intersection", "left_lane_mark_type", "right_lane_mark_type",
                   "left_lane_boundary", "right_lane_boundary")


def combine_exact_observations(sources, reviewed_aliases=None, geometry_aliases=None, ownership_review=None):
    """Unify identical directed lane geometry and carry its observed links.

    Source IDs alone do not identify physical lanes across logs. Equivalence
    here requires exact boundary coordinates, markings, type and intersection
    status. Approximate overlaps are left for ownership/registration review.
    """
    reviewed_aliases = {int(k): int(v) for k,v in (reviewed_aliases or {}).items()}
    geometry_aliases = {int(k):int(v) for k,v in (geometry_aliases or {}).items()}
    observed = {item["id"]:item for _,data in sources for item in data["lane_segments"].values()}
    structural_fields = [f for f in GEOMETRY_FIELDS if "mark_type" not in f]
    ownership_review = ownership_review or {}
    source_id_choices = {int(k):v for k,v in ownership_review.get("source_id_choices",{}).items()}
    source_id_resolutions = []
    for key, choice in source_id_choices.items():
        alternatives = [(source, data["lane_segments"][str(key)]) for source,data in sources
                        if str(key) in data["lane_segments"]]
        source, selected = next((source,lane) for source,lane in alternatives
                                if Path(source).parts[-3] == choice["source_log"])
        if not choice.get("reason") or not choice.get("evidence"):
            raise ValueError("Same-ID marking ownership requires Agent reason and evidence")
        if any(lane[field] != selected[field] for _,lane in alternatives for field in structural_fields):
            raise ValueError("Same-ID marking ownership cannot replace measured boundary geometry or lane type")
        observed[key] = selected
        source_id_resolutions.append(dict(source_lane_id=key,selected_source=source,
            reason=choice["reason"],evidence=choice["evidence"],
            alternate_sources=[path for path,_ in alternatives if path != source]))
    for key, owner in reviewed_aliases.items():
        if any(observed[key][f] != observed[owner][f] for f in structural_fields):
            raise ValueError(f"Reviewed marking alias changes measured geometry or lane type: {key} -> {owner}")
    for key,owner in geometry_aliases.items():
        if any(observed[key][f]!=observed[owner][f] for f in ('lane_type','is_intersection')):
            raise ValueError(f'Reviewed geometry alias changes observed road type: {key} -> {owner}')
    ownership_review = ownership_review or {}
    source_aliases = {int(k): int(v) for k,v in ownership_review.get('source_aliases',{}).items()}
    for key, owner in source_aliases.items():
        if observed[key]['lane_type'] != observed[owner]['lane_type']:
            raise ValueError(f'Reviewed source ownership changes travel mode: {key} -> {owner}')
        if not ownership_review.get('reason') or not ownership_review.get('evidence'):
            raise ValueError('Source ownership requires Agent reasoning and evidence')
    selected_aliases={**reviewed_aliases,**geometry_aliases,**source_aliases}
    neighbor_choices = {int(k):v for k,v in ownership_review.get('neighbor_choices',{}).items()}
    ownership_resolutions = []
    canonical, geometry_ids, aliases, evidence = {}, {}, {}, defaultdict(list)
    definitions, id_conflicts = {}, []
    for source, data in sources:
        for item in data["lane_segments"].values():
            key = item["id"]
            # Use the source fields as a stable textual identity.
            geometry_item = observed[key] if key in source_id_choices else item
            geometry = json.dumps([geometry_item[f] for f in GEOMETRY_FIELDS], sort_keys=True)
            if key in definitions and definitions[key] != geometry:
                id_conflicts.append({"source_lane_id": key, "source": source})
                continue
            definitions[key] = geometry
            reference = observed[selected_aliases[key]] if key in selected_aliases else geometry_item
            reference_geometry = json.dumps([reference[f] for f in GEOMETRY_FIELDS], sort_keys=True)
            owner = geometry_ids.setdefault(reference_geometry, reference["id"])
            aliases[key] = owner
            canonical.setdefault(owner, deepcopy(reference))
            evidence[owner].append({"source": source, "source_lane_id": key, "lane": item})
    neighbor_conflicts, external_neighbors = [], []
    for key, item in canonical.items():
        for field in ("successors", "predecessors"):
            item[field] = sorted({aliases.get(target, target) for observation in evidence[key]
                                  for target in observation["lane"][field]}-{key})
        for field in ("left_neighbor_id", "right_neighbor_id"):
            values = {aliases.get(value,value) for observation in evidence[key]
                      if (value := observation["lane"][field]) is not None}
            internal = values & canonical.keys()
            external = values - canonical.keys()
            choice = neighbor_choices.get(key,{}).get(field)
            if len(values) > 1 and choice is not None:
                original_neighbor = choice['source_neighbor_id']
                selected = aliases.get(int(original_neighbor),int(original_neighbor)) if original_neighbor is not None else None
                observed_absence = selected is None and observed[key][field] is None
                if (selected not in values and not observed_absence) or not choice.get('reason'):
                    raise ValueError(f'Ownership choice is not an observed neighbor: {key} {field}')
                item[field] = selected
                ownership_resolutions.append(dict(canonical_lane_id=key,field=field,selected_observed_lane=selected,
                    alternative_observed_lanes=sorted(values-{selected}),reason=choice['reason']))
            elif len(values) > 1 and len(internal) == 1:
                item[field] = next(iter(internal))
                external_neighbors.append({"canonical_lane_id":key, "field":field,
                                           "selected_observed_lane":item[field],
                                           "external_source_references":sorted(external)})
            elif len(values) > 1:
                neighbor_conflicts.append({"canonical_lane_id": key, "field": field,
                                           "candidate_ids": sorted(values)})
            elif values:
                item[field] = next(iter(values))
        item["source_observations"] = [dict(source=row["source"],source_lane_id=row["source_lane_id"],
            definition_differences={f:row["lane"][f] for f in GEOMETRY_FIELDS if row["lane"][f] != item[f]},
            observed_links={f:row["lane"][f] for f in ("predecessors","successors","left_neighbor_id","right_neighbor_id")})
            for row in evidence[key]]
    return {"lanes": list(canonical.values()), "aliases": {str(k):v for k,v in aliases.items() if k!=v},
            "source_id_geometry_conflicts": id_conflicts, "neighbor_conflicts": neighbor_conflicts,
            "source_id_marking_resolutions": source_id_resolutions,
            "reviewed_marking_aliases": reviewed_aliases,
            "reviewed_geometry_aliases": geometry_aliases,
            "reviewed_source_ownership": ownership_review,
            "source_ownership_neighbor_resolutions": ownership_resolutions,
            "external_neighbor_observations": external_neighbors,
            "scope": "Exact source equivalence only; approximate co-location and missing source height still require review."}


def load_city_observations(config, data_root):
    sources = []
    for relative in config["sources"]["av2_logs"]:
        path = next((Path(data_root)/relative/"map").glob("log_map_archive_*.json"))
        sources.append((str(path.relative_to(data_root)), json.loads(path.read_text())))
    return combine_exact_observations(sources)
