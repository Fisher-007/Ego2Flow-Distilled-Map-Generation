"""Realize reviewed single-entry timing for selected source controller groups."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("network", "association", "conditions", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    root = ET.parse(a.network).getroot()
    source_geometry = [ET.tostring(x) for x in root if x.tag != "tlLogic"]
    spec = json.loads(a.conditions.read_text())
    groups = json.loads(a.association.read_text())["groups"]
    edges = {x.get("id"): x for x in root.findall("edge")}
    programs = {x.get("id"): x for x in root.findall("tlLogic")}
    rows = []
    for key in spec["controllers"]:
        nodes = set(groups[key.removeprefix("MAP_")]["source_nodes"])
        indices = {}
        for connection in root.findall("connection"):
            if connection.get("tl") != key:
                continue
            edge = edges[connection.get("from")]
            if edge.get("from") not in nodes and edge.get("to") in nodes:
                indices.setdefault(edge.get("id"), set()).add(int(connection.get("linkIndex")))
        program = programs[key]
        before = [dict(x.attrib) for x in program.findall("phase")]
        all_external = set().union(*indices.values())
        generated = []
        reviewed = spec.get("entry_link_groups", {}).get(key, {})
        releases = {}
        for source_edge, controlled in sorted(indices.items()):
            groups_for_edge = reviewed.get(source_edge, [sorted(controlled)])
            assigned = [i for group in groups_for_edge for i in group]
            if len(assigned) != len(set(assigned)) or set(assigned) != controlled:
                raise ValueError("Reviewed groups must partition actual external link indices: " + source_edge)
            for group_index, group in enumerate(groups_for_edge):
                if not group:
                    raise ValueError("Reviewed release group is empty: " + source_edge)
                name = source_edge if len(groups_for_edge) == 1 else source_edge + ":" + str(group_index)
                releases[name] = set(group)
        if set(reviewed) - set(indices):
            raise ValueError("Reviewed group names an absent external source edge")
        for source_edge, controlled in sorted(releases.items()):
            old = next(x for x in before if any(x["state"][i] in "Gg" for i in controlled))
            for state_name, seconds in (("G", spec["green_s"]), ("y", spec["yellow_s"]), ("r", spec["all_red_s"])):
                state = list(old["state"])
                for index in all_external:
                    state[index] = state_name if index in controlled else "r"
                generated.append(dict(duration=str(seconds), state="".join(state)))
        for child in list(program):
            if child.tag == "phase":
                program.remove(child)
        for phase in generated:
            ET.SubElement(program, "phase", phase)
        rows.append(dict(controller=key, source_entries={k: sorted(v) for k, v in indices.items()},
                         release_groups={k: sorted(v) for k, v in releases.items()},
                         previous_phases=before, phases=generated,
                         internal_link_states="Retained from the original SUMO phase serving this entry; only external entries are serialized. Native binding uses external entries."))
    if source_geometry != [ET.tostring(x) for x in root if x.tag != "tlLogic"]:
        raise ValueError("Signal timing revision changed source geometry or connections")
    a.output.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(a.output / "control.net.xml", encoding="utf-8", xml_declaration=True)
    report = dict(conditions=spec, source_network=str(a.network), controllers=rows,
                  source_geometry_and_connections_unchanged=True,
                  scope="Explicit inferred simulator timing for reviewed incompatible co-releases; not observed city signal timing.")
    (a.output / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(dict(revised_controllers=len(rows), retained_source_geometry=True), flush=True)


if __name__ == "__main__":
    main()
