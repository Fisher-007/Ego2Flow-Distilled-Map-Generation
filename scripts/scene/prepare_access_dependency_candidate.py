"""Assemble road proposals and coupled building geometry for application checks."""
import argparse,json
from pathlib import Path
from scripts.scene.apply_access_group import apply


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("batch-root","feedback-selection","output"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument('--scene',type=Path,help='Current source road scene with updated building dependencies')
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text());manifest=read(a.batch_root/"manifest.json");selection=read(a.feedback_selection)["regions"]
    source=read(a.scene or manifest["source_scene"]);lanes={};paths=[];outputs=[];regions=[]
    for row in manifest["records"]:
        name=row["region"];choice=selection[name];condition=read(row["condition"]);candidate=read(choice["paths"]);review=read(choice["review"])
        if not candidate["passed"] or not review["passed"]:raise ValueError("Unreviewed projected candidate: "+name)
        lanes.update({l["id"]:l for l in condition["lanes"]if l["id"]in condition["generated_path_ids"]})
        paths.extend(condition["paths"]);outputs.extend(candidate["paths"]);regions.append(name)
    a.output.mkdir(parents=True,exist_ok=True)
    conditions=a.batch_root/"graph_conditions.json"
    combined=dict(region=dict(id="city_dependency_candidate"),lanes=list(lanes.values()),paths=paths,graph_conditions=str(conditions.resolve()))
    candidate,review,_=apply(source,combined,dict(paths=outputs,passed=True))
    review.update(regions=regions,selected_geometric_candidates=len(regions),accepted_map=False,
        purpose="Reconstruct road-dependent terrain/building foundations/passages and diagnose original width/occupancy conflicts. Does not override failed application checks.",
        required_next_validation="Normal apply_access_batch against source roads and realized dependencies; native road/terrain/building checks remain mandatory.")
    candidate["dependency_candidate_status"]=dict(accepted_map=False,application_passed=review["passed"],review=str((a.output/"dependency_candidate_review.json").resolve()))
    (a.output/"dependency_candidate_scene.json").write_text(json.dumps(candidate,separators=(",",":"))+"\n")
    (a.output/"dependency_candidate_review.json").write_text(json.dumps(review,indent=2)+"\n")
    print(dict(selected=len(regions),application_passed=review["passed"],accepted_map=False,changed_av2=len(review["changed_av2"]),internal_gaps=len(review["internal_piece_gaps"]),boundary_changes=len(review["boundary_changes"]),building_conflicts=sum(r["width_envelope_inside_building_m2"]>=.01 for r in review["building_review"])),flush=True)


if __name__=="__main__":main()
