"""Review generated access-road collisions against source building volumes."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import LineString,Polygon,box
from shapely.strtree import STRtree
from scripts.scene.apply_access_group import apply
from map_generation.building_passages import BuildingVolumes,realize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("batch-root","application","output"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--scene",type=Path,help="Same source roads with currently realized building dependencies")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.batch_root/"manifest.json").read_text())
    source_path=a.scene or Path(manifest["source_scene"]);source=json.loads(source_path.read_text());before={l["id"]:l for l in source["fusion_map"]["lanes"]}
    records=json.loads((a.application/"group_reviews.json").read_text())["records"]
    rejected=[r for r in records if r["status"]=="application_failed"]
    volumes=BuildingVolumes(source["buildings"]);conditions={r["region"]:r["condition"]for r in manifest["records"]}
    candidate=json.loads((a.application/"scene.json").read_text());candidate_lanes={l["id"]:l for l in candidate["fusion_map"]["lanes"]}
    diagnostics=[];prepared={}
    def contacts(lane,volumes):
        points=np.asarray(lane["centerline"]);line=LineString(points[:,:2]);envelope=line.buffer(lane["width_m"]/2,cap_style="flat");out=[]
        for i in volumes.tree.query(envelope,predicate="intersects"):
            b=volumes.buildings[int(i)]
            if b["base_z_m"]>=points[:,2].max()+2.2 or b["roof_z_m"]<=points[:,2].min()-.01:continue
            shape=volumes.shapes[int(i)];area=envelope.intersection(shape).area
            if area<.01:continue
            out.append(dict(building_id=b["id"],parent=b.get("passage_parent_id"),tags=b["tags"],base_z_m=b["base_z_m"],roof_z_m=b["roof_z_m"],road_z_range_m=[float(points[:,2].min()),float(points[:,2].max())],overlap_m2=float(area),centerline_inside_m=float(line.intersection(shape).length)))
        return out
    for row in rejected:
        c=json.loads(Path(conditions[row["region"]]).read_text());paths=json.loads(Path(row["paths"]).read_text())
        updated,review,_=apply(source,c,paths,volumes);after={l["id"]:l for l in updated["fusion_map"]["lanes"]};keys={q["lane_id"]for path in c["paths"]for q in path["segments"]}
        candidate_lanes.update({k:after[k]for k in keys});prepared[row["region"]]=(c,paths,after,keys)
        diagnostics.append(dict(region=row["region"],source_condition=conditions[row["region"]],candidate_paths=row["paths"],application_passed=review["passed"],other_failures={k:review.get(k)for k in ("internal_piece_gaps","boundary_changes","changed_av2")},lanes=[dict(id=k,tags=before[k].get("source_tags",{}),source=before[k]["source"],permissions=before[k]["allowed_classes"],original_contacts=contacts(before[k],volumes),candidate_contacts=contacts(after[k],volumes))for k in sorted(keys)if contacts(after[k],volumes)]))
        print(dict(region=row["region"],collision_lanes=len(diagnostics[-1]["lanes"])),flush=True)
    candidate["fusion_map"]["lanes"]=[candidate_lanes[l["id"]]for l in candidate["fusion_map"]["lanes"]]
    refreshed,passage_review=realize(candidate)
    (a.output/"passage_refresh_review.json").write_text(json.dumps(passage_review,indent=2)+"\n")
    # Write the building candidate for subsequent road application and native review.
    building_source=dict(source,buildings=refreshed["buildings"],building_passage_source_buildings=refreshed["building_passage_source_buildings"],building_passage_review=passage_review)
    (a.output/"source_with_candidate_passages.json").write_text(json.dumps(building_source,separators=(",",":"))+"\n")
    new_volumes=BuildingVolumes(refreshed["buildings"])
    for item in diagnostics:
        c,paths,after,keys=prepared[item["region"]];_,review,_=apply(building_source,c,paths,new_volumes)
        item["after_source_passage_refresh"]=dict(passed=review["passed"],building_overlaps=[r for r in review["building_review"]if r["width_envelope_inside_building_m2"]>=.01],internal_piece_gaps=review["internal_piece_gaps"],boundary_changes=review["boundary_changes"])
    shapes=[LineString(np.asarray(l["centerline"])[:,:2])for l in before.values()];road_tree=STRtree(shapes);all_lanes=list(before.values())
    for start in range(0,len(diagnostics),4):
        fig,axes=plt.subplots(2,2,figsize=(16,14));
        for ax,item in zip(axes.flat,diagnostics[start:start+4]):
            c,paths,after,keys=prepared[item["region"]];points=np.concatenate([np.asarray(after[k]["centerline"])[:,:2]for k in keys]);lo=points.min(axis=0)-5;hi=points.max(axis=0)+5;bounds=box(*lo,*hi)
            for i in volumes.tree.query(bounds,predicate="intersects"):
                poly=volumes.shapes[int(i)];x,y=poly.exterior.xy;ax.fill(x,y,color="#d59c95",alpha=.45);b=source["buildings"][int(i)];center=poly.representative_point();ax.text(center.x,center.y,str(b["id"]).split(":")[0],fontsize=5,clip_on=True)
            for i in road_tree.query(bounds,predicate="intersects"):
                q=np.asarray(all_lanes[int(i)]["centerline"]);ax.plot(q[:,0],q[:,1],color="#b5b5b5",lw=.7)
            for k in keys:
                old=np.asarray(before[k]["centerline"]);new=np.asarray(after[k]["centerline"]);ax.plot(old[:,0],old[:,1],color="green",lw=1.6,alpha=.8);ax.plot(new[:,0],new[:,1],color="#1769aa",lw=1.5);ax.scatter(new[[0,-1],0],new[[0,-1],1],s=7,c="#1769aa")
            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect("equal");ax.set_title(item["region"]+" | passage refresh: "+str(item["after_source_passage_refresh"]["passed"]));ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(diagnostics[start:start+4]):]:ax.axis("off")
        fig.suptitle("Actual generated road vs original source: blue candidate, green source, red building footprint (height detailed in JSON)");fig.tight_layout();fig.savefig(a.output/f"obstacles_{start//4:02d}.png",dpi=130);plt.close(fig)
    report=dict(source_scene=str(source_path),application=str(a.application),configured_regions=len(manifest["records"]),rejected_application_regions=len(rejected),after_source_passage_refresh_passed=sum(x["after_source_passage_refresh"]["passed"]for x in diagnostics),records=diagnostics,scope="Diagnostic geometry and source-tagged passage dependency refresh, not acceptance of all roads or removal of building evidence.")
    (a.output/"review.json").write_text(json.dumps(report,indent=2)+"\n");print({k:v for k,v in report.items()if k!="records"},flush=True)


if __name__=="__main__":main()
