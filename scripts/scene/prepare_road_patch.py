"""Prepare a local native road replacement from a saved physical-contact review."""
import argparse,json
from pathlib import Path
import orjson
from map_generation.surface_patch import prepare_road_patch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('previous-roads','roads','assets','contact-review','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();read=lambda path:orjson.loads(path.read_bytes())
    before,after,assets,review=map(read,(a.previous_roads,a.roads,a.assets,a.contact_review))
    touched={r['id']for r in review['patches']}|{key for r in review['patches']for key in r['original_mesh_ids']}
    result,report=prepare_road_patch(before,after,assets,touched)
    report.update(previous_roads=str(a.previous_roads),roads=str(a.roads),previous_assets=str(a.assets),contact_review=str(a.contact_review))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'road_meshes.json').write_bytes(orjson.dumps(result))
    (a.output/'patch.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'scene.json').write_text(json.dumps(dict(buildings=[],source_scene=assets['scene']))+'\n')
    (a.output/'terrain_mesh.json').write_text(json.dumps(dict(vertices=[],faces=[],report=dict(scope='Road-only patch; native terrain retained for the contact trial.')))+'\n')
    print(dict(changed_meshes=len(report['changed_mesh_ids']),tiles=len(report['replace_surface_names']),triangles=report['triangles']),flush=True)


if __name__=='__main__':main()
