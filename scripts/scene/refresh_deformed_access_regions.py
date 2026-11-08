"""Transport reviewed inference intervals through an explicit source deformation."""
import argparse,json
from copy import deepcopy
from pathlib import Path
import numpy as np
from map_generation.access_fusion import section


def refresh(previous,scene,regions):
    old={l['id']:l for l in previous['fusion_map']['lanes']};new={l['id']:l for l in scene['fusion_map']['lanes']}
    out=deepcopy(regions);changed=[];mappings={}
    for key in old:
        a=np.asarray(old[key]['centerline']);b=np.asarray(new[key]['centerline'])
        if a.shape!=b.shape:raise ValueError('Explicit vertex correspondence required for '+key)
        if np.array_equal(a[:,:2],b[:,:2]):continue
        mappings[key]=(np.r_[0.,np.linalg.norm(np.diff(a[:,:2],axis=0),axis=1).cumsum()],
                       np.r_[0.,np.linalg.norm(np.diff(b[:,:2],axis=0),axis=1).cumsum()])
    for region in out['regions']:
        edits=[]
        for row in region['intervals']:
            key=row['lane_id']
            if key not in mappings:continue
            a,b=mappings[key];before=[row['start_m'],row['end_m']]
            row['start_m'],row['end_m']=map(float,np.interp(before,a,b))
            row['length_m']=float(b[-1])
            edits.append(dict(lane_id=key,before_interval_m=before,after_interval_m=[row['start_m'],row['end_m']]))
        if edits:
            points=np.vstack([section(new[row['lane_id']]['centerline'],row['start_m'],row['end_m'])for row in region['intervals']])[:,:2]
            region['bbox_local_xy']=list(map(float,[*points.min(axis=0),*points.max(axis=0)]));region['extent_xy_m']=(points.max(axis=0)-points.min(axis=0)).tolist()
            region['mutable_length_m']=sum(row['end_m']-row['start_m']for row in region['intervals'])
            region['source_deformation_interval_transport']=edits;changed.append(region['id'])
    report=dict(regions=len(out['regions']),changed_region_scopes=changed,deformed_source_lanes=len(mappings),
        preserved_region_ids=True,preserved_seed_and_fixed_port_identities=True,
        method='Piecewise chord-station correspondence on the same explicitly deformed source vertices; transport prior complete semantic interval scopes without reselection, retain all original region/seed/fixed-port identities, recompute actual region bounds.')
    out['source_deformation_refresh']=report;out.pop('reuse_reason',None);return out,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['previous-scene','scene','regions','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(p.read_text());out,review=refresh(read(a.previous_scene),read(a.scene),read(a.regions));out['source_scene']=str(a.scene.resolve())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(review,flush=True)
if __name__=='__main__':main()
