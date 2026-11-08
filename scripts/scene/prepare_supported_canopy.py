"""Bind reviewed source canopy anchors to actual final terrain triangles."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import shapely
from scripts.scene.build_assets import plane_heights

def prepare(evidence, terrain):
    result=deepcopy(evidence)
    xyz=np.asarray(terrain['vertices'],dtype=float)
    faces=np.asarray(terrain['faces'],dtype=int)
    triangles=xyz[faces]
    lo,hi=triangles[:,:,:2].min(1),triangles[:,:,:2].max(1)
    accepted=[];unsupported=[];changes=[]
    for row in result['canopy_candidates']:
        point=np.asarray(row['position_enu_m'][:2])
        ids=np.flatnonzero(((lo<=point+1e-7)&(hi>=point-1e-7)).all(1))
        candidates=triangles[ids]
        if len(ids):
            normal=np.cross(candidates[:,1]-candidates[:,0],candidates[:,2]-candidates[:,0])
            candidates=candidates[np.abs(normal[:,2])>1e-10]
            candidates=candidates[shapely.dwithin(shapely.polygons(candidates[:,:,:2]),shapely.Point(point),1e-7)]
        if not len(candidates):
            unsupported.append(dict(id=row['id'],position_enu_m=row['position_enu_m'],reason='No final terrain top triangle at the retained source anchor'))
            continue
        heights=plane_heights(candidates,np.repeat(point[None],len(candidates),axis=0))
        if np.ptp(heights)>1e-4:
            raise ValueError('Multiple terrain levels at source canopy anchor: '+row['id'])
        z=float(np.mean(heights));height=float(row['observed_top_z_m']-z)
        if height<=0:
            unsupported.append(dict(id=row['id'],position_enu_m=row['position_enu_m'],ground_z_m=z,reason='Observed crown top is not above final terrain support'))
            continue
        changes.append(dict(id=row['id'],previous_z_m=row['position_enu_m'][2],final_z_m=z,
            height_m=height,triangle_support_spread_m=float(np.ptp(heights))))
        row['position_enu_m']=[*point.tolist(),z];row['target_height_m']=height
        row['native_support_method']='Exact final terrain top triangle plane at unchanged selected source XY'
        row['native_realization_status']='supported_native_asset_pending'
        accepted.append(row)
    result['canopy_candidates']=accepted
    result['unsupported_source_anchors']=unsupported
    result['support_review']=dict(input_selected=len(evidence['canopy_candidates']),supported=len(accepted),
        unsupported=len(unsupported),changes=changes,
        scope='Source XY and absolute observed crown tops are unchanged. Unsupported anchors remain in the source record; no tree is placed on a guessed bridge or moved off a road.')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['evidence','terrain','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();result=prepare(json.loads(a.evidence.read_text()),json.loads(a.terrain.read_text()))
    result['support_review'].update(source_evidence=str(a.evidence),terrain=str(a.terrain))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'support_review.json').write_text(json.dumps(result['support_review'],indent=2)+'\n')
    print({k:v for k,v in result['support_review'].items()if k!='changes'})
if __name__=='__main__':main()
