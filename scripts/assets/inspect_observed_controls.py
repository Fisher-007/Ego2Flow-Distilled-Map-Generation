"""Retrieve calibrated source views of annotated controls for agent review."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from scripts.assets.prepare_actor_layers import source_tracks,save
from scripts.scene.prepare_dynamic_replay import vertices


def inspect_controls(scene, log_dir, output):
    from types import SimpleNamespace
    from map_generation.coordinates import source_log_registration, register_local_pose
    a=SimpleNamespace(log_dir=Path(log_dir),output=Path(output))
    a.output.mkdir(parents=True,exist_ok=True)
    s=scene;origin=np.asarray(s['coordinates']['origin_city_xyz'])
    registration=source_log_registration(scene,a.log_dir.name)
    tracks,_=source_tracks(a.log_dir,origin);loader=AV2SensorDataLoader(a.log_dir.parent,a.log_dir.parent)
    controls={k:v for k,v in tracks.items() if v[0]['category']=='STOP_SIGN'}
    results=[]
    for track,rows in controls.items():
        candidates=[]
        pos=np.median([r['position_local_xyz_m'] for r in rows],axis=0)
        dims=np.median([r['dimensions_lwh_m'] for r in rows],axis=0)
        rotation=np.array(rows[len(rows)//2]['city_rotation'])
        for folder in sorted((a.log_dir/'sensors/cameras').glob('ring_*')):
            camera=loader.get_log_pinhole_camera(a.log_dir.name,folder.name);best=None
            for path in sorted(folder.glob('*.jpg')):
                t=int(path.stem)
                if t<rows[0]['timestamp_ns'] or t>rows[-1]['timestamp_ns']:continue
                pose=loader.get_city_SE3_ego(a.log_dir.name,t).compose(camera.ego_SE3_cam)
                xyz=(vertices(pos,rotation,dims)+origin-pose.translation)@pose.rotation
                if xyz[:,2].min()<=.1:continue
                uv=xyz@camera.intrinsics.K.T;uv=uv[:,:2]/uv[:,2:3]
                lo=np.maximum(uv.min(0),[0,0]);hi=np.minimum(uv.max(0),[camera.width_px-1,camera.height_px-1])
                if (hi<=lo).any():continue
                area=float(np.prod(hi-lo))
                if best is None or area>best['box_area_px']:
                    best={'camera':folder.name,'timestamp_ns':t,'source_image':str(path),'box_xyxy_px':[*lo.tolist(),*hi.tolist()],'box_area_px':area}
            if best:candidates.append(best)
        selected=sorted(candidates,key=lambda c:c['box_area_px'],reverse=True)[:3]
        for index,row in enumerate(selected):
            im=cv2.imread(row['source_image']);x1,y1,x2,y2=row['box_xyxy_px']
            cx,cy=(x1+x2)/2,(y1+y2)/2;span=max(x2-x1,y2-y1,180)*1.8
            left,top=max(0,int(cx-span/2)),max(0,int(cy-span/2));right,bottom=min(im.shape[1],int(cx+span/2)),min(im.shape[0],int(cy+span/2))
            cv2.rectangle(im,(int(x1),int(y1)),(int(x2),int(y2)),(0,120,255),2)
            filename=f'{track[:8]}_{index}_{row["camera"]}.jpg';cv2.imwrite(str(a.output/filename),im[top:bottom,left:right])
            row['evidence_crop']=filename
        map_pos,map_rotation=register_local_pose(pos,rotation,registration) if registration else (pos,rotation)
        results.append({'source_id':'object:'+track,'track_uuid':track,'category':'STOP_SIGN',
            'first_timestamp_ns':int(rows[0]['timestamp_ns']),'last_timestamp_ns':int(rows[-1]['timestamp_ns']),
            'median_position_local_xyz_m':map_pos.tolist(),'dimensions_lwh_m':dims.tolist(),
            'city_rotation':map_rotation.tolist(),'views':selected,
            'source_position_local_xyz_m':pos.tolist(),'source_city_rotation':rotation.tolist(),
            'log_registration':registration})
    save(a.output/'controls.json',{'log_id':a.log_dir.name,'controls':results,'scope':'Retrieved observed control cuboids and images. Annotation category and nearby lanes do not by themselves establish the controlled approach or sign-face direction.'})
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--log-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();results=inspect_controls(json.loads(a.scene.read_text()),a.log_dir,a.output)
    print(json.dumps({'controls':len(results),'views':sum(len(r['views']) for r in results),'output':str(a.output)}))


if __name__=='__main__':main()
