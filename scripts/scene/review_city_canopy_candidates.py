"""Project extracted canopy candidates into their actual calibrated source views."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from scripts.scene.prepare_city_actor_layers import group_area_views

CAMERA_ORDER = ['ring_front_left', 'ring_front_center', 'ring_front_right',
                'ring_side_left', 'ring_side_right', 'ring_rear_left', 'ring_rear_right']

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['canopy-root','output']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);records=[];source_counts={}
 for source in sorted(a.canopy_root.glob('*/extracted/evidence.json')):
  q=json.loads(source.read_text());scene=json.loads(Path(q['scene']).read_text());origin=np.asarray(scene['coordinates']['origin_city_xyz']);log=Path(q['log']);loader=AV2SensorDataLoader(log.parent,log.parent);registration=q.get('source_log_registration')
  source_counts[str(source)]=len(q['canopy_candidates'])
  for area,views in group_area_views(q['views']).items():
   rows=[];panels=[]
   for view in sorted(views,key=lambda v:CAMERA_ORDER.index(v['camera'])):
    real=cv2.imread(view['file']);camera=loader.get_log_pinhole_camera(log.name,view['camera']);ego=loader.get_city_SE3_ego(log.name,view['timestamp_ns']);scale=800/real.shape[1];panel=cv2.resize(real,(800,round(real.shape[0]*scale)));visible=[]
    for i,c in enumerate(q['canopy_candidates']):
     base=np.array(c['position_enu_m']);points=np.array([base,base+[0,0,.4*c['target_height_m']],base+[0,0,c['target_height_m']]])
     if registration:points=(points-np.array(registration['translation_local_m']))@np.array(registration['rotation_matrix'])
     city=points+origin;uv,cp,mask=camera.project_ego_to_img(ego.inverse().transform_point_cloud(city))
     if not mask[1]or cp[1,2]<.5:continue
     xy=np.rint(uv[1]*scale).astype(int)
     if not(0<=xy[0]<panel.shape[1]and 0<=xy[1]<panel.shape[0]):continue
     visible.append(dict(candidate=i,pixel=uv[1].tolist(),depth_m=float(cp[1,2])))
     if mask[0]and mask[2]:cv2.line(panel,tuple(np.rint(uv[0]*scale).astype(int)),tuple(np.rint(uv[2]*scale).astype(int)),(90,160,90),1)
     cv2.circle(panel,tuple(xy),3,(0,255,255),-1);cv2.putText(panel,str(i),tuple(xy+[3,-3]),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),3,cv2.LINE_AA);cv2.putText(panel,str(i),tuple(xy+[3,-3]),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1,cv2.LINE_AA)
    panel=cv2.copyMakeBorder(panel,32,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255));cv2.putText(panel,view['id']+' / '+str(len(visible))+' projected candidates',(8,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1,cv2.LINE_AA)
    path=a.output/(view['id']+'.jpg');cv2.imwrite(str(path),panel);rows.append(dict(view=view,image=str(path),visible=visible));panels.append(panel)
   h=max(x.shape[0]for x in panels);panels=[cv2.copyMakeBorder(x,0,h-x.shape[0],0,0,cv2.BORDER_CONSTANT,value=(255,255,255))for x in panels];sheet=np.concatenate(panels,axis=1);path=a.output/(area+'.jpg');cv2.imwrite(str(path),sheet);records.append(dict(area=area,log_id=log.name,evidence=str(source),source_registration=registration,candidate_count=len(q['canopy_candidates']),views=rows,sheet=str(path)))
 for i in range(0,len(records),2):
  selected=sorted(records,key=lambda q:q['area'])[i:i+2];images=[cv2.imread(q['sheet'])for q in selected];w=max(x.shape[1]for x in images);images=[cv2.copyMakeBorder(x,0,0,0,w-x.shape[1],cv2.BORDER_CONSTANT,value=(255,255,255))for x in images];cv2.imwrite(str(a.output/('sheet_%02d.jpg'%(i//2))),np.concatenate(images,axis=0))
 report=dict(records=sorted(records,key=lambda q:q['area']),source_log_count=len(source_counts),
  source_candidate_count=sum(source_counts.values()),
  candidate_reference='Each visible.candidate is the original zero-based index in record.evidence. Repeated Area/view projections share that source candidate; they do not create physical objects.',scope='Calibrated source projection only. Agent must distinguish tree crowns/trunks from facade returns and occlusion. Candidate location/extent is approximate; no candidate is selected automatically.')
 (a.output/'review.json').write_text(json.dumps(report,indent=2)+'\n');print(dict(logs=len(source_counts),areas=len(records),views=sum(len(q['views'])for q in records),candidates=sum(source_counts.values()),sheets=(len(records)+1)//2),flush=True)
if __name__=='__main__':main()
