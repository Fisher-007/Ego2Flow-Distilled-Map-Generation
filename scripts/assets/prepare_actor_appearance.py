"""Estimate vehicle paint from calibrated image and body-height LiDAR samples."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import cv2,numpy as np,pandas as pd
from av2.utils.io import read_city_SE3_ego
from scripts.assets.prepare_observed_access import camera


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--history',type=Path,required=True)
    p.add_argument('--log-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--review',type=Path,help='External image-review corrections to color proposals; explicit source evidence required')
    p.add_argument('--frame-stride',type=int,default=10)
    p.add_argument('--max-lidar-age-s',type=float,default=.075,
                   help='Maximum absolute image/sweep separation; sparse missing sweeps remain unobserved')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    h=json.loads(a.history.read_text());K=np.array(h['K_av2']);origin=np.array(h['origin_city_xyz'])
    poses=read_city_SE3_ego(a.log_dir);sweeps=sorted((a.log_dir/'sensors/lidar').glob('*.feather'))
    times=np.array([int(x.stem) for x in sweeps],dtype=np.int64);samples=defaultdict(list);skipped=[]
    for index in range(0,len(h['frames']),a.frame_stride):
        row=h['frames'][index];j=int(np.argmin(abs(times-row['timestamp_ns'])))
        age=(int(times[j])-row['timestamp_ns'])/1e9
        if abs(age)>a.max_lidar_age_s:
            skipped.append(dict(history_index=index,nearest_sweep_difference_s=age));continue
        table=pd.read_feather(sweeps[j])
        local=poses[int(times[j])].transform_point_cloud(table[['x','y','z']].to_numpy())-origin
        if h.get('log_registration'):
            registration=h['log_registration']
            local=local@np.array(registration['rotation_matrix']).T+registration['translation_local_m']
        T=camera(h,index);cv=(local-T[:3,3])@T[:3,:3];uvh=cv@K.T
        valid=cv[:,2]>1.;uv=np.zeros((len(cv),2),int);uv[valid]=np.rint(uvh[valid,:2]/uvh[valid,2:]).astype(int)
        image=cv2.imread(row['source_image']);height,width=image.shape[:2]
        valid&=(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
        # A small projection depth buffer rejects returns behind other observed surfaces.
        bins=np.clip(uv[:,1]//5,0,(height-1)//5)*((width+4)//5)+np.clip(uv[:,0]//5,0,(width-1)//5)
        depth=np.full(((height+4)//5)*((width+4)//5),np.inf);np.minimum.at(depth,bins[valid],cv[valid,2])
        visible=valid&(cv[:,2]<=depth[bins]+.6)
        for actor in row['actors']:
            if actor['category'] not in ('REGULAR_VEHICLE','LARGE_VEHICLE','BOX_TRUCK','TRUCK','BUS','SCHOOL_BUS'):continue
            object_xyz=(local-actor['position_local_xyz_m'])@np.array(actor['city_rotation'])
            l,w,z=actor['dimensions_lwh_m']
            select=visible&(abs(object_xyz[:,0])<l*.48)&(abs(object_xyz[:,1])<w*.51)&(object_xyz[:,2]>-z*.22)&(object_xyz[:,2]<-z*.015)
            ids=np.flatnonzero(select)
            if len(ids)<12:continue
            pixels=image[uv[ids,1],uv[ids,0]];hsv=cv2.cvtColor(pixels.reshape(-1,1,3),cv2.COLOR_BGR2HSV)[:,0]
            keep=(hsv[:,2]>25)&(hsv[:,2]<245);pixels=pixels[keep];ids=ids[keep]
            if len(pixels)<12:continue
            samples[actor['track_uuid']].append({'index':index,'pixels':pixels,'uv':uv[ids],
                'image':row['source_image'],'source_dimensions':actor['dimensions_lwh_m'],
                'lidar_timestamp_ns':int(times[j]),'lidar_image_difference_s':age})
    result={};review=[]
    cv2.setRNGSeed(7)
    for track,rows in samples.items():
        # Weight views by projected body coverage.
        rows=sorted(rows,key=lambda r:len(r['pixels']),reverse=True)[:3]
        pixels=np.concatenate([r['pixels'] for r in rows]);lab=cv2.cvtColor(pixels.reshape(-1,1,3),cv2.COLOR_BGR2LAB)[:,0].astype(np.float32)
        _,labels,centers=cv2.kmeans(lab,3,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,.2),3,cv2.KMEANS_PP_CENTERS)
        counts=np.bincount(labels[:,0],minlength=3);winner=int(counts.argmax());selected=pixels[labels[:,0]==winner]
        # Use an upper-middle exposure estimate for body color.
        light=selected.mean(1);selected=selected[light>=np.quantile(light,.4)]
        bgr=np.median(selected,axis=0).round().astype(int);rgb=bgr[::-1].tolist()
        result[track]={'color_rgb':rgb,'support_points':len(pixels),'dominant_fraction':float(counts[winner]/len(pixels)),
            'source_frames':[{'history_index':r['index'],'image':r['image'],'points':len(r['pixels']),
                'lidar_timestamp_ns':r['lidar_timestamp_ns'],'lidar_image_difference_s':r['lidar_image_difference_s']} for r in rows],
            'scope':'Observed paint estimate, not calibrated material albedo; shape, windows and lighting remain asset-dependent.'}
        best=rows[0];im=cv2.imread(best['image']);uv=best['uv'];lo=np.maximum(uv.min(0)-35,0);hi=np.minimum(uv.max(0)+35,[im.shape[1]-1,im.shape[0]-1])
        crop=im[lo[1]:hi[1]+1,lo[0]:hi[0]+1];crop=cv2.resize(crop,(240,150))
        tile=np.full((190,240,3),255,np.uint8);tile[:150]=crop;tile[150:175]=bgr
        cv2.putText(tile,track[:8],(5,187),0,.4,(0,0,0),1);review.append(tile)
    for page in range(0,len(review),24):
        tiles=review[page:page+24];tiles += [np.full_like(tiles[0],255)]*((-len(tiles))%6)
        grid=np.vstack([np.hstack(tiles[i:i+6]) for i in range(0,len(tiles),6)])
        cv2.imwrite(str(a.output/f'color_review_{page//24}.jpg'),grid)
    if a.review:
        revisions=json.loads(a.review.read_text())
        for correction in revisions['corrections']:
            keys=[key for key in result if key.startswith(correction['track_prefix'])]
            if len(keys)!=1:raise ValueError('Color review must identify one observed track')
            row=result[keys[0]];row['automatic_color_rgb']=row['color_rgb']
            row['color_rgb']=correction['color_rgb'];row['visual_review']=correction
            for attribute in ('body_style','asset_family','required_body_features'):
                if correction.get(attribute):row[attribute]=correction[attribute]
    (a.output/'appearance.json').write_text(json.dumps({'tracks':result,'source_history':str(a.history),
        'skipped_unpaired_sweeps':skipped,'max_lidar_age_s':a.max_lidar_age_s,
        'method':'Source cuboid body-band LiDAR returns, calibrated RGB projection, depth visibility, dominant Lab cluster from strongest three views.',
        'review_sheets':'color_review images show automatic proposals before external corrections; final choices and source crops are retained separately.',
        'unobserved':'No inferred color assigned when fewer than 12 usable paint samples.'},indent=2)+'\n')
    print({'tracks_with_color_evidence':len(result),'review_pages':(len(review)+23)//24})


if __name__=='__main__':main()
