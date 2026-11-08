"""Accumulate AV2 LiDAR and propose static context for image review.

Source poses align sweeps; annotated cuboids remove foreground objects.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.cuboid import CuboidList
from av2.utils.io import read_city_SE3_ego
from map_generation.coordinates import source_log_registration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--log-dir', type=Path, required=True)
    p.add_argument('--views', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    scene = json.loads(a.scene.read_text()); origin = np.array(scene['coordinates']['origin_city_xyz'])
    registration = source_log_registration(scene, a.log_dir.name)
    poses = read_city_SE3_ego(a.log_dir)
    annotations = pd.read_feather(a.log_dir/'annotations.feather')
    loader = AV2SensorDataLoader(a.log_dir.parent, a.log_dir.parent)
    ground = ArgoverseStaticMap.from_map_dir(a.log_dir/'map', build_raster=True).raster_ground_height_layer
    sweeps, points = [], []
    for path in sorted((a.log_dir/'sensors/lidar').glob('*.feather')):
        ts = int(path.stem); xyz = pd.read_feather(path)[['x','y','z']].to_numpy(dtype=float)
        rows = annotations[annotations.timestamp_ns == ts]
        keep = np.ones(len(xyz), dtype=bool)
        for cuboid in CuboidList.from_dataframe(rows).cuboids:
            keep &= ~cuboid.compute_interior_points(xyz)[1]
        city = poses[ts].transform_point_cloud(xyz[keep])
        points.append(city-origin)
        sweeps.append({'timestamp_ns':ts, 'points':len(xyz), 'annotated_foreground_removed':int((~keep).sum()), 'cuboids':len(rows)})
    xyz = np.concatenate(points)
    # 15 cm voxels suppress repeated returns without changing coordinates.
    _, idx = np.unique(np.floor(xyz/.15).astype(int), axis=0, return_index=True)
    xyz = xyz[idx]; city = xyz+origin
    ground_z = ground.get_ground_height_at_xy(city[:,:2])-origin[2]
    if registration:
        basis = np.asarray(registration['rotation_matrix'])
        translation = np.asarray(registration['translation_local_m'])
        ground_z = (np.c_[xyz[:,:2], ground_z] @ basis.T + translation)[:,2]
        xyz = xyz @ basis.T + translation
    height = xyz[:,2]-ground_z
    valid = np.isfinite(height)
    xyz, height, ground_z = xyz[valid], height[valid], ground_z[valid]
    np.savez_compressed(a.output/'static_points.npz', xyz=xyz, height_m=height)

    buildings = shapely.union_all([shapely.Polygon(b['exterior'], b.get('holes',[])).buffer(1.) for b in scene['buildings']])
    roads = shapely.union_all([(shapely.Polygon(l['left_boundary']+list(reversed(l['right_boundary']))).buffer(0) if 'left_boundary' in l else shapely.LineString(l['centerline']).buffer(l.get('width_m',3.2)/2)) for l in scene['fusion_map']['lanes']])
    high = (height>3.) & (height<28.) & ~shapely.contains(buildings, shapely.points(xyz[:,:2]))
    # Group nearby canopy returns. Ground/roof height and positions are measured;
    # the native tree species and individual tree decomposition remain approximate.
    canopy = xyz[high]; cells, counts = np.unique(np.floor(canopy[:,:2]/1.5).astype(int),axis=0,return_counts=True)
    cells = cells[counts>=8]; centers = (cells+.5)*1.5
    graph = cKDTree(centers).sparse_distance_matrix(cKDTree(centers),2.2,output_type='coo_matrix')
    _, labels = connected_components(graph, directed=False)
    candidates=[]
    for label in np.unique(labels):
        group=centers[labels==label]
        if len(group)<3:continue
        # Long connected crowns are represented by spaced existing tree assets.
        selected=[]
        for xy in group:
            if selected and np.linalg.norm(np.array(selected)-xy,axis=1).min()<6.:continue
            if shapely.distance(roads,shapely.Point(xy))<1.2:continue
            selected.append(xy)
            near=np.linalg.norm(xyz[:,:2]-xy,axis=1)<4.5
            observed=height[near & (height>3.) & (height<28.)]
            if len(observed)<50:continue
            z=(float(ground.get_ground_height_at_xy(np.array([xy+origin[:2]]))[0]-origin[2])
               if registration is None else float(np.median(ground_z[near])))
            ground_source='AV2 ground raster at candidate center'
            if registration:
                ground_source='Median of source-registered ground samples within 4.5m'
            if not np.isfinite(z):
                z=float(np.median(ground_z[near]));ground_source='Median of measured finite ground samples within 4.5m; center raster missing'
            h=float(np.quantile(observed,.96))
            candidates.append({'position_enu_m':[*xy.tolist(),z], 'target_height_m':h,
                'ground_source':ground_source, 'crown_radius_m':4.5, 'support_points':len(observed), 'cluster':int(label),
                'road_clearance_m':float(shapely.distance(roads,shapely.Point(xy)))})
    views=[]
    for view in json.loads(a.views.read_text())['views']:
        folder=a.output/view['id'];folder.mkdir(exist_ok=True)
        ts=view['timestamp_ns'];name=view['camera']
        cam=loader.get_log_pinhole_camera(a.log_dir.name,name)
        ego=loader.get_city_SE3_ego(a.log_dir.name,ts)
        uv,cp,mask=cam.project_ego_to_img(ego.inverse().transform_point_cloud(city[valid]))
        mask &= cp[:,2]>.5
        pix=np.rint(uv[mask]).astype(int);z=cp[mask,2]
        inside=(pix[:,0]>=0)&(pix[:,0]<cam.width_px)&(pix[:,1]>=0)&(pix[:,1]<cam.height_px)
        pix,z=pix[inside],z[inside]
        depth=np.full((cam.height_px,cam.width_px),np.inf,np.float32)
        np.minimum.at(depth,(pix[:,1],pix[:,0]),z)
        measured=np.isfinite(depth);depth[~measured]=0
        np.savez_compressed(folder/'lidar_depth.npz',depth_m=depth,observed=measured)
        inverse=np.zeros_like(depth);inverse[measured]=1/np.maximum(depth[measured],1.)
        image=np.rint(np.clip(inverse*255,0,255)).astype(np.uint8)
        cv2.imwrite(str(folder/'lidar_inverse_depth.png'),image)
        real=cv2.imread(str(a.log_dir/'sensors/cameras'/name/(str(ts)+'.jpg')))
        overlay=real.copy(); color=cv2.applyColorMap(np.rint(np.clip(depth/60,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)
        overlay[measured]=color[measured]
        cv2.imwrite(str(folder/'lidar_overlay.jpg'),overlay)
        views.append({**view,'observed_pixel_fraction':float(measured.mean()),'depth_limit':'Sparse measured camera-forward depth; zero means unknown, not sky/far. Do not feed this directly as dense Cosmos control.'})
    report={'scene':str(a.scene),'log':str(a.log_dir),'source_log_registration':registration,'sweeps':sweeps,'static_voxels':len(xyz),
        'canopy_candidates':candidates,'views':views,
        'limits':'Color/semantic verification and native asset placement required. These are canopy candidates, not individually identified trees. LiDAR does not observe every facade or recover albedo.'}
    (a.output/'evidence.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,8));sc=ax.scatter(xyz[::3,0],xyz[::3,1],c=height[::3],s=.2,cmap='viridis',vmin=0,vmax=20)
    for l in scene['lanes']:
        for side in ['left_boundary','right_boundary']:
            b=np.array(l[side]);ax.plot(b[:,0],b[:,1],color='gray',lw=.4)
    for i,c in enumerate(candidates):
        x,y,_=c['position_enu_m'];ax.text(x,y,str(i),fontsize=6,color='red')
    lo,hi=np.quantile(xyz[:,:2],[.01,.99],axis=0)
    ax.set_aspect('equal');ax.set(xlim=(lo[0]-10,hi[0]+10),ylim=(lo[1]-10,hi[1]+10),title='AV2 static LiDAR: height above ground and canopy candidates',xlabel='East (m)',ylabel='North (m)');fig.colorbar(sc,ax=ax,label='Height (m)');fig.tight_layout();fig.savefig(a.output/'canopy_evidence.png',dpi=160)
    print(json.dumps({'sweeps':len(sweeps),'static_voxels':len(xyz),'canopy_candidates':len(candidates),'views':views},indent=2))


if __name__=='__main__':main()
