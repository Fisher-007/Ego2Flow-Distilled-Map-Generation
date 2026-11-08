"""Estimate canopy asset anchors from local LiDAR height peaks and source views."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree
import shapely


def read(path):return json.loads(Path(path).read_text())
def save(path,data):path.write_text(json.dumps(data,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('scene','selection','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    scene=read(a.scene);selection=read(a.selection)
    buildings=shapely.union_all([shapely.Polygon(b['exterior'],b.get('holes',[])).buffer(1.) for b in scene['buildings']])
    road=shapely.union_all([(shapely.Polygon(l['left_boundary']+list(reversed(l['right_boundary']))).buffer(0)
        if 'left_boundary' in l else shapely.LineString(l['centerline']).buffer(l['width_m']/2))
        for l in scene['fusion_map']['lanes']])
    all_rows=[];rejected=[];area_records=[]
    for record in selection['records']:
        evidence=read(record['evidence']);data=np.load(Path(record['evidence']).with_name('static_points.npz'))
        xyz=data['xyz'];height=data['height_m'];selected=record['candidates']
        folder=a.output/record['log_id']/'extracted';folder.mkdir(parents=True,exist_ok=True)
        if not selected:
            evidence['canopy_candidates']=[];save(folder/'evidence.json',evidence)
            area_records.append(dict(area=record['area'],selected=0,refined=0));continue
        # Restrict CHM/peak search to image-qualified neighborhoods, retaining
        # observed local coordinates and ground estimates after registration.
        anchors=np.array([r['position_enu_m'][:2] for r in selected])
        near=cKDTree(anchors).query(xyz[:,:2])[0]<7.
        keep=near&(height>3)&(height<28)&~shapely.contains(buildings,shapely.points(xyz[:,:2]))
        high=xyz[keep];h=height[keep]
        if not len(high):
            rejected.extend(dict(area=record['area'],candidate_id=r['candidate_id'],reason='No qualified static crown returns outside source building envelope')for r in selected)
            evidence['canopy_candidates']=[];save(folder/'evidence.json',evidence);continue
        cell=.5;origin=np.floor(high[:,:2].min(0)/cell)*cell
        ij=np.floor((high[:,:2]-origin)/cell).astype(int);size=ij.max(0)+1
        chm=np.zeros(size);counts=np.zeros(size,dtype=int)
        np.maximum.at(chm,(ij[:,0],ij[:,1]),h);np.add.at(counts,(ij[:,0],ij[:,1]),1)
        observed=counts>0;weight=gaussian_filter(observed.astype(float),1.)
        smooth=gaussian_filter(chm,1.)/np.maximum(weight,1e-9)
        cells=np.argwhere((counts>=2)&(smooth>=3.5)&(smooth>=maximum_filter(smooth,size=7)))
        peaks=origin+(cells+.5)*cell
        valid=(shapely.distance(road,shapely.points(peaks))>=1.2)&~shapely.contains(buildings,shapely.points(peaks))
        peaks=peaks[valid];tree=cKDTree(high[:,:2]);rows=[]
        for source in selected:
            center=np.array(source['position_enu_m'][:2]);distance=np.linalg.norm(peaks-center,axis=1)
            eligible=np.flatnonzero(distance<=4.5)
            if len(eligible):center=peaks[min(eligible,key=lambda i:distance[i])]
            if shapely.distance(road,shapely.Point(center))<1.2 or buildings.covers(shapely.Point(center)):
                rejected.append(dict(area=record['area'],candidate_id=source['candidate_id'],reason='Approximate stem anchor lacks road/building clearance',source=source));continue
            indices=tree.query_ball_point(center,4.5)
            if len(indices)<30:
                rejected.append(dict(area=record['area'],candidate_id=source['candidate_id'],reason='Fewer than30 local static crown returns',source=source));continue
            local=h[indices];top=float(np.quantile(local,.99))
            ground=float(np.median(high[indices,2]-local))
            row=dict(position_enu_m=[*center.tolist(),ground],target_height_m=top,
                observed_top_z_m=float(np.quantile(high[indices,2],.99)),
                native_height_rule='Use observed absolute crown top minus verified final support Z; provisional source-raster-relative height is not native support',
                crown_radius_m=float(np.clip(np.quantile(np.linalg.norm(high[indices,:2]-center,axis=1),.9),1.,7.)),
                support_points=len(indices),context_kind='lidar_supported',
                source_area=record['area'],source_log=record['log_id'],
                source_candidate_ids=[source['candidate_id']],source_evidence=record['evidence'],
                actual_viewed_sheet=record['actual_viewed_sheet'],source_finding=record['finding'],
                source_views=evidence['views'],ground_source='Median of registered local static point Z minus source ground-relative height',
                road_clearance_m=float(shapely.distance(road,shapely.Point(center))),
                building_clearance_m=float(shapely.distance(buildings,shapely.Point(center))),
                anchor_displacement_m=float(np.linalg.norm(center-np.array(source['position_enu_m'][:2]))),
                placement_scope='Approximate crown-peak anchor; unobserved stem/species and crown shape require native review')
            duplicate=next((r for r in rows if np.linalg.norm(np.array(r['position_enu_m'][:2])-center)<3.),None)
            if duplicate:
                duplicate['source_candidate_ids'].extend(row['source_candidate_ids'])
                duplicate.setdefault('merged_envelopes',[]).append(row)
            else:rows.append(row)
        evidence['canopy_candidates']=rows
        evidence['refinement']=dict(source_selection=str(a.selection),source_scene=str(a.scene),
            method='Selected-neighborhood source CHM peaks; registered local ground median; nearby peaks within3m consolidated; native asset approximation remains explicit.')
        save(folder/'evidence.json',evidence);all_rows.extend(rows)
        area_records.append(dict(area=record['area'],selected=len(selected),refined=len(rows),evidence=str(folder/'evidence.json')))
        print(area_records[-1],flush=True)
    # Prefer bare or dormant observations for repeated views of the same crown.
    def bare(row):return any(t in row['source_finding'].lower() for t in ('bare','dormant'))
    consolidated=[]
    for row in sorted(all_rows,key=lambda r:(not bare(r),-r['support_points'])):
        center=np.array(row['position_enu_m'][:2])
        duplicate=next((r for r in consolidated if np.linalg.norm(np.array(r['position_enu_m'][:2])-center)<3.),None)
        if duplicate:duplicate.setdefault('alternate_source_observations',[]).append(row)
        else:consolidated.append(row)
    report=dict(canopy_candidates=consolidated,source_selection=str(a.selection),source_scene=str(a.scene),
        areas=area_records,selected_source_candidates=sum(len(r['candidates'])for r in selection['records']),
        refined_per_log_candidates=len(all_rows),consolidated_candidates=len(consolidated),rejections=rejected,
        source_epoch_policy=selection.get('source_epoch_policy'),
        scope='Only visually qualified tree neighborhoods. Species, unseen crown geometry and precise stems are not source facts. Native collision, support, visibility and actual seasonal assets still require review.',
        method_reference='https://r-lidar.github.io/lidRbook/itd.html')
    save(a.output/'evidence.json',report)
    print({k:v for k,v in report.items() if k not in ('canopy_candidates','areas','rejections')},flush=True)


if __name__=='__main__':main()
