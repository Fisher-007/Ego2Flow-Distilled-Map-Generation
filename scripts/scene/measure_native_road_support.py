"""Measure native waypoint support against physical road meshes.

Prefer covering triangles and source ownership. Uncovered points use a 0.1 mm
proximity bound; competing height layers are included in the report.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import shapely
from map_generation.av2_surfaces import mesh_triangles, plane_z


def summarize(rows):
    valid=[r for r in rows if r.get('surface_z_m') is not None]
    residual=np.asarray([abs(r['surface_minus_native_z_m'])for r in valid])
    return dict(samples=len(rows),supported=len(valid),missing=len(rows)-len(valid),
        ownership_matched=sum(r['selection']=='source_owner'for r in valid),
        diagnostic_fallback=sum(r['selection']=='nearest_height_diagnostic'for r in valid),
        abs_z_quantiles_0_50_90_99_100_m=np.quantile(residual,[0,.5,.9,.99,1]).tolist() if len(valid) else [],
        above_0_05_m=int((residual>.05).sum()),above_0_10_m=int((residual>.10).sum()),
        above_0_20_m=int((residual>.20).sum()),above_0_50_m=int((residual>.50).sum()),
        competing_layers_above_0_10_m=sum(r['candidate_z_spread_m']>.1 for r in valid),
        worst=sorted(valid,key=lambda r:abs(r['surface_minus_native_z_m']),reverse=True)[:30],
        missing_examples=[r for r in rows if r.get('surface_z_m') is None][:30])


def measure(xodr,mapping_file,roads_file,output,step_m=2.,samples_only=False,native_samples=None):
    import carla
    output.mkdir(parents=True,exist_ok=True)
    if native_samples is None:
        mapping=json.loads(mapping_file.read_text())
        intervals=mapping['lane_to_road_intervals']
        specs={}
        for source,records in intervals.items():
            for r in records:
                low,high=sorted([r['road_s_start_m'],r['road_s_end_m']])
                if high<=low:continue
                grid=np.linspace(low,high,max(2,int(np.ceil((high-low)/step_m))+1))
                epsilon=min(.0001,(high-low)/10)
                grid[0]+=epsilon;grid[-1]-=epsilon
                # CARLA converts station arguments to float32; stay inside the source interval.
                grid[-1]=min(grid[-1],float(np.nextafter(np.float32(high),np.float32(-np.inf))))
                for s in grid:
                    key=(int(r['road_id']),int(r['lane_id']),round(float(s),8))
                    specs.setdefault(key,set()).add(source)
        del mapping,intervals
        print('Native sampling specifications',len(specs),flush=True)
        native=carla.Map('physical_support_measurement',xodr.read_text())
        samples=[];null=[]
        for (rid,lid,s),sources in sorted(specs.items()):
            w=native.get_waypoint_xodr(rid,lid,s)
            if w is None:
                null.append(dict(road_id=rid,lane_id=lid,s_m=s,source_lane_ids=sorted(sources)));continue
            p=w.transform.location
            samples.append(dict(road_id=rid,lane_id=lid,s_m=s,source_lane_ids=sorted(sources),
                                native_xyz_m=[p.x,-p.y,p.z]))
        del native,specs
        (output/'native_waypoints.json').write_text(json.dumps(samples,separators=(',',':'))+'\n')
        (output/'native_lookup_missing.json').write_text(json.dumps(null,separators=(',',':'))+'\n')
    else:
        samples=json.loads(native_samples.read_text())
        null=json.loads((native_samples.parent/'native_lookup_missing.json').read_text())
    if samples_only:
        print(dict(native_samples=len(samples),missing=len(null),step_m=step_m),flush=True)
        return
    roads=json.loads(roads_file.read_text());meshes=roads['meshes']
    triangles=[];owners=[];metadata=[]
    for i,m in enumerate(meshes):
        tri=mesh_triangles(m)
        normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        tri=tri[abs(normal[:,2])>1e-14]
        triangles.append(tri);owners.extend([i]*len(tri))
        sources=set(m.get('source_lane_ids',[]));sources.add(m.get('source_lane_id'))
        sources.update([m.get('from_source_lane_id'),m.get('to_source_lane_id')])
        metadata.append(dict(id=m['id'],kind=m.get('kind'),sources=sources))
    triangles=np.concatenate(triangles);owners=np.asarray(owners,dtype=np.int32)
    del meshes,roads
    tree=shapely.STRtree(shapely.polygons(triangles[:,:,:2]))
    print('Physical support triangles',len(triangles),flush=True)
    for start in range(0,len(samples),20000):
        batch=samples[start:start+20000];xyz=np.asarray([r['native_xyz_m']for r in batch])
        points=shapely.points(xyz[:,:2])
        pairs=tree.query(points,predicate='covered_by')
        missing=np.ones(len(batch),dtype=bool)
        missing[pairs[0]]=False
        if missing.any():
            # Apply proximity matching only where no triangle covers the waypoint.
            indices=np.flatnonzero(missing)
            nearby=tree.query(points[indices],predicate='dwithin',distance=1e-4)
            pairs=np.concatenate([pairs,np.vstack([indices[nearby[0]],nearby[1]])],axis=1)
        heights=plane_z(triangles[pairs[1]],xyz[pairs[0],:2])
        by_point=[[]for r in batch]
        for point,face,z in zip(pairs[0],pairs[1],heights):
            by_point[point].append((int(owners[face]),float(z)))
        for row,candidates in zip(batch,by_point):
            if not candidates:
                row.update(surface_z_m=None,selection='missing');continue
            sources=set(row['source_lane_ids'])
            own=[c for c in candidates if sources&metadata[c[0]]['sources']]
            pool=own or candidates
            selected=min(pool,key=lambda c:abs(c[1]-row['native_xyz_m'][2]))
            index,z=selected
            zs=[v for _,v in candidates]
            row.update(surface_z_m=z,surface_minus_native_z_m=z-row['native_xyz_m'][2],
                mesh_id=metadata[index]['id'],mesh_kind=metadata[index]['kind'],
                selection='source_owner'if own else 'nearest_height_diagnostic',
                candidate_z_min_m=min(zs),candidate_z_max_m=max(zs),candidate_z_spread_m=max(zs)-min(zs),
                candidate_mesh_ids=sorted({metadata[i]['id']for i,_ in candidates}))
        print('Measured',min(start+len(batch),len(samples)),flush=True)
    (output/'native_support_samples.json').write_text(json.dumps(samples,separators=(',',':'))+'\n')
    report=dict(xodr=str(xodr),mapping=str(mapping_file),roads=str(roads_file),step_m=step_m,
        summary=summarize(samples),native_lookup_missing=len(null),native_lookup_missing_examples=null[:30],
        by_mesh_kind={kind:summarize([r for r in samples if r.get('mesh_kind')==kind])
                      for kind in sorted({r['mesh_kind']for r in samples if 'mesh_kind'in r},key=str)},
        scope=__doc__,coordinates='source local ENU; CARLA native Y inverted')
    (output/'native_lookup_missing.json').write_text(json.dumps(null,separators=(',',':'))+'\n')
    (output/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report['summary'].items()if k not in ['worst','missing_examples']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['xodr','mapping','roads','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--step-m',type=float,default=2.)
    p.add_argument('--samples-only',action='store_true')
    p.add_argument('--native-samples',type=Path,help='Explicit previous waypoint cache from exactly the same OpenDRIVE and sampling step')
    a=p.parse_args();measure(a.xodr,a.mapping,a.roads,a.output,a.step_m,a.samples_only,a.native_samples)


if __name__=='__main__':main()
