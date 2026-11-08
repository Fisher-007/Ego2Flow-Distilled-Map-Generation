"""Encode dense source height feedback on an unchanged native plan curve.

The canonical source may add collinear height knots. Keep the compiler's original
source XY samples and exact fitted XY polynomials, and carry the dense heights
as separate native arc-length elevation polynomials.
"""
import argparse,json
from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy.interpolate import PPoly
from map_generation.road_builder import fit_lane,chord_station_to_arc

def stations(points):
    return np.r_[0.,np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1).cumsum()]

def adapt(before,projected,selection):
    result=deepcopy(before)
    old={l['id']:l for l in before['fusion_map']['lanes']}
    target={l['id']:l for l in projected['fusion_map']['lanes']}
    output={l['id']:l for l in result['fusion_map']['lanes']}
    records=[]
    for key in selection['source_lane_ids']:
        lane=old[key];new=target[key]
        if lane['source']=='av2':raise ValueError('Dense inferred-height adaptation cannot alter observed AV2: '+key)
        if lane['successors']!=new['successors']:raise ValueError('Source topology changed: '+key)
        original=np.asarray(lane['centerline'],float);dense=np.asarray(new['centerline'],float)
        a=original[np.r_[True,np.linalg.norm(np.diff(original[:,:2],axis=0),axis=1)>1e-7]]
        b=dense[np.r_[True,np.linalg.norm(np.diff(dense[:,:2],axis=0),axis=1)>1e-7]]
        sa=stations(a);sb=stations(b)
        trace_error=float(np.linalg.norm(b[:,:2]-np.column_stack([np.interp(sb,sa,a[:,i])for i in range(2)]),axis=1).max())
        original_error=float(np.linalg.norm(a[:,:2]-np.column_stack([np.interp(sa,sb,b[:,i])for i in range(2)]),axis=1).max())
        if max(trace_error,original_error,abs(sa[-1]-sb[-1]))>1e-7:raise ValueError('Height projection changed source XY trace: '+key)
        geometry,_=fit_lane(lane,1.)
        arc=np.array([chord_station_to_arc(lane,geometry,s,1.)for s in sb]);arc[0]=0.;arc[-1]=geometry.length
        slopes=np.diff(b[:,2])/np.diff(arc)
        elevation=PPoly(np.vstack([np.zeros_like(slopes),np.zeros_like(slopes),slopes,b[:-1,2]]),arc)
        rows=np.column_stack([arc[:-1],elevation.c[::-1].T])
        encoded=output[key];points=original.copy();points[:,2]=np.interp(stations(original),sb,b[:,2])
        encoded['centerline']=points.tolist()
        encoded['parametric_xy_segments']=[c.coefficients.tolist()for c in geometry.curves]
        encoded['parametric_elevation_rows']=rows.tolist()
        encoded['native_height_projection']=dict(method='dense_source_height_on_retained_native_plan',source_height_knots=len(b),source_xy_unchanged=True,source_station_heights_m=np.column_stack([sb,b[:,2]]).tolist())
        fitted,_=fit_lane(encoded,1.);samples=np.unique(np.r_[arc,np.arange(0,geometry.length,.05)])
        xy_error=max(float(np.linalg.norm(geometry.xy(s)-fitted.xy(s)))for s in samples)
        z_error=float(abs(fitted.elevations(arc)-b[:,2]).max())
        derivative=elevation.derivative();extrema=elevation.derivative(2).roots(extrapolate=False)
        max_grade=float(abs(derivative(np.r_[arc,extrema[(extrema>=0)&(extrema<=geometry.length)]])).max())
        records.append(dict(source_lane_id=key,original_knots=len(original),dense_height_knots=len(b),
            native_plan_curves=len(geometry.curves),source_trace_error_m=trace_error,original_knots_trace_error_m=original_error,
            native_xy_change_m=xy_error,native_length_change_m=fitted.length-geometry.length,
            dense_height_encoding_error_m=z_error,maximum_native_grade=max_grade))
    result['physical_layer_relations']=deepcopy(projected.get('physical_layer_relations',[]))
    report=dict(method=__doc__,records=records,changed_source_lane_ids=list(selection['source_lane_ids']),
        unchanged_other_source_lanes=len(old)-len(records),source_topology_unchanged=True,
        native_plan_geometry_unchanged=all(r['native_xy_change_m']<1e-10 and abs(r['native_length_change_m'])<1e-10 for r in records),
        maximum_height_encoding_error_m=max(r['dense_height_encoding_error_m']for r in records),
        scope='Compiler representation only. Resample native surfaces and measure full physical support after the elevation rewrite.')
    return result,report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['previous-scene','scene','selection','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();scene,report=adapt(json.loads(a.previous_scene.read_text()),json.loads(a.scene.read_text()),json.loads(a.selection.read_text()))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'scene.json').write_text(json.dumps(scene,separators=(',',':'))+'\n')
    (a.output/'height_encoding.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
