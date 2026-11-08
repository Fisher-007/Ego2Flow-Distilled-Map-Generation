"""Join consecutive observed lane pieces without changing their source paths."""
from copy import deepcopy
import numpy as np
from shapely.geometry import LineString
from map_generation.road_builder import fit_lane,bridge_geometry


def concatenate_lanes(keys, lanes):
    records=[lanes[key] for key in keys]
    for a,b in zip(records,records[1:]):
        if a['successors']!=[b['id']]:raise ValueError('Lane chain crosses a source branch')
        if a.get('is_intersection') or b.get('is_intersection'):raise ValueError('Lane chain crosses a source intersection')
        if set(a['allowed_classes'])!=set(b['allowed_classes']):raise ValueError('Lane chain changes vehicle permissions')
    result=deepcopy(records[0]);points=[];coefficients=[];parts=[];marks=[];cursor=0.;previous=None
    boundaries={side:[] for side in ['left_boundary','right_boundary']};last_boundaries={}
    for record in records:
        geometry=fit_lane(record,1.)[0];xyz=np.asarray(record['centerline']);line=LineString(xyz)
        if previous is not None:
            gap=float(np.linalg.norm(xyz[0]-np.asarray(points[-1])))
            if gap>.02:raise ValueError('Source lane chain has an unresolved endpoint gap')
            bridge=bridge_geometry(previous,geometry)
            if bridge is not None:coefficients.extend([c.coefficients.tolist() for c in bridge.curves])
            cursor+=float(np.linalg.norm(xyz[0,:2]-np.asarray(points[-1])[:2]))
        begin=cursor;cursor+=line.length
        parts.append(dict(source_lane_id=record['id'],chord_start_m=begin,chord_end_m=cursor))
        marks.append(dict(chord_start_m=begin,chord_end_m=cursor,left=record['left_mark_type'],right=record['right_mark_type'],
            left_start_xyz=record['left_boundary'][0],right_start_xyz=record['right_boundary'][0]))
        points.extend(xyz.tolist() if not points or not np.array_equal(xyz[0],points[-1]) else xyz[1:].tolist())
        coefficients.extend([c.coefficients.tolist() for c in geometry.curves]);previous=geometry
        for side in boundaries:
            b=np.asarray(record[side])
            if side in last_boundaries and np.array_equal(b,last_boundaries[side]):continue
            if boundaries[side] and np.linalg.norm(b[0]-boundaries[side][-1])>.03:
                raise ValueError('Consecutive source boundary caps require review: '+record['id']+' '+side)
            boundaries[side].extend(b[1:].tolist() if boundaries[side] and np.array_equal(b[0],boundaries[side][-1]) else b.tolist())
            last_boundaries[side]=b
    result.update(centerline=points,parametric_xy_segments=coefficients,**boundaries,
        source_chain=parts,marking_intervals=marks,speed_mps=min(r['speed_mps'] for r in records),
        successors=records[-1]['successors'])
    result.pop('boundary_shape',None)
    return result
