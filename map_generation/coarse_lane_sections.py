"""Prepare shared cross sections from SUMO lane order, widths, and connections."""
from copy import deepcopy
import math
import numpy as np
from shapely.geometry import LineString
from .road_builder import fit_lane
from .lane_sections import reference_station,normal_at,intersect_boundary


def shared_permissions(lanes):
    """Read this edge direction's OSM change tags in left-to-right order."""
    tags=lanes[0].get('source_tags',{})
    direction='backward' if lanes[0]['edge_id'].startswith('-') else 'forward'
    key=next((k for k in ['change:lanes:'+direction,'change:'+direction,'change:lanes','change']if k in tags),None)
    values=tags[key].split('|') if key and ':lanes' in key else [tags[key]if key else 'yes']*len(lanes)
    if len(values)!=len(lanes):
        raise ValueError('OSM change tag lane count differs from retained passenger lanes: '+str((key,values)))
    options={'yes':(True,True),'no':(False,False),'not_left':(False,True),'not_right':(True,False),
        'only_left':(True,False),'only_right':(False,True)}
    if any(v not in options for v in values):raise ValueError('Unresolved OSM lane change value: '+str(values))
    pairs=[(options[a][1],options[b][0])for a,b in zip(values,values[1:])]
    if any(a!=b for a,b in pairs):
        raise ValueError('One-way divider crossing requires agreement between native waypoint and planner permissions')
    return [a for a,b in pairs],dict(key=key,values=values,
        source='Explicit OSM change tag'if key else 'Default crossing between consecutive same-direction passenger lanes; no explicit restriction')


def prepare_group(lanes):
    """Accept consecutive retained lanes ordered left to right on one SUMO edge."""
    edge=lanes[0]['edge_id']
    if any(l['source']!='osm_sumo' or l['edge_id']!=edge or l.get('internal') for l in lanes):
        raise ValueError('Coarse section must belong to one external SUMO edge')
    indices=[int(l['source_lane_id'].rsplit('_',1)[1]) for l in lanes]
    if not all(a-b==1 for a,b in zip(indices,indices[1:])):
        raise ValueError('Source lane order contains a gap or repeated clipped pieces')
    permissions,permission_source=shared_permissions(lanes)
    reference=fit_lane(lanes[len(lanes)//2],1.)[0]
    lines=[LineString(l['centerline']) for l in lanes]
    start=max(reference_station(reference,l['centerline'][0])for l in lanes)+.002
    end=min(reference_station(reference,l['centerline'][-1])for l in lanes)-.002
    if end<=start:raise ValueError('Retained coarse lanes have no common cross section')
    stations=np.unique(np.r_[np.linspace(start,end,max(3,math.ceil(end-start)+1)),reference.knots[(reference.knots>start)&(reference.knots<end)]])
    boundaries=[[]for _ in range(len(lanes)+1)];widths=np.array([l['width_m']for l in lanes]);deviations=[]
    for station in stations:
        origin=reference.xy(station);normal=normal_at(reference,station)
        hits=[intersect_boundary(line,origin,normal)for line in lines]
        offsets=np.array([h[0]for h in hits]);heights=np.array([h[1]for h in hits])
        if np.any(np.diff(offsets)>=0):raise ValueError('Retained SUMO lane centers cross or reverse order')
        shared=(offsets[:-1]*widths[1:]+offsets[1:]*widths[:-1])/(widths[:-1]+widths[1:])
        levels=(heights[:-1]*widths[1:]+heights[1:]*widths[:-1])/(widths[:-1]+widths[1:])
        edges=np.r_[offsets[0]+widths[0]/2,shared,offsets[-1]-widths[-1]/2]
        zs=np.r_[heights[0],levels,heights[-1]]
        deviations.extend(abs(-np.diff(edges)-widths).tolist())
        for points,offset,z in zip(boundaries,edges,zs):points.append([*(origin+normal*offset),z])
    result=deepcopy(lanes)
    for i,lane in enumerate(result):
        lane.update(left_boundary=boundaries[i],right_boundary=boundaries[i+1],
            left_neighbor=result[i-1]['id']if i else None,
            right_neighbor=result[i+1]['id']if i+1<len(result)else None,
            left_mark_type=('DASHED_WHITE'if permissions[i-1]else 'SOLID_WHITE')if i else 'NONE',
            right_mark_type=('DASHED_WHITE'if permissions[i]else 'SOLID_WHITE')if i+1<len(result)else 'NONE',
            coarse_boundary_evidence=dict(edge_id=edge,lane_indices=indices,
                source='OSM via SUMO lane inference',
                boundary_method='Shared normal intersections of source centers; divider partitions center separation by source nominal widths.',
                marking_source='Divider appearance inferred from crossing permission; precise paint is unobserved.',
                change_permissions=permission_source,
                max_nominal_width_difference_m=max(deviations)))
    return result,dict(edge_id=edge,source_lane_ids=[l['id']for l in lanes],
        length_m=end-start,max_nominal_width_difference_m=max(deviations),source_centers_preserved=True)
