"""Compile source lane boundaries into shared OpenDRIVE cross sections.

SciPy fits reference curves and profiles; scenariogeneration writes lane
widths, offsets, and markings.
"""
import math
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar
from shapely.geometry import LineString, Point
from scenariogeneration import xodr
from map_generation.road_builder import fit_lane, lane_type, stable_elevation_rows


def normal_at(geometry, station):
    i = min(len(geometry.curves)-1, max(0, int(np.searchsorted(geometry.knots, station, side='right')-1)))
    curve = geometry.curves[i]
    tangent = curve.tangent(curve.parameter(station-geometry.knots[i]))
    return np.array([-tangent[1], tangent[0]])


def intersect_boundary(line, origin, normal):
    cross = LineString([origin-50*normal, origin+50*normal])
    hit = line.intersection(cross)
    if hit.is_empty:
        raise ValueError('Cross section extends beyond supplied source boundary')
    points = [hit] if hit.geom_type == 'Point' else list(hit.geoms)
    if any(p.geom_type != 'Point' for p in points):
        raise ValueError('Source boundary folds along the road cross section')
    point = min(points, key=lambda p: p.distance(Point(origin)))
    station = line.project(point)
    position = np.asarray(line.interpolate(station).coords[0])
    return float((position[:2]-origin) @ normal), float(position[2]), station


def marking_pattern(value, side):
    # Official av2.rendering.map defines compound names relative to the
    # observed lane side. Convert to global positive-t/negative-t stripes
    # before comparing reciprocal records on the same physical boundary.
    if value in ('SOLID_DASH_YELLOW','DASH_SOLID_YELLOW'):
        solid_left=(value=='SOLID_DASH_YELLOW')==(side=='right')
        return ('SOLID','DASH','YELLOW') if solid_left else ('DASH','SOLID','YELLOW')
    return value


def source_mark(value, station=0., side='right'):
    types = {'DASHED_WHITE': xodr.RoadMarkType.broken,
             'DASHED_YELLOW': xodr.RoadMarkType.broken,
             'SOLID_WHITE': xodr.RoadMarkType.solid,
             'SOLID_YELLOW': xodr.RoadMarkType.solid,
             'DOUBLE_SOLID_YELLOW': xodr.RoadMarkType.solid_solid,
             'DOUBLE_SOLID_WHITE': xodr.RoadMarkType.solid_solid,
             'NONE': xodr.RoadMarkType.none}
    pattern=marking_pattern(value,side)
    if isinstance(pattern,tuple):
        kind=xodr.RoadMarkType.solid_broken if pattern[0]=='SOLID' else xodr.RoadMarkType.broken_solid
        mark=xodr.RoadMark(kind,width=.36,soffset=station,color=xodr.RoadMarkColor.yellow,laneChange=xodr.LaneChange.none)
        for stripe,offset in zip(pattern[:2],[.12,-.12]):
            mark.add_specific_road_line(xodr.RoadLine(width=.12,length=3.,space=0. if stripe=='SOLID'else 3.,
                toffset=offset,soffset=0.,color=xodr.RoadMarkColor.yellow))
        return mark
    if value not in types:
        raise ValueError('Road mark requires source interpretation: ' + str(value))
    return xodr.RoadMark(types[value], width=.12, soffset=station,
        color=xodr.RoadMarkColor.yellow if 'YELLOW' in value else xodr.RoadMarkColor.white,
        laneChange=xodr.LaneChange.both if value == 'DASHED_WHITE' else xodr.LaneChange.none)


def polynomial_rows(stations, values):
    spline = PchipInterpolator(stations, values)
    return [(float(s), *map(float, spline.c[:,i][::-1])) for i,s in enumerate(stations[:-1])]


def reference_station(geometry, point):
    """Locate a source boundary event on the exact common reference curve."""
    point=np.asarray(point)[:2]
    if not hasattr(geometry,'projection_samples'):
        grid=np.linspace(0,geometry.length,max(3,math.ceil(geometry.length)+1))
        geometry.projection_samples=(grid,np.asarray([geometry.xy(s)for s in grid]))
    grid,xy=geometry.projection_samples
    index=int(np.argmin(np.sum((xy-point)**2,axis=1)))
    low,high=grid[max(0,index-2)],grid[min(len(grid)-1,index+2)]
    result=minimize_scalar(lambda s:float(np.sum((geometry.xy(s)-point)**2)),bounds=(low,high),method='bounded',options={'xatol':1e-8})
    return float(min([0.,geometry.length,result.x],key=lambda s:np.sum((geometry.xy(s)-point)**2)))


def common_source_interval(lanes, limits=None, requested_interval=None):
    """Use the intersection of supplied source caps and retained native spans.

    Junction movements retain the short end tails. No source path is removed.
    Limits are cumulative centerline chord stations for each lane chain.
    """
    geometry=fit_lane(lanes[len(lanes)//2],1.)[0];starts=[];ends=[]
    for i,lane in enumerate(lanes):
        center=LineString(lane['centerline'])
        low,high=limits[i] if limits is not None else (0.,center.length)
        starts.append(reference_station(geometry,center.interpolate(low).coords[0]))
        ends.append(reference_station(geometry,center.interpolate(high).coords[0]))
        for side in ['left_boundary','right_boundary']:
            starts.append(reference_station(geometry,lane[side][0]))
            ends.append(reference_station(geometry,lane[side][-1]))
    start,end=max(starts)+.01,min(ends)-.01
    if end<=start:raise ValueError('Source caps have no common longitudinal interval')
    if requested_interval is not None:
        low, high = map(float, requested_interval)
        if low < start - 1e-6 or high > end + 1e-6 or high <= low:
            raise ValueError('Selected shared interval must stay inside source/native caps')
        return low, high
    return start,end


def marking_profile(source, side, geometry, start, end):
    intervals=source.get('marking_intervals',[])
    if not intervals:return [dict(s_m=0.,type=source[side+'_mark_type'])]
    changes=[]
    for i,record in enumerate(intervals):
        value=record[side]
        if i and value==intervals[i-1][side]:continue
        station=0. if not i else reference_station(geometry,record[side+'_start_xyz'])
        changes.append((station,value))
    active=changes[0][1]
    for station,value in changes:
        if station<=start:active=value
    return [dict(s_m=0.,type=active)]+[dict(s_m=s-start,type=value) for s,value in changes if start<s<end]


def marking_at(profile, station):
    return profile[max(0,int(np.searchsorted([r['s_m'] for r in profile],station,side='right')-1))]['type']


def compile_section(lanes, road_id=1, margin=8., spacing=.1, interval=None, center_mode='source'):
    """Lanes are ordered left to right, with source reciprocal neighbors.

    Compile the supplied common interval; road-end graph connections belong to
    the network assembly step. Actual source meshes retain their boundary Z;
    stock CARLA navigation uses the common reference elevation.
    """
    for left,right in zip(lanes, lanes[1:]):
        if left.get('right_neighbor') != right['id'] or right.get('left_neighbor') != left['id']:
            raise ValueError('Lane order lacks reciprocal source neighbor evidence')
    reference = lanes[len(lanes)//2]
    geometry,_ = fit_lane(dict(reference, width_m=reference.get('width_m', reference.get('mean_ladder_width_m'))),1.)
    start,end = interval if interval is not None else (margin,geometry.length-margin)
    if end <= start:
        raise ValueError('No shared interval remains after source endpoint margin')
    profiles=[{side:marking_profile(lane,side,geometry,start,end) for side in ['left','right']} for lane in lanes]
    for first,second in zip(profiles,profiles[1:]):
        a,b=first['right'],second['left']
        if len(a)!=len(b) or any(marking_pattern(x['type'],'right')!=marking_pattern(y['type'],'left') or abs(x['s_m']-y['s_m'])>.02 for x,y in zip(a,b)):
            raise ValueError('Shared source marking profiles disagree')
    stations = np.linspace(start,end,max(3,math.ceil((end-start)/spacing)+1))
    # Include measured boundary stations to preserve taper and cap corners.
    events=sorted({reference_station(geometry,p) for lane in lanes for side in ['left_boundary','right_boundary'] for p in lane[side]})
    stations=np.unique(np.r_[stations,[s for s in events if start<s<end],geometry.knots[(geometry.knots>start)&(geometry.knots<end)]])
    stations=stations[np.r_[True,np.diff(stations)>1e-6]]
    boundaries = [LineString(lanes[0]['left_boundary'])] + [LineString(l['right_boundary']) for l in lanes]
    centers = [LineString(l['centerline']) for l in lanes]
    shared = [LineString(l['left_boundary']) for l in lanes[1:]]
    samples=[];shared_errors=[]
    for station in stations:
        origin=geometry.xy(station);normal=normal_at(geometry,station)
        hits=[intersect_boundary(b,origin,normal) for b in boundaries]
        for i,b in enumerate(shared,1):
            other=intersect_boundary(b,origin,normal)
            shared_errors.append(abs(hits[i][0]-other[0]))
        offsets=np.array([h[0] for h in hits]);widths=-np.diff(offsets)
        if np.any(widths <= 0):
            raise ValueError('Source cross section changes lane order')
        center_hits=[intersect_boundary(c,origin,normal) for c in centers]
        # AV2's longitudinal ladder midpoint differs slightly from a normal
        # boundary midpoint on tapering roads. Preserve its center paths while
        # minimizing the largest boundary displacement in the native road.
        signs=(-1.)**np.arange(len(lanes)+1);bias=np.zeros(len(lanes)+1)
        for i,hit in enumerate(center_hits):bias[i+1]=2*hit[0]-bias[i]
        targets=signs*(offsets-bias)
        first=float((targets.min()+targets.max())/2)
        native_offsets=signs*first+bias if center_mode=='source' else offsets.copy()
        if np.any(-np.diff(native_offsets)<=0):
            raise ValueError('Center-preserving cross section has nonpositive widths')
        samples.append(dict(s_m=float(station-start),reference_xyz=geometry.xyz(station).tolist(),
            normal_xy=normal.tolist(),boundary_offsets_m=offsets.tolist(),
            boundary_z_m=[h[1] for h in hits],widths_m=widths.tolist(),
            native_boundary_offsets_m=native_offsets.tolist(),
            source_center_chord_stations_m=[h[2] for h in center_hits]))
    if max(shared_errors,default=0) > .01:
        raise ValueError('Source shared boundaries disagree by more than 1 cm')
    local=stations-start;offsets=np.array([s['native_boundary_offsets_m'] for s in samples]);widths=-np.diff(offsets,axis=1)
    plan=xodr.PlanView();geometry.add_plan(plan,start,end)
    native_lanes=xodr.Lanes();center=xodr.Lane()
    for row in profiles[0]['left']:center.add_roadmark(source_mark(row['type'],row['s_m'],'left'))
    section=xodr.LaneSection(0,center)
    for i,source in enumerate(lanes):
        profile=polynomial_rows(local,widths[:,i]);_,a,b,c,d=profile[0]
        lane=xodr.Lane(lane_type(source.get('allowed_classes',['passenger'])),a=a,b=b,c=c,d=d)
        for station,a,b,c,d in profile[1:]:lane.add_lane_width(a=a,b=b,c=c,d=d,soffset=station)
        for row in profiles[i]['right']:lane.add_roadmark(source_mark(row['type'],row['s_m']))
        section.add_right_lane(lane)
    native_lanes.add_lanesection(section)
    for row in polynomial_rows(local,offsets[:,0]):native_lanes.add_laneoffset(xodr.LaneOffset(*row))
    road=xodr.Road(road_id,plan,native_lanes)
    road.add_type(xodr.RoadType.town,speed=min(l.get('speed_mps',11.176) for l in lanes),speed_unit='m/s')
    rows,_=stable_elevation_rows(geometry.elevation_rows(start,end,0),end-start)
    for row in rows:road.add_elevation(*row)
    mapping=[dict(source_lane_id=l['id'],road_id=road_id,lane_id=-i-1,section_id=0,
        source_left_mark=l['left_mark_type'],source_right_mark=l['right_mark_type'],marking_profiles=profiles[i]) for i,l in enumerate(lanes)]
    return road,dict(source_lanes=mapping,reference_source_lane_id=reference['id'],
        reference_source_interval_m=[start,end],length_m=end-start,samples=samples,profile_sample_step_m=spacing,
        maximum_shared_boundary_disagreement_m=max(shared_errors,default=0),
        maximum_native_boundary_adjustment_m=float(np.max(abs(offsets-np.array([s['boundary_offsets_m'] for s in samples])))),
        center_mode=center_mode,
        width_source=('Normal source-boundary intersections; minimax native boundary displacement constrained by observed lane centers. Physical mesh keeps source boundaries.'
            if center_mode=='source' else 'Measured normal source-boundary intersections for both native widths and physical mesh; native centers are transverse boundary midpoints.'),
        compound_marking_scope='Official AV2 side-relative compound stripe order retained;12cm stripe width,24cm center separation and3m dash rhythm are inferred display defaults, not source measurements. Existing unsupported crossing permissions remain none.',
        native_height_scope='Common reference elevation; source transverse heights retained for physical meshes and independently checked.')


def surface_meshes(section):
    """Use the measured cross-section boundary heights for physical lane faces."""
    meshes=[]
    for i,lane in enumerate(section['source_lanes']):
        vertices=[]
        for sample in section['samples']:
            origin=np.asarray(sample['reference_xyz'][:2]);normal=np.asarray(sample['normal_xy'])
            for b in [i,i+1]:
                vertices.append([*(origin+normal*sample['boundary_offsets_m'][b]),sample['boundary_z_m'][b]])
        vertices=np.asarray(vertices);faces=[]
        for n in range(len(section['samples'])-1):
            j=2*n
            for face in [[j,j+1,j+3],[j,j+3,j+2]]:
                tri=vertices[face]
                if np.cross(tri[1]-tri[0],tri[2]-tri[0])[2]<0:face=face[::-1]
                faces.append(face)
        meshes.append(dict(id='multilane_'+str(lane['road_id'])+'_'+str(i),kind='av2_boundary_surface',
            source_lane_id=lane['source_lane_id'],xodr_road_id=lane['road_id'],xodr_lane_id=lane['lane_id'],
            vertices=vertices.tolist(),faces=faces,left_boundary=vertices[::2].tolist(),right_boundary=vertices[1::2].tolist()))
    return meshes
