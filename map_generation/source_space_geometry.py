"""Realize an Agent-selected shared passage from its retained source axis."""
from copy import deepcopy
import numpy as np
import shapely


def declared_shared_lane(tags):
    """Read an explicit single physical lane shared by both directions."""
    return (tags.get('lanes') == '1' and tags.get('lanes:both_ways') == '1'
            and tags.get('oneway', 'no') not in ('yes', '1', '-1'))


def displace_xy(points, center_xy, displacement_xy, radius_m):
    """The same compact spatial field for navigation and dependent surfaces."""
    points=np.asarray(points,float).copy()
    t=np.clip(np.linalg.norm(points[:,:2]-np.asarray(center_xy),axis=1)/radius_m,0,1)
    weight=(1-t)**4*(1+4*t)
    points[:,:2]+=weight[:,None]*np.asarray(displacement_xy)
    return points,weight


def deform_inferred_neighborhood(lanes, center_xy, displacement_xy, radius_m):
    """Apply one smooth spatial displacement to connected inferred geometry.

    Equal shared endpoints receive the same displacement. Source identities,
    permissions and observed AV2 geometry stay fixed; the caller reviews the
    source space and checks vehicle clearance before compiling the candidate.
    """
    output=[];changed={}
    center=np.asarray(center_xy,float);delta=np.asarray(displacement_xy,float)
    for original in lanes:
        if original.get('source') not in ('osm_sumo','source_path_access_transition','boundary_interpolation'):
            output.append(original);continue
        points,weight=displace_xy(original['centerline'],center,delta,radius_m)
        if not np.any(weight>0):output.append(original);continue
        lane=deepcopy(original);lane['centerline']=points.tolist()
        lane['source_space_revision']=dict(role='reviewed_local_clearance',center_xy_m=center.tolist(),
            displacement_xy_m=delta.tolist(),radius_m=radius_m,source_successors_preserved=True)
        for key in ('boundary_shape','parametric_xy_segments','left_boundary','right_boundary'):
            lane.pop(key,None)
        changed[lane['id']]=float(np.linalg.norm(delta)*weight.max());output.append(lane)
    return output,dict(changed_lanes=changed,center_xy_m=center.tolist(),displacement_xy_m=delta.tolist(),
        radius_m=radius_m,observed_av2_preserved=True,native_validation='pending')


def share_inferred_passage(lanes, source_axis, source_way_id, width_m):
    """Fit both inferred directions to one passage and carry their end shifts.

    The caller selects this representation after reviewing source space.
    Original source tags, observed lanes and external neighboring roads stay
    fixed. Connected inferred movements absorb the changed entry/exit poses.
    """
    before={l['id']:l for l in lanes};after=dict(before)
    axis=shapely.LineString(np.asarray(source_axis)[:,:2]);changed={}
    selected=[k for k,l in before.items() if l.get('source')=='osm_sumo'
              and not l.get('internal')
              and l.get('edge_id','').lstrip('-').split('#')[0]==str(source_way_id)]
    if not selected:raise ValueError('Reviewed passage has no retained inferred lanes')
    for key in selected:
        lane=deepcopy(before[key]);points=np.asarray(lane['centerline'])
        xy=shapely.get_coordinates(shapely.line_interpolate_point(axis,
            shapely.line_locate_point(axis,shapely.points(points[:,:2]))))
        delta=xy-points[:,:2];points[:,:2]=xy
        lane.update(centerline=points.tolist(),width_m=width_m,end_widths_m=[width_m,width_m],
            source_space_revision=dict(source_way_id=str(source_way_id),
                role='bidirectional_shared_passage',width_m=width_m,
                source_axis_retained=True,
                opposing_traffic_requirement='Yield before entering an occupied shared strip.',
                opposing_traffic_validation='pending'))
        for field in ['boundary_shape','parametric_xy_segments','left_boundary','right_boundary']:
            lane.pop(field,None)
        after[key]=lane;changed[key]=dict(start_delta_xy_m=delta[0].tolist(),end_delta_xy_m=delta[-1].tolist())
    # Source internal movements have identified external entry/exit parents.
    # A common offset at each endpoint preserves their source adjacency.
    predecessors={k:[] for k in before}
    for key,lane in before.items():
        for target in lane['successors']:
            if target in predecessors:predecessors[target].append(key)
    withdrawn_turns=[]
    for key,original in before.items():
        connector=(original.get('source')=='osm_sumo' and original.get('internal')) or original.get('source') in ('source_path_access_transition','boundary_interpolation')
        if not connector:continue
        parents=[p for p in predecessors[key] if p in selected]
        children=[p for p in original['successors'] if p in selected]
        if not parents and not children:continue
        if parents and children:
            # Opposing inferred directions share one narrow endpoint. The
            # importer-created forward U-turn has no car-sized turning area.
            a=np.asarray(before[parents[0]]['centerline']);b=np.asarray(before[children[0]]['centerline'])
            u=a[-1,:2]-a[-2,:2];v=b[1,:2]-b[0,:2]
            if u@v/(np.linalg.norm(u)*np.linalg.norm(v))<-.9:
                lane=deepcopy(original)
                lane['allowed_classes']=sorted(set(original['allowed_classes']) & {'pedestrian','bicycle'})
                lane['source_space_revision']=dict(source_way_id=str(source_way_id),role='unsupported_inferred_forward_turn',reason='Both car directions share one narrow dead-end strip; the default forward turnaround cannot fit the intended vehicle.')
                after[key]=lane;changed[key]=dict(start_delta_xy_m=[0.,0.],end_delta_xy_m=[0.,0.])
                withdrawn_turns.append(key);continue
        start=np.asarray(changed[parents[0]]['end_delta_xy_m']) if parents else np.zeros(2)
        end=np.asarray(changed[children[0]]['start_delta_xy_m']) if children else np.zeros(2)
        points=np.asarray(original['centerline']);station=np.r_[0,np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1).cumsum()]
        t=station/station[-1];blend=t*t*(3-2*t)
        points[:,:2]+=(1-blend[:,None])*start+blend[:,None]*end
        lane=deepcopy(original);lane.update(centerline=points.tolist(),
            end_widths_m=[width_m if parents else original['width_m'],width_m if children else original['width_m']],
            source_space_revision=dict(source_way_id=str(source_way_id),role='shared_passage_approach'))
        for field in ['boundary_shape','parametric_xy_segments','left_boundary','right_boundary']:
            lane.pop(field,None)
        after[key]=lane;changed[key]=dict(start_delta_xy_m=start.tolist(),end_delta_xy_m=end.tolist())
    return [after[l['id']] for l in lanes],dict(source_way_id=str(source_way_id),
        shared_lane_ids=selected,changed_lanes=changed,width_m=width_m,
        source_successors_preserved=True,ordinary_forward_turns_withdrawn=withdrawn_turns,
        observed_av2_preserved=True)
