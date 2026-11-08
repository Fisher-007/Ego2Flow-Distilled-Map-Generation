"""Reconnect source approaches to observed AV2 road stations.

Directed OSM/SUMO paths supply vehicle permissions; lane splits retain source
geometry and longitudinal connections.
"""
from collections import defaultdict
from copy import deepcopy
import heapq
from itertools import count

import numpy as np
from shapely.geometry import LineString, Point, box
from shapely.strtree import STRtree
from scipy.interpolate import CubicHermiteSpline

from .fusion import station_xyz, tangent, seam_geometry


def section(points, start, end):
    points=np.asarray(points)
    stations=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1))]
    inside=points[(stations>start+1e-6)&(stations<end-1e-6)]
    return np.vstack([station_xyz(points,start),inside,station_xyz(points,end)])


def missing_source_branches(scene):
    """Expose partially retained source forks for evidence review.

    Explicitly withdrawn inferred movements stay withdrawn. An access match
    accounts for the actual first branch traversed in its source path.
    """
    fused=scene['fusion_map'];lanes={l['id']:l for l in fused['lanes']}
    originals={l['id']:l for l in scene['coarse_map']['lanes']}
    covered=defaultdict(set)
    for row in fused.get('boundary_matches',[]):
        path=row['coarse_connection_path']
        if row['direction']=='entry' and row['seam_id']in lanes and len(path)>1:
            covered[row['coarse_lane_id']].add(path[-2])
    for row in fused.get('source_access_review',{}).get('connections',[]):
        if not row['incoming']:continue
        source=lanes.get(row['coarse_lane_id'],{}).get('source_lane_id')
        branch=next((k for k,_,_ in row['source_path'] if k!=source),None)
        if branch:covered[row['coarse_lane_id']].add(branch)
    retired=set();review=fused.get('inferred_movement_review',{})
    while review:
        for row in review.get('records',[]):retired.update(row['decision']['internal_source_lanes'])
        review=review.get('previous_review')
    for row in fused.get('zero_length_source_connections',{}).get('contracted',[]):
        if len(row['source_path'])>1:covered[row['from_lane']].add(row['source_path'][1])
    branch_reviews=list(fused.get('source_branch_reviews',[]))
    if fused.get('source_branch_review'):branch_reviews.append(fused['source_branch_review'])
    for branch_review in branch_reviews:
        port=branch_review['review']['port'];covered[port['coarse_lane_id']].add(port['source_branch_id'])
    gaps=[]
    for lane in lanes.values():
        if lane['source']!='osm_sumo' or abs(lane['source_interval_m'][1]-lane['source_length_m'])>.001:continue
        original=originals[lane['source_lane_id']]
        represented={lanes[k].get('source_lane_id') for k in lane['successors'] if k in lanes}|covered[lane['id']]
        for branch in original['successors']:
            common=set(lane['allowed_classes'])&set(originals[branch]['allowed_classes'])
            if branch in represented or branch in retired or 'passenger' not in common:continue
            gaps.append(dict(coarse_lane_id=lane['id'],source_branch_id=branch,incoming=True,
                retained_successors=lane['successors'],permissions=sorted(common),
                status='Source branch has no direct retained counterpart; AV2 ownership and existing matches require review'))
    return gaps


def source_direction_conflicts(scene):
    """Measure internal source stubs whose headings oppose both incident roads."""
    originals={l['id']:l for l in scene['coarse_map']['lanes']}
    predecessors=defaultdict(list)
    for lane in originals.values():
        for target in lane['successors']:predecessors[target].append(lane['id'])
    result={}
    for key,lane in originals.items():
        if not lane.get('internal') or not predecessors[key] or not lane['successors']:continue
        def direction(points,end):
            edges=np.diff(np.asarray(points)[:,:2],axis=0)
            lengths=np.linalg.norm(edges,axis=1);valid=np.flatnonzero(lengths>1e-9)
            if not len(valid):return None
            index=valid[-1 if end else 0]
            return edges[index]/lengths[index]
        start=direction(lane['centerline'],False);end=direction(lane['centerline'],True)
        incoming=[direction(originals[p]['centerline'],True)for p in predecessors[key]]
        outgoing=[direction(originals[p]['centerline'],False)for p in lane['successors']]
        if start is None or end is None or any(v is None for v in incoming+outgoing):continue
        before=[float(v@start)for v in incoming]
        after=[float(v@end)for v in outgoing]
        if max(before)<-.5 and max(after)<-.5:
            result[key]=dict(source_lane_id=key,predecessors=predecessors[key],
                successors=lane['successors'],entry_direction_cosines=before,
                exit_direction_cosines=after,
                reason='Internal geometric stub opposes both incident road directions; preserve traversal, seek observed attachment beyond this geometric fold.')
    return result


def restore_access(scene, options, reviewed_connections=None):
    fused=scene['fusion_map'];lanes=fused['lanes'];by_id={l['id']:l for l in lanes}
    originals={l['id']:l for l in scene['coarse_map']['lanes']}
    pred=defaultdict(list);source_pred=defaultdict(list);retained=defaultdict(list)
    for l in lanes:
        if l['source']=='osm_sumo':retained[l['source_lane_id']].append(l)
        for target in l['successors']:pred[target].append(l['id'])
    for l in originals.values():
        for target in l['successors']:source_pred[target].append(l['id'])
    fine=[l for l in lanes if l['source']=='av2' and
          (options.get('include_fine_junctions',False) or not l['is_intersection'])]
    shapes=[LineString(np.asarray(l['centerline'])[:,:2]) for l in fine]
    index=STRtree(shapes)
    max_distance=options.get('max_path_m',60.)
    match_distance=options.get('alignment_distance_m',1.75)
    transition_length=options.get('minimum_transition_m',8.)
    blend_length=options.get('blend_length_m',8.)
    original_lines={key:LineString(np.asarray(l['centerline'])[:,:2]) for key,l in originals.items()}
    direction_conflicts=source_direction_conflicts(scene)
    retry=options.get('retry_unresolved',False)
    results=deepcopy(reviewed_connections['connections']) if retry else []
    unresolved=[];coarse_continuations=[];geometry_rejections=[]
    retry_ports={(r['coarse_lane_id'],r['direction']=='into_fine') for r in reviewed_connections['unresolved']} if retry else None

    def search(part, incoming, branch=None):
        # incoming means an original source approach enters an observed lane.
        start=part['source_interval_m'][1 if incoming else 0]
        serial=count();queue=[(0.,next(serial),part['source_lane_id'],start,[],set(part['allowed_classes']))]
        if branch is not None:
            # Follow the missing source branch with its permissions.
            queue=[(0.,next(serial),branch,0. if incoming else original_lines[branch].length,
                    [(part['source_lane_id'],start,start)],set(part['allowed_classes']))]
        visited=set();found={}
        while queue:
            travelled,_,key,station,path,permissions=heapq.heappop(queue)
            common=permissions & set(originals[key]['allowed_classes'])
            state=(key,frozenset(common))
            if state in visited or travelled>max_distance or not common:continue
            visited.add(state)
            line=original_lines[key];length=line.length
            lo,hi=(station,length) if incoming else (0.,station)
            # Stop at the first retained road port to recover the missing interval.
            retained_ports=[]
            for existing in retained[key]:
                if existing['id']==part['id']:continue
                entry=existing['source_interval_m'][0 if incoming else 1]
                if lo-.001<=entry<=hi+.001:retained_ports.append((abs(entry-station),entry,existing['id']))
            blocking=min(retained_ports) if retained_ports else None
            if blocking:
                if incoming:hi=blocking[1]
                else:lo=blocking[1]
            samples=np.linspace(lo,hi,max(2,int((hi-lo)/1.5)+1))
            if not incoming:samples=samples[::-1]
            if key in direction_conflicts:samples=[]
            selected=None
            for s in samples:
                used=abs(s-station)
                if travelled+used>max_distance:break
                if blocking and used>=blocking[0]-1e-6:break
                # Allow longitudinal room for the lateral transition.
                if travelled+used<transition_length:continue
                xyz=station_xyz(originals[key]['centerline'],s)
                d=station_xyz(originals[key]['centerline'],min(length,s+1))-station_xyz(originals[key]['centerline'],max(0,s-1))
                d=d[:2]/max(np.linalg.norm(d[:2]),1e-9)
                ranked=[]
                for i in index.query(box(xyz[0]-match_distance,xyz[1]-match_distance,xyz[0]+match_distance,xyz[1]+match_distance)):
                    candidate=fine[i]
                    if shapes[i].length<1.:continue
                    motor_modes={'passenger','bus','delivery','truck','taxi'}
                    # Require a mixed-use segment in the source path for bicycle-to-mixed-road access.
                    if (not common & motor_modes and candidate['lane_type']!='BIKE'
                            and not set(originals[key]['allowed_classes']) & motor_modes):continue
                    modes=common & set(candidate['allowed_classes'])
                    if 'passenger' in common and 'passenger' not in modes:continue
                    if not modes:continue
                    fs=float(np.clip(shapes[i].project(Point(xyz[:2])),.5,shapes[i].length-.5))
                    target=station_xyz(candidate['centerline'],fs)
                    distance=np.linalg.norm(target[:2]-xyz[:2])
                    if distance>match_distance:continue
                    fd=station_xyz(candidate['centerline'],min(shapes[i].length,fs+1))-station_xyz(candidate['centerline'],max(0,fs-1))
                    cosine=float(d@fd[:2]/max(np.linalg.norm(fd[:2]),1e-9))
                    if cosine<.9:continue
                    maximum_curvature=options.get('maximum_motor_curvature_per_m') if 'passenger' in modes else None
                    if maximum_curvature:
                        endpoint=np.asarray(part['centerline'])[-1 if incoming else 0]
                        coarse_direction=tangent(part['centerline'],incoming)[:2]
                        fine_direction=fd[:2]/np.linalg.norm(fd[:2])
                        ends=np.array([endpoint[:2],target[:2]]) if incoming else np.array([target[:2],endpoint[:2]])
                        directions=np.array([coarse_direction,fine_direction]) if incoming else np.array([fine_direction,coarse_direction])
                        chord=np.linalg.norm(ends[1]-ends[0])
                        if chord<.001:continue
                        curve=CubicHermiteSpline([0.,1.],ends,chord*directions)
                        u=np.linspace(0,1,33);dv=curve(u,1);ddv=curve(u,2)
                        curvature=abs(dv[:,0]*ddv[:,1]-dv[:,1]*ddv[:,0])/np.maximum(np.linalg.norm(dv,axis=1),1e-8)**3
                        if curvature.max()>maximum_curvature:
                            geometry_rejections.append(dict(coarse_lane_id=part['id'],incoming=incoming,
                                fine_lane_id=candidate['id'],fine_station_m=fs,maximum_curvature_per_m=float(curvature.max())))
                            continue
                    ranked.append((distance+4*(1-cosine),candidate['id'],fs,target,modes))
                if ranked:
                    _,fid,fs,target,modes=min(ranked,key=lambda v:v[0])
                    selected=dict(fine_lane_id=fid,fine_station_m=fs,source_path=path+[(key,station,float(s))],
                                  fine_lane_is_intersection=bool(by_id[fid]['is_intersection']),
                                  permissions=sorted(modes),travel_m=travelled+used,alignment_xy_m=float(np.linalg.norm(target[:2]-xyz[:2])))
                    break
            if selected:
                fid=selected['fine_lane_id']
                if fid not in found or selected['travel_m']<found[fid]['travel_m']:found[fid]=selected
                continue
            if blocking:
                coarse_continuations.append(dict(coarse_lane_id=part['id'],incoming=incoming,
                    target=blocking[2],source_path=path+[(key,station,float(blocking[1]))],
                    travel_m=travelled+blocking[0],permissions=sorted(common)))
                continue
            for target in (originals[key]['successors'] if incoming else source_pred[key]):
                end=length if incoming else 0.
                heapq.heappush(queue,(travelled+abs(end-station),next(serial),target,
                    0. if incoming else original_lines[target].length,path+[(key,station,end)],common))
        return list(found.values())

    selected_ports=options.get('source_ports')
    for part in lanes if reviewed_connections is None or retry else []:
        if part['source']!='osm_sumo':continue
        key=part['source_lane_id'];s0,s1=part['source_interval_m'];length=part['source_length_m']
        for incoming,needed in ((True,not part['successors'] and (s1<length-.001 or bool(originals[key]['successors']))),
                                (False,not pred[part['id']] and (s0>.001 or bool(source_pred[key])))):
            requested=[r for r in selected_ports if r['coarse_lane_id']==part['id'] and r['incoming']==incoming] if selected_ports is not None else None
            if requested is not None:needed=bool(requested)
            if retry and (part['id'],incoming) not in retry_ports:continue
            if not needed:continue
            if requested is None:candidates=search(part,incoming)
            else:
                candidates=[]
                for request in requested:
                    branch=request.get('source_branch_id')
                    if branch is not None:
                        neighbors=originals[key]['successors'] if incoming else source_pred[key]
                        at_end=abs(s1-length)<.001 if incoming else abs(s0)<.001
                        if branch not in neighbors or not at_end:raise ValueError('Reviewed branch is not a source endpoint continuation')
                    candidates.extend(search(part,incoming,branch))
            if not candidates:
                unresolved.append(dict(coarse_lane_id=part['id'],direction='into_fine' if incoming else 'out_of_fine'))
            for candidate in candidates:
                results.append(dict(candidate,coarse_lane_id=part['id'],incoming=incoming))

    if options.get('propose_only'):
        return dict(connections=results,unresolved=unresolved,
                    pending_coarse_continuations=coarse_continuations,
                    geometry_rejected_candidates=geometry_rejections)

    if reviewed_connections is not None and not retry:
        results=deepcopy(reviewed_connections['connections'])
        unresolved=deepcopy(reviewed_connections['unresolved'])

    stations=defaultdict(list)
    for r in results:stations[r['fine_lane_id']].append(r['fine_station_m'])
    pieces={};split_report={}
    for key,values in stations.items():
        original=by_id[key];length=LineString(np.asarray(original['centerline'])[:,:2]).length
        cuts=[0.]
        for value in sorted(values):
            if value-cuts[-1]>.1 and length-value>.1:cuts.append(value)
        cuts.append(length);chain=[]
        next_part=1
        for n,(start,end) in enumerate(zip(cuts[:-1],cuts[1:])):
            lane=deepcopy(original)
            if n==0:lane['id']=key
            else:
                while key+f':access_part:{next_part}' in by_id:next_part+=1
                lane['id']=key+f':access_part:{next_part}';next_part+=1
            lane['centerline']=section(original['centerline'],start,end).tolist()
            lane['source_parent_lane_id']=key;lane['source_parent_interval_m']=[start,end]
            # Existing fine boundaries remain in the source layer. Derived
            # segments use their corresponding station ranges for rendering.
            for field in ('left_boundary','right_boundary'):
                bl=LineString(np.asarray(original[field])[:,:2]).length
                lane[field]=section(original[field],start/length*bl,end/length*bl).tolist()
            lane['successors']=[];chain.append(lane)
        for a,b in zip(chain[:-1],chain[1:]):a['successors']=[b['id']]
        chain[-1]['successors']=original['successors'][:]
        pieces[key]=chain;split_report[key]=[dict(id=l['id'],interval_m=l['source_parent_interval_m']) for l in chain]
    seams=[]
    for n,r in enumerate(results):
        part=by_id[r['coarse_lane_id']];key=r['fine_lane_id'];chain=pieces[key];incoming=r['incoming']
        endpoints=[(lane['source_parent_interval_m'][0 if incoming else 1],lane) for lane in chain]
        station,fine_part=min(endpoints,key=lambda pair:abs(pair[0]-r['fine_station_m']))
        target=np.asarray(fine_part['centerline'])[0 if incoming else -1]
        path=r['source_path'] if incoming else list(reversed(r['source_path']))
        segments=[section(originals[k]['centerline'],min(a,b),max(a,b)) for k,a,b in path if abs(b-a)>.001]
        source=np.vstack(segments) if segments else np.asarray(part['centerline'])[[-1 if incoming else 0]]
        # Blend the unknown lateral/height transition into the observed road,
        # while retaining the preceding original access path.
        if not incoming:source=source[::-1]
        endpoint=np.asarray(part['centerline'])[-1 if incoming else 0]
        if np.linalg.norm(source[0]-endpoint)>1e-6:source=np.vstack([endpoint,source])
        source_length=LineString(source[:,:2]).length if len(source)>1 else 0.
        target_direction=tangent(fine_part['centerline'],not incoming)
        target_direction=target_direction if incoming else -target_direction
        # Keep both endpoint tangents directed through the transition gap.
        choice=None;first=None
        spans=sorted(set([min(blend_length,source_length),*np.arange(12.,min(source_length,40.)+.01,4.),min(source_length,40.)]))
        for span in spans:
            join=max(0.,source_length-span)
            prefix=section(source,0,join) if join>.01 else source[:1]
            direction=tangent(prefix,True) if join>.01 else tangent(part['centerline'],incoming)*(1 if incoming else -1)
            chord=target[:2]-prefix[-1,:2];length=np.linalg.norm(chord)
            if length<1e-6:continue
            cosines=[float(chord@direction[:2]/length),float(chord@target_direction[:2]/length)]
            bridge=seam_geometry(prefix[-1],target,direction,target_direction)
            original_tail=section(source,join,source_length) if source_length>join+.01 else source
            reference=LineString(np.vstack([original_tail,target])[:,:2])
            deviation=float(LineString(np.asarray(bridge)[:,:2]).hausdorff_distance(reference))
            candidate=(prefix,bridge,dict(blend_length_m=float(span),endpoint_forward_cosines=cosines,
                                         forward_gap=bool(min(cosines)>.05),source_path_deviation_m=deviation))
            if first is None:first=candidate
            if min(cosines)>.05 and deviation<=max(part['width_m'],fine_part['width_m']):
                choice=candidate;break
        prefix,bridge,feedback=choice or first
        geometry=np.vstack([prefix[:-1],bridge])
        if not incoming:geometry=geometry[::-1]
        ident=options.get('seam_id_prefix','seam:source_access:')+str(n)
        if ident in by_id:raise ValueError('New access seam ID already exists: '+ident)
        seam=dict(id=ident,source='source_path_access_transition',internal=False,
            centerline=geometry.tolist(),width_m=min(part['width_m'],fine_part['width_m']),
            end_widths_m=([part['width_m'],fine_part['width_m']] if incoming else [fine_part['width_m'],part['width_m']]),
            speed_mps=min(part['speed_mps'],fine_part['speed_mps']),allowed_classes=r['permissions'],
            successors=[fine_part['id']] if incoming else [part['id']])
        if incoming:part['successors'].append(ident)
        else:fine_part['successors'].append(ident)
        r.update(seam_id=ident,attached_fine_part=fine_part['id'],attached_station_m=station,geometry_feedback=feedback)
        seams.append(seam)
    fused['lanes']=[part for lane in lanes for part in pieces.get(lane['id'],[lane])]+seams
    fused['source_access_review']=dict(connections=results,unresolved=unresolved,fine_station_splits=split_report,
        pending_coarse_continuations=coarse_continuations,
        geometry_rejected_candidates=geometry_rejections,
        method='Follow the original directed coarse path and its intersected permissions to a direction-aligned observed fine road station; preserve longitudinal fine connectivity.',
        status='Constructed candidates; geometry, access rules and native driving remain subject to review.')
    return scene
