"""Estimate bridge-deck profiles from source tags and approach elevations."""
from collections import defaultdict
import numpy as np


def profiles(scene, approach_m=18.):
    lanes=scene['fusion_map']['lanes'];groups=defaultdict(list)
    for lane in lanes:
        if lane['source']=='osm_sumo' and lane.get('source_tags',{}).get('bridge') not in (None,'no','false','0'):
            groups[lane['edge_id'].lstrip('-').split('#')[0]].append(lane)
    result=[]
    for key,members in groups.items():
        points=np.concatenate([l['centerline'] for l in members]);center=points[:,:2].mean(0)
        # Width can exceed the length of a short bridge: use road direction.
        p=np.array(members[0]['centerline'],dtype=float);axis=(p[-1,:2]-p[0,:2]);axis/=np.linalg.norm(axis)
        normal=np.array([-axis[1],axis[0]]);st=(points[:,:2]-center)@axis
        lo,hi=float(st.min()),float(st.max());half_width=float(abs((points[:,:2]-center)@normal).max()+max(l['width_m'] for l in members)/2)
        other=np.concatenate([l['centerline'] for l in lanes if l not in members and l['source']=='osm_sumo'])
        s=(other[:,:2]-center)@axis;n=(other[:,:2]-center)@normal
        # Sample both banks outside the tagged bridge span.
        banks=[];bank_sources=[];ground=np.array(scene['terrain']['vertices'])
        for lower,upper in [(lo-approach_m,lo-3.),(hi+3.,hi+approach_m)]:
            selected=(s>=lower)&(s<=upper)&(abs(n)<half_width+2.)
            if selected.sum()>=3:
                banks.append([float(np.median(s[selected])),float(np.median(other[selected,2]))]);bank_sources.append('Existing approach road samples')
            else:
                gs=(ground[:,:2]-center)@axis;gn=(ground[:,:2]-center)@normal
                selected=(gs>=lower)&(gs<=upper)&(abs(gn)<half_width)
                if selected.sum()<3:raise ValueError('Bridge needs terrain coverage on both banks: '+key)
                banks.append([float(np.median(gs[selected])),float(np.median(ground[selected,2]))]);bank_sources.append('Bare-earth bank elevation in continuation corridor; no native road beyond current crop')
        grade=(banks[1][1]-banks[0][1])/(banks[1][0]-banks[0][0]);intercept=banks[0][1]-grade*banks[0][0]
        if abs(grade)>.12:raise ValueError('Approach-supported bridge grade needs source review: '+key)
        result.append({'osm_way':key,'lane_ids':[l['id'] for l in members],
            'center_xy_m':center.tolist(),'axis_xy':axis.tolist(),'station_range_m':[lo,hi],
            'half_width_m':half_width,'approach_blend_m':approach_m,'bank_anchors_station_z_m':banks,
            'bank_sources':bank_sources,
            'grade':grade,'intercept_z_m':intercept,
            'source_tags':members[0]['source_tags'],
            'scope':'OSM bridge extent and connected approach elevations; inferred shared deck profile, not surveyed deck height or clearance.'})
    return result


def weights(xy, profile):
    offset=np.asarray(xy)-profile['center_xy_m'];axis=np.array(profile['axis_xy'])
    station=offset@axis;lateral=abs(offset@np.array([-axis[1],axis[0]]));lo,hi=profile['station_range_m']
    distance=np.maximum(np.maximum(lo-station,station-hi),0)
    u=np.clip(distance/profile['approach_blend_m'],0,1)
    w=(1-3*u*u+2*u*u*u)*np.clip((profile['half_width_m']+2-lateral)/2,0,1)
    for anchor in profile.get('fixed_connection_heights',[]):
        distance=np.linalg.norm(np.asarray(xy)-anchor['xy_m'],axis=1)
        t=np.clip((distance-anchor['preserve_radius_m'])/anchor['blend_m'],0,1)
        w*=t*t*(3-2*t)
    return w,station


def adjust(vertices, plans):
    v=np.array(vertices,dtype=float)
    for plan in plans:
        w,s=weights(v[:,:2],plan)
        if 'station_z_samples' in plan:
            samples=np.asarray(plan['station_z_samples']);target=np.interp(s,samples[:,0],samples[:,1])
        else:target=plan['intercept_z_m']+plan['grade']*s
        v[:,2]+=w*(target-v[:,2])
    return v


def shared_road_profiles(scene, way_ids, approach_m=3.):
    """Average source-center heights along a selected straight, same-level roadway."""
    result=[]
    for way in way_ids:
        members=[l for l in scene['fusion_map']['lanes'] if l['source']=='osm_sumo'
                 and l.get('edge_id','').lstrip('-').split('#')[0]==str(way)]
        if len(members)<2:raise ValueError('Shared roadway needs adjacent source lanes: '+str(way))
        layers={str(l.get('source_tags',{}).get('layer','0')) for l in members}
        if len(layers)!=1 or any(l.get('source_tags',{}).get('bridge') not in (None,'no','false','0') for l in members):
            raise ValueError('Review separate-level or bridge surfaces independently')
        points=np.concatenate([l['centerline'] for l in members]);center=points[:,:2].mean(0)
        line=np.asarray(members[0]['centerline']);axis=line[-1,:2]-line[0,:2];axis/=np.linalg.norm(axis)
        normal=np.array([-axis[1],axis[0]]);s=(points[:,:2]-center)@axis;n=(points[:,:2]-center)@normal
        for lane in members:
            path=np.asarray(lane['centerline']);lateral=(path[:,:2]-center)@normal
            if np.ptp(lateral)>1.:raise ValueError('Curved roadway requires an arc-length profile, not this straight-section repair')
        lo,hi=float(s.min()),float(s.max());stations=np.linspace(lo,hi,max(2,int(np.ceil(hi-lo)) + 1))
        heights=[]
        for lane in members:
            path=np.asarray(lane['centerline']);st=(path[:,:2]-center)@axis;order=np.argsort(st)
            heights.append(np.interp(stations,st[order],path[order,2]))
        result.append({'osm_way':str(way),'lane_ids':[l['id'] for l in members],
            'center_xy_m':center.tolist(),'axis_xy':axis.tolist(),'station_range_m':[lo,hi],
            'half_width_m':float(abs(n).max()+max(l['width_m'] for l in members)/2),
            'approach_blend_m':approach_m,'station_z_samples':np.c_[stations,np.mean(heights,axis=0)].tolist(),
            'scope':'Reviewed same-level straight OSM roadway: shared inferred cross-section height across opposite lanes, with a short approach blend. Source XY/permissions retained.'})
    return result


def connected_bridge_profile(scene, bridge, approach_m=65.):
    """Fit a shared cross-section profile along a selected bridge corridor."""
    from scipy.interpolate import PchipInterpolator
    from scipy.ndimage import gaussian_filter1d
    axis = np.asarray(bridge['axis_xy']); center = np.asarray(bridge['center_xy_m'])
    normal = np.array([-axis[1], axis[0]])
    lo, hi = bridge['station_range_m']; begin = lo - approach_m
    stations = np.arange(begin, hi + .5, .5)
    samples = []
    for lane in scene['fusion_map']['lanes']:
        if lane['source'] != 'osm_sumo' or lane.get('source_tags', {}).get('name') != bridge['source_tags'].get('name'):
            continue
        points = np.asarray(lane['centerline']); direction = points[-1, :2]-points[0, :2]
        if np.linalg.norm(direction) < .1 or abs(direction @ axis)/np.linalg.norm(direction) < .97:
            continue
        s = (points[:, :2]-center) @ axis
        if np.max(abs((points[:, :2]-center) @ normal)) > bridge['half_width_m']+3:
            continue
        order = np.argsort(s); z = np.interp(stations, s[order], points[order, 2])
        z[(stations < s.min()) | (stations > s.max())] = np.nan
        samples.append(z)
    values = np.asarray(samples); count = np.isfinite(values).sum(axis=0)
    known = count > 0
    z = np.nansum(values[:, known], axis=0)/count[known]
    target = PchipInterpolator(stations[known], z, extrapolate=True)(stations)
    target = gaussian_filter1d(target, 2, mode='nearest')
    # Match the deck and its derivative without a second per-lane DEM blend.
    deck = bridge['intercept_z_m'] + bridge['grade']*stations
    mix = np.clip((stations-(lo-8))/8, 0, 1); mix = mix*mix*(3-2*mix)
    target = target*(1-mix)+deck*mix
    lanes={lane['id']:lane for lane in scene['fusion_map']['lanes']};anchors=[]
    for key,lane in lanes.items():
        for next_key in lane['successors']:
            if next_key not in lanes:continue
            following=lanes[next_key]
            if (lane['source']=='osm_sumo')==(following['source']=='osm_sumo'):continue
            coarse=lane if lane['source']=='osm_sumo' else following
            point=np.asarray(lane['centerline'][-1]);next_point=np.asarray(following['centerline'][0])
            if np.linalg.norm(point[:2]-next_point[:2])>.1:continue
            anchors.append({'from':key,'to':next_key,'xy_m':point[:2].tolist(),
                'source_z_m':float(point[2]),'preserve_radius_m':max(4.,coarse['width_m']),
                'blend_m':8.,'scope':'Retain the existing coarse/fine connection height and its lane-width neighborhood while blending the inferred coarse profile.'})
    return dict(bridge, station_range_m=[float(begin), float(hi)],
                half_width_m=bridge['half_width_m']+7, approach_blend_m=6.,
                fixed_connection_heights=anchors,
                station_z_samples=np.c_[stations, target].tolist(),
                scope='Shared inferred same-level corridor profile from retained named-road centers; reviewed bridge deck held. Not a surveyed crossfall or permission change.')
