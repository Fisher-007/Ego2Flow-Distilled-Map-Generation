"""Pack parallel source approaches into native multi-lane roads.

Inputs contain left-to-right lane groups sharing junction endpoints or
open map boundaries. Movements use their retained source intervals.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scenariogeneration import xodr
from map_generation.lane_sections import compile_section,normal_at,common_source_interval,marking_at,intersect_boundary
from map_generation.lane_chains import concatenate_lanes
from map_generation.road_builder import fit_lane,chord_station_to_arc,arc_station_to_chord,make_road,bridge_geometry


def pack(scene, original, groups, output, margin=None, center_mode='boundaries', spacing=.1, section_cache=None, section_intervals=None):
    if margin is not None and section_intervals:
        raise ValueError('Select source intervals or a diagnostic margin, not both')
    selected_intervals={tuple(map(tuple,row['source_lane_ids'])):row['reference_interval_m']
                        for row in (section_intervals or [])}
    lanes={l['id']:l for l in scene['fusion_map']['lanes']}
    predecessors={k:[] for k in lanes}
    for key,lane in lanes.items():
        for target in lane['successors']:
            if target in predecessors:predecessors[target].append(key)
    mapping=json.loads((original/'road_lane_ids.json').read_text())
    tree=ET.parse(original/'roads.xodr');root=tree.getroot()
    from scripts.scene.transfer_native_signals import capture_signals, restore_signals
    original_signals=capture_signals(root,mapping)
    elements={int(r.get('id')):r for r in root.findall('road')}
    from scripts.scene.native_group_endpoints import normalize_endpoints
    ends,direct_edges=normalize_endpoints(root,mapping,groups)
    old_metadata=deepcopy(mapping['roads']);old_ids=dict(mapping['key_to_road_id'])
    geometries={};native_ids={};group_spans={};sections=[];removed=set();replacements={}
    start_tails={};end_tails={}
    def geometry(key):
        if key not in geometries:geometries[key]=fit_lane(lanes[key],1.)[0]
        return geometries[key]
    for supplied in groups:
        chains=[[k] if isinstance(k,str) else k for k in supplied]
        keys=[k for chain in chains for k in chain]
        print(dict(stage='pack_source_group',group_index=len(sections),source_lane_ids=keys),flush=True)
        for chain in chains:
            for a,b in zip(chain,chain[1:]):
                if predecessors[b]!=[a]:raise ValueError('Lane chain crosses a source merge: '+b)
        for left,right in zip(chains,chains[1:]):
            lp={lanes[k].get('source_parent_lane_id',k) for k in left}
            rp={lanes[k].get('source_parent_lane_id',k) for k in right}
            if not (any(lanes[k].get('right_neighbor') in rp for k in left)
                    and any(lanes[k].get('left_neighbor') in lp for k in right)):
                raise ValueError('Source chains lack reciprocal lateral evidence')
        sources=[concatenate_lanes(chain,lanes) for chain in chains]
        for i,source in enumerate(sources):
            source['left_neighbor']=sources[i-1]['id'] if i else None
            source['right_neighbor']=sources[i+1]['id'] if i+1<len(sources) else None
        old_roads=[old_ids[k] for k in keys]
        if len(set(old_roads))!=len(keys):raise ValueError('Already packed source group')
        for endpoint in ['start','end']:
            boundary_keys=[chain[0 if endpoint=='start' else -1] for chain in chains]
            values={ends.get((k,endpoint)) for k in boundary_keys}
            if len(values)!=1:raise ValueError('Source group requires aligned corridor endpoints: '+str(keys))
            if values=={None}:
                link='predecessor' if endpoint=='start' else 'successor'
                if any(elements[old_ids[k]].find('link/'+link) is not None for k in boundary_keys):
                    raise ValueError('Pack the continuing source corridor before an unmatched direct road link: '+str(keys))
        limits=[]
        for source,chain in zip(sources,chains):
            start=[s for s in mapping['lane_to_road_intervals'][chain[0]] if s['junction_id'] is None][0]['source_s_start_m']
            end=[s for s in mapping['lane_to_road_intervals'][chain[-1]] if s['junction_id'] is None][0]['source_s_end_m']
            limits.append((arc_station_to_chord(lanes[chain[0]],geometry(chain[0]),start),
                source['source_chain'][-1]['chord_start_m']+arc_station_to_chord(lanes[chain[-1]],geometry(chain[-1]),end)))
        rid=min(old_roads)
        requested_interval=selected_intervals.get(tuple(map(tuple,chains)))
        interval=common_source_interval(sources,limits,requested_interval=requested_interval) if margin is None else None
        cached=Path(section_cache)/str(rid) if section_cache else None
        if cached and cached.with_suffix('.json').exists():
            section=json.loads(cached.with_suffix('.json').read_text())
            if section['source_lane_records']!=sources or section['native_limits_m']!=[list(v)for v in limits] or section['center_mode']!=center_mode:
                raise ValueError('Cached section belongs to different source boundaries or native limits')
            expected_interval=list(interval) if interval is not None else [margin,fit_lane(sources[len(sources)//2],1.)[0].length-margin]
            if section['reference_source_interval_m'] != expected_interval:
                raise ValueError('Cached section belongs to a different selected interval')
            road_element=ET.parse(cached.with_suffix('.xml')).getroot()
        else:
            road,section=compile_section(sources,road_id=rid,margin=margin,interval=interval,center_mode=center_mode,spacing=spacing)
            road_element=road.get_element()
        if section['maximum_native_boundary_adjustment_m']>.05:raise ValueError('Native cross section moves a source boundary by over 5 cm')
        for endpoint,link in [('start','predecessor'),('end','successor')]:
            jid=ends.get((chains[0][0 if endpoint=='start' else -1],endpoint))
            if jid is not None:
                links=road_element.find('link')
                if links is None:links=ET.Element('link');road_element.insert(0,links)
                ET.SubElement(links,link,elementType='junction',elementId=str(jid))
        for i,source in enumerate(sources):
            chord=np.array([s['source_center_chord_stations_m'][i] for s in section['samples']])
            stations=np.array([s['s_m'] for s in section['samples']])
            prefix=[];suffix=[]
            for part in source['source_chain']:
                key=part['source_lane_id'];begin,finish=part['chord_start_m'],part['chord_end_m']
                g=geometry(key)
                prefix_end=chord_station_to_arc(lanes[key],g,min(max(chord[0]-begin,0.),finish-begin))
                suffix_start=chord_station_to_arc(lanes[key],g,min(max(chord[-1]-begin,0.),finish-begin))
                if prefix_end>1e-8:prefix.append([key,0.,prefix_end])
                if g.length-suffix_start>1e-8:suffix.append([key,suffix_start,g.length])
                native_ids[key]=(rid,-i-1);mapping['key_to_road_id'][key]=rid
                start=max(chord[0],begin);end=min(chord[-1],finish)
                if end<=start:continue
                sample=np.unique(np.r_[start,chord[(chord>start)&(chord<end)],end])
                local=sample-part['chord_start_m'];arc=[chord_station_to_arc(lanes[key],geometry(key),s) for s in local]
                native_stations=np.interp(sample,chord,stations).tolist()
                spans=[r for r in mapping['lane_to_road_intervals'][key] if r['junction_id'] is None]
                if len(spans)!=1 or arc[0]<spans[0]['source_s_start_m']-1e-6 or arc[-1]>spans[0]['source_s_end_m']+1e-6:
                    raise ValueError('Shared interval extends beyond the original external source interval: '+key)
                group_spans[key]=dict(road_id=rid,lane_id=-i-1,section_id=0,junction_id=None,
                    road_s_start_m=native_stations[0],road_s_end_m=native_stations[-1],source_s_start_m=arc[0],source_s_end_m=arc[-1],
                    road_stations_m=native_stations,source_stations_m=arc)
            def with_source_bridges(parts):
                result=[]
                for part in parts:
                    if result:
                        a,b=result[-1][0],part[0];bridge=bridge_geometry(geometry(a),geometry(b))
                        if bridge is not None:
                            key=f'bridge:{a}->{b}';geometries[key]=bridge;result.append([key,0.,bridge.length])
                    result.append(part)
                return result
            start_tails[chains[i][0]]=with_source_bridges(prefix)
            end_tails[chains[i][-1]]=with_source_bridges(suffix)
        section['source_lane_records']=sources
        replacements[rid]=road_element;removed.update(old_roads)
        for old in old_roads:mapping['roads'].pop(str(old));mapping['road_lane_ids'].pop(str(old),None)
        mapping['roads'][str(rid)]=dict(kind='lane_group',source_lane_ids=keys,native_lanes=section['source_lanes'],
            junction_id=None,allowed_classes=sorted(set(c for l in sources for c in l['allowed_classes'])),
            speed_mps=min(l['speed_mps'] for l in sources))
        sections.append(section)
    rebuilt=[]
    for old_rid,meta in old_metadata.items():
        if meta['kind']!='movement':continue
        a,b=meta['from_lane_id'],meta['to_lane_id']
        if a not in native_ids and b not in native_ids:continue
        rid=int(old_rid);pieces=deepcopy(meta['source_intervals'])
        if a in end_tails:pieces=end_tails[a]+pieces[1:]
        if b in start_tails:pieces=pieces[:-1]+start_tails[b]
        for source,target in zip(meta['source_lane_path'],meta['source_lane_path'][1:]):
            first,second=geometry(source),geometry(target)
            key=f'bridge:{source}->{target}'
            if any(p[0]==key for p in pieces):geometries[key]=bridge_geometry(first,second)
        if any(end<=start for _,start,end in pieces):raise ValueError('Shared cut reverses a retained movement interval')
        road=make_road(rid,pieces,geometries,meta['allowed_classes'],meta['speed_mps'],meta['junction_id'])
        # Preserve external endpoints when source labels refer to internal movement intervals.
        def retained_endpoint(link):
            element=elements[rid]
            road_link=element.find('link/'+link)
            lane_link=element.find('lanes/laneSection/right/lane/link/'+link)
            return int(road_link.get('elementId')),int(lane_link.get('id'))
        before=native_ids[a] if a in native_ids else retained_endpoint('predecessor')
        after=native_ids[b] if b in native_ids else retained_endpoint('successor')
        road.add_predecessor(xodr.ElementType.road,before[0],xodr.ContactPoint.end)
        road.add_successor(xodr.ElementType.road,after[0],xodr.ContactPoint.start)
        road.lanes.lanesections[0].rightlanes[0].add_link('predecessor',before[1]).add_link('successor',after[1])
        replacements[rid]=road.get_element();mapping['roads'][old_rid]['source_intervals']=pieces
        path=[]
        for key,_,_ in pieces:
            if key in lanes and (not path or path[-1]!=key):path.append(key)
        mapping['roads'][old_rid]['source_lane_path']=path
        rebuilt.append(rid)
    # Build packed-junction movements from retained endpoint tails and source bridges.
    next_id=max(elements)+1
    for edge in direct_edges:
        a,b=edge['source'],edge['target'];jid=edge['junction_id']
        pieces=deepcopy(end_tails.get(a,[]))
        bridge=bridge_geometry(geometry(a),geometry(b))
        if bridge is not None:
            key=f'bridge:{a}->{b}';geometries[key]=bridge;pieces.append([key,0.,bridge.length])
        pieces.extend(deepcopy(start_tails.get(b,[])))
        if not pieces:raise ValueError('Direct source edge has no retained tail interval: '+str([a,b]))
        before=native_ids.get(a,(old_ids[a],-1));after=native_ids.get(b,(old_ids[b],-1))
        rid=next_id;next_id+=1
        classes=sorted(set(lanes[a]['allowed_classes']) & set(lanes[b]['allowed_classes']))
        speed=min(lanes[a]['speed_mps'],lanes[b]['speed_mps'])
        road=make_road(rid,pieces,geometries,classes,speed,jid)
        road.add_predecessor(xodr.ElementType.road,before[0],xodr.ContactPoint.end)
        road.add_successor(xodr.ElementType.road,after[0],xodr.ContactPoint.start)
        road.lanes.lanesections[0].rightlanes[0].add_link('predecessor',before[1]).add_link('successor',after[1])
        replacements[rid]=road.get_element()
        path=[]
        for key,_,_ in pieces:
            if key in lanes and (not path or path[-1]!=key):path.append(key)
        mapping['roads'][str(rid)]=dict(kind='movement',from_lane_id=a,to_lane_id=b,
            source_lane_path=path,source_intervals=pieces,allowed_classes=classes,
            speed_mps=speed,junction_id=jid,original_direct_source_edge=[a,b])
        group=next(g for g in mapping['physical_junctions'] if g['id']==jid);group['movement_road_ids'].append(rid)
        junction=next(e for e in root.findall('junction') if int(e.get('id'))==jid)
        connection=ET.SubElement(junction,'connection',incomingRoad=str(before[0]),
            connectingRoad=str(rid),contactPoint='start',id=str(len(junction.findall('connection'))))
        ET.SubElement(connection,'laneLink',attrib={'from':str(before[1]),'to':'-1'})
        edge.update(movement_road_id=rid,from_native=list(before),to_native=list(after))
        rebuilt.append(rid)
    incoming_map={old_ids[k]:v for k,v in native_ids.items()}
    for junction in root.findall('junction'):
        for connection in junction.findall('connection'):
            if int(connection.get('connectingRoad')) in {e['movement_road_id']for e in direct_edges}:continue
            original_id=int(connection.get('incomingRoad'))
            if original_id in incoming_map:
                rid,lid=incoming_map[original_id];connection.set('incomingRoad',str(rid))
                for link in connection.findall('laneLink'):link.set('from',str(lid))
    for rid,element in elements.items():
        if rid in replacements or rid in removed:root.remove(element)
    position=next((i for i,e in enumerate(root) if e.tag=='junction'),len(root))
    for _,element in sorted(replacements.items()):
        root.insert(position,element);position+=1
    # Rebuild station mappings for affected movements.
    for key,spans in mapping['lane_to_road_intervals'].items():
        mapping['lane_to_road_intervals'][key]=[s for s in spans if s['road_id'] not in removed and s['road_id'] not in rebuilt]
    for key,span in group_spans.items():mapping['lane_to_road_intervals'][key].append(span)
    for rid in rebuilt:
        cursor=0.;meta=mapping['roads'][str(rid)]
        for key,start,end in meta['source_intervals']:
            if key in lanes:
                mapping['lane_to_road_intervals'][key].append(dict(road_id=rid,lane_id=-1,section_id=0,
                    road_s_start_m=cursor,road_s_end_m=cursor+end-start,source_s_start_m=start,source_s_end_m=end,junction_id=meta['junction_id']))
            cursor+=end-start
    for spans in mapping['lane_to_road_intervals'].values():spans.sort(key=lambda r:(r['source_s_start_m'],r['road_id'],r['lane_id']))
    previous_sections=mapping.get('grouped_sections',[])
    previous_splits=mapping.get('return_road_splits',[])
    # New endpoint unions can also turn an existing packed section into a
    # return road; split every affected section using its retained definition.
    mapping['grouped_sections']=previous_sections+sections
    split_return_roads(root,mapping)
    mapping['return_road_splits']=previous_splits+mapping['return_road_splits']
    from scripts.scene.native_group_endpoints import split_retained_return_roads
    split_retained_return_roads(root,mapping,lanes)
    from map_generation.native_source_extents import restore_terminal_extents
    restore_terminal_extents(root,mapping,json.loads((original/'road_lane_ids.json').read_text()),lanes)
    signal_review=restore_signals(root,mapping,original_signals,original/'roads.xodr')
    output.mkdir(parents=True,exist_ok=True)
    (output/'signal_transfer_review.json').write_text(json.dumps(signal_review,indent=2)+'\n')
    tree.write(output/'roads.xodr',encoding='utf-8',xml_declaration=True)
    (output/'road_lane_ids.json').write_text(json.dumps(mapping,indent=2)+'\n')
    (output/'section.json').write_text(json.dumps(sections[0],indent=2)+'\n')
    return review_compiled(scene,original,output,mapping)


def split_return_roads(root,mapping):
    """Avoid CARLA reverse-entry links when a road has the same end junction.

    Two consecutive native road IDs represent the same observed corridor.
    Its first road owns the junction entry and its second owns the exit.
    """
    elements={int(e.get('id')):e for e in root.findall('road')}
    next_id=max(elements)+1;records=[]
    for section in mapping['grouped_sections']:
        rid=section['source_lanes'][0]['road_id'];element=elements[rid]
        pred,succ=element.find('link/predecessor'),element.find('link/successor')
        if pred is None or succ is None or pred.get('elementType')!='junction' or pred.attrib!=succ.attrib:continue
        second=next_id;next_id+=1;start,end=section['reference_source_interval_m'];middle=(start+end)/2;cut=middle-start
        roads=[]
        for native_id,interval in [(rid,(start,middle)),(second,(middle,end))]:
            road,_=compile_section(section['source_lane_records'],road_id=native_id,interval=interval,center_mode=section['center_mode'],spacing=section.get('profile_sample_step_m',.1))
            roads.append(road)
        junction=int(pred.get('elementId'))
        roads[0].add_predecessor(xodr.ElementType.junction,junction)
        roads[0].add_successor(xodr.ElementType.road,second,xodr.ContactPoint.start)
        roads[1].add_predecessor(xodr.ElementType.road,rid,xodr.ContactPoint.end)
        roads[1].add_successor(xodr.ElementType.junction,junction)
        for i in range(len(section['source_lanes'])):
            roads[0].lanes.lanesections[0].rightlanes[i].add_link('successor',-i-1)
            roads[1].lanes.lanesections[0].rightlanes[i].add_link('predecessor',-i-1)
        position=list(root).index(element);root.remove(element)
        for i,road in enumerate(roads):root.insert(position+i,road.get_element())
        for other in root.findall('road'):
            link=other.find('link/predecessor')
            if int(other.get('id'))!=second and link is not None and link.get('elementType')=='road' and link.get('elementId')==str(rid):link.set('elementId',str(second))
        for link in root.findall('junction/connection'):
            if link.get('incomingRoad')==str(rid):link.set('incomingRoad',str(second))
        mapping['roads'][str(second)]=deepcopy(mapping['roads'][str(rid)])
        for entry in mapping['roads'][str(second)]['native_lanes']:entry['road_id']=second
        for key,spans in mapping['lane_to_road_intervals'].items():
            revised=[]
            for span in spans:
                if span['road_id']!=rid:revised.append(span);continue
                for native_id,low,high in [(rid,0.,cut),(second,cut,section['length_m'])]:
                    a,b=max(low,span['road_s_start_m']),min(high,span['road_s_end_m'])
                    if b<=a:continue
                    stations=np.asarray(span['road_stations_m']);source=np.asarray(span['source_stations_m'])
                    selected=np.unique(np.r_[a,stations[(stations>a)&(stations<b)],b]);local=np.interp(selected,stations,source)
                    revised.append(dict(span,road_id=native_id,road_s_start_m=a-low,road_s_end_m=b-low,
                        source_s_start_m=float(local[0]),source_s_end_m=float(local[-1]),road_stations_m=(selected-low).tolist(),source_stations_m=local.tolist()))
            mapping['lane_to_road_intervals'][key]=revised
        section['native_sections']=[dict(road_id=rid,start_m=0.,end_m=cut),dict(road_id=second,start_m=cut,end_m=section['length_m'])]
        records.append(dict(source_road_id=rid,native_road_ids=[rid,second],native_lane_ids=[e['lane_id'] for e in section['source_lanes']],cut_s_m=cut,junction_id=junction,
            reason='Stock CARLA GetJunctionLanes checks both movement endpoints; separate road IDs preserve directed entrance and exit.'))
    mapping['return_road_splits']=records


def review_compiled(scene,original,output,mapping=None):
    """Recheck an existing packed map after a targeted encoding correction."""
    if mapping is None:mapping=json.loads((output/'road_lane_ids.json').read_text())
    previous=json.loads((original/'road_lane_ids.json').read_text())
    incoming_map={}
    previous_groups={tuple(tuple(part['source_lane_id'] for part in source['source_chain'])
        for source in section['source_lane_records']) for section in previous.get('grouped_sections',[])}
    new_sections=[section for section in mapping['grouped_sections'] if tuple(
        tuple(part['source_lane_id'] for part in source['source_chain'])
        for source in section['source_lane_records']) not in previous_groups]
    for section in new_sections:
        for entry,source in zip(section['source_lanes'],section['source_lane_records']):
            for part in source['source_chain']:
                incoming_map[previous['key_to_road_id'][part['source_lane_id']]]=(entry['road_id'],entry['lane_id'])
    report=review_native(scene,mapping,output/'roads.xodr')
    from map_generation.native_source_extents import review_source_extents
    report['source_extents']=review_source_extents(previous,mapping)
    report['passed'] &= report['source_extents']['passed']
    import carla
    before=carla.Map(scene['map_id']+'_before',(original/'roads.xodr').read_text())
    after=carla.Map(scene['map_id']+'_after',(output/'roads.xodr').read_text())
    def identity(wp):return incoming_map.get(wp.road_id,(wp.road_id,wp.lane_id))
    def terminal_segment(a,b):
        return (a.road_id,a.lane_id)==(b.road_id,b.lane_id) and abs(a.s-b.s)>1e-4
    # Distinguish terminal-lane interior samples from successor edges.
    from map_generation.native_topology import topology_with_non_driving
    previous_topology,before_types=topology_with_non_driving(before,original/'roads.xodr')
    current_topology,after_types=topology_with_non_driving(after,output/'roads.xodr')
    (output/'non_driving_topology.json').write_text(json.dumps(dict(before=before_types,after=after_types),indent=2)+'\n')
    report['native_topology_type_scope']=dict(method=after_types['method'],before_non_driving_lanes=before_types['queried_non_driving_lanes'],after_non_driving_lanes=after_types['queried_non_driving_lanes'])
    previous_split_ids={r['source_road_id'] for r in previous.get('return_road_splits',[])}
    splits={r['source_road_id']:r['native_road_ids'] for r in mapping.get('return_road_splits',[])
        if r['source_road_id'] not in previous_split_ids}
    previous_direct={tuple(e.get('original_direct_source_edge',[]))for e in previous['roads'].values()if 'original_direct_source_edge'in e}
    added_direct={ (e['old_from_road_id'],e['old_to_road_id']):e for e in mapping.get('source_endpoint_containers',{}).get('direct_source_edges',[]) if (e['source'],e['target'])not in previous_direct}
    expected=set()
    for a,b in previous_topology:
        first,second=identity(a),identity(b)
        if terminal_segment(a,b) or (first==second and (a.road_id,a.lane_id)!=(b.road_id,b.lane_id)):continue
        first=(splits.get(first[0],[first[0]])[-1],first[1])
        edge=added_direct.get((a.road_id,b.road_id))
        if edge:
            middle=(edge['movement_road_id'],-1);expected.update([(first,middle),(middle,second)])
        else:expected.add((first,second))
    for section in new_sections:
        rid=section['source_lanes'][0]['road_id']
        if rid in splits:
            for entry in section['source_lanes']:expected.add(((rid,entry['lane_id']),(splits[rid][-1],entry['lane_id'])))
    for record in mapping.get('return_road_splits',[]):
        if record['source_road_id']not in previous_split_ids:
            for lid in record.get('native_lane_ids',[]):expected.add(((record['native_road_ids'][0],lid),(record['native_road_ids'][1],lid)))
    previous_tails={tuple(r['native_road_ids'])for r in previous.get('source_terminal_tails',[])}
    for record in mapping.get('source_terminal_tails',[]):
        if tuple(record['native_road_ids'])not in previous_tails:
            expected.update((tuple(a),tuple(b))for a,b in record['expected_native_edges'])
    actual={((a.road_id,a.lane_id),(b.road_id,b.lane_id)) for a,b in current_topology if not terminal_segment(a,b)}
    terminal_continuations=[]
    for a,b in current_topology:
        if terminal_segment(a,b):
            wp=after.get_waypoint_xodr(b.road_id,b.lane_id,b.s+(-.001 if b.lane_id<0 else .001))
            if wp.next(.02):terminal_continuations.append([b.road_id,b.lane_id,b.s])
    report['terminal_lane_segments']=dict(before=sum(terminal_segment(a,b) for a,b in previous_topology),
        after=sum(terminal_segment(a,b) for a,b in current_topology),unexpected_continuations=terminal_continuations)
    report['native_topology_missing']=sorted(expected-actual)
    report['native_topology_extra']=sorted(actual-expected)
    report['passed'] &= expected==actual and not terminal_continuations
    rebuilt=[int(key) for key,meta in mapping['roads'].items() if meta['kind']=='movement' and meta!=previous['roads'].get(key)]
    report.update(group_count=len(mapping['grouped_sections']),source_group_lane_count=len(incoming_map),rebuilt_movement_roads=rebuilt,
        source_network=str(original),scope='Source-supported grouped approaches and all retained junction branches. Open fixture ends remain explicit; whole-city corridor and traffic checks are separate.')
    (output/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def review_native(scene,mapping,path):
    import carla
    from shapely.geometry import LineString,Point,Polygon
    native=carla.Map(scene['map_id'],path.read_text());lanes={l['id']:l for l in scene['fusion_map']['lanes']}
    errors=[];links=[];lateral=[]
    for section in mapping['grouped_sections']:
        records={l['id']:l for l in section.get('source_lane_records',lanes.values())}
        reference=fit_lane(records[section['reference_source_lane_id']],1.)[0]
        for i,entry in enumerate(section['source_lanes']):
            source=records[entry['source_lane_id']]
            line=LineString(source['centerline'])
            boundaries={side:LineString(source[side+'_boundary']) for side in ['left','right']}
            polygon=Polygon([*source['left_boundary'],*reversed(source['right_boundary'])])
            values=[];bad=[];boundary_errors=[];permission_errors=[];compiled_center_errors=[];outside=[]
            epsilon=min(.1,section['length_m']/4)
            for station in np.linspace(epsilon,section['length_m']-epsilon,101):
                for part in section.get('native_sections',[]):
                    if abs(station-part['start_m'])<1e-8:station=part['start_m']
                segment=next((s for s in section.get('native_sections',[]) if s['start_m']<=station<s['end_m']),dict(road_id=entry['road_id'],start_m=0.))
                wp=native.get_waypoint_xodr(segment['road_id'],entry['lane_id'],float(station-segment['start_m']))
                if wp is None:raise ValueError('Native waypoint missing inside compiled section: '+str(dict(source_lane=entry['source_lane_id'],segment=segment,lane_id=entry['lane_id'],station_m=float(station),length_m=section['length_m'])))
                p=wp.transform.location
                values.append(line.distance(Point(p.x,-p.y)))
                normal=normal_at(reference,station+section['reference_source_interval_m'][0])
                origin=reference.xy(station+section['reference_source_interval_m'][0])
                transverse=[intersect_boundary(boundaries[side],origin,normal)[0] for side in ['left','right']]
                desired_offset=sum(transverse)/2 if section['center_mode']=='boundaries' else intersect_boundary(line,origin,normal)[0]
                compiled_center_errors.append(float(np.linalg.norm(np.array([p.x,-p.y])-(origin+normal*desired_offset))))
                if not polygon.covers(Point(p.x,-p.y)):outside.append(float(station))
                for side,target in [('left',i-1),('right',i+1)]:
                    adjacent=getattr(wp,'get_'+side+'_lane')();expected=section['source_lanes'][target]['lane_id'] if 0<=target<len(section['source_lanes']) else None
                    if (adjacent.lane_id if adjacent else None)!=expected:bad.append([station,side])
                    edge=np.array([p.x,-p.y])+(1 if side=='left' else -1)*normal*wp.lane_width/2
                    boundary_errors.append(boundaries[side].distance(Point(edge)))
                    flag=carla.LaneChange.Left if side=='left' else carla.LaneChange.Right
                    profile=entry['marking_profiles'][side]
                    if bool(wp.lane_change & flag)!=(marking_at(profile,station)=='DASHED_WHITE'):permission_errors.append([station,side])
            lateral.append(dict(**entry,max_centerline_xy_error_m=max(values),missing_neighbors=bad,
                max_boundary_xy_error_m=max(boundary_errors),permission_errors=permission_errors,
                max_compiled_center_xy_error_m=max(compiled_center_errors),native_centers_outside_source_polygon=outside))
    # Every retained source edge must have a native continuation with its
    # actual lane ID, including edges represented within one movement road.
    spans=mapping['lane_to_road_intervals']
    for a,lane in lanes.items():
        for b in lane['successors']:
            if b not in lanes:continue
            found=False
            for first in spans[a]:
                if abs(first['source_s_end_m']-max(s['source_s_end_m'] for s in spans[a]))>1e-4:continue
                for second in spans[b]:
                    if second['source_s_start_m']>1e-4:continue
                    f=(first['road_id'],first['lane_id']);s=(second['road_id'],second['lane_id'])
                    if f==s and second['road_s_start_m']>=first['road_s_end_m']-1e-4:found=True;break
                    wp=native.get_waypoint_xodr(*f,first['road_s_end_m']-.001)
                    remaining=first['road_s_end_m']-wp.s
                    if any((w.road_id,w.lane_id)==s for w in wp.next(remaining+min(.02,(second['road_s_end_m']-second['road_s_start_m'])/2))):found=True;break
                if found:break
            links.append(dict(source=a,target=b,reachable=found))
            if not found:errors.append([a,b])
    return dict(passed=not errors and all(not r['missing_neighbors'] and not r['permission_errors']
        and not r['native_centers_outside_source_polygon'] and r['max_compiled_center_xy_error_m']<.03
        and r['max_boundary_xy_error_m']<.03 for r in lateral),
        lateral=lateral,source_edge_count=len(links),missing_source_edges=errors,source_edges=links,
        center_comparison='Check native boundary and native center encoding against the stated cross-section definition, each under 3 cm, and require centers inside the observed source polygon. AV2 ladder-midpoint differences remain separately reported; recorded ego trajectories and physical boundaries stay unchanged.',
        native_topology_edges=len(native.get_topology()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['scene','roads','group','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--margin',type=float,help='Optional diagnostic endpoint margin; defaults to actual shared source/native caps')
    p.add_argument('--center-mode',choices=['source','boundaries'],default='boundaries')
    p.add_argument('--sample-step',type=float,default=.1,help='Width and offset profile sample spacing in metres; source boundary and reference knots are included')
    p.add_argument('--intervals',type=Path,help='Agent-selected finite source-chain section intervals')
    p.add_argument('--section-cache',type=Path,help='Reuse sections compiled for the same source records and native intervals')
    a=p.parse_args();group=json.loads(a.group.read_text());groups=group['groups'] if 'groups' in group else [group['lanes']]
    report=pack(json.loads(a.scene.read_text()),a.roads,groups,a.output,a.margin,a.center_mode,a.sample_step,a.section_cache,
                json.loads(a.intervals.read_text())['sections'] if a.intervals else None)
    print({k:v for k,v in report.items() if k not in ['source_edges','lateral']},flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
