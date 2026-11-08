"""Retain original source extents when grouped roads have open end tails."""
import xml.etree.ElementTree as ET


def interval_union(spans):
    rows=[]
    for a,b in sorted((float(s['source_s_start_m']),float(s['source_s_end_m']))for s in spans):
        if rows and a<=rows[-1][1]+1e-8:rows[-1][1]=max(rows[-1][1],b)
        else:rows.append([a,b])
    return rows


def interval_difference(first,second):
    rows=[]
    for low,high in first:
        parts=[[low,high]]
        for start,end in second:
            revised=[]
            for a,b in parts:
                if end<=a or start>=b:revised.append([a,b])
                else:
                    if a<start:revised.append([a,start])
                    if end<b:revised.append([end,b])
            parts=revised
        rows.extend(parts)
    return rows


def review_source_extents(reference,mapping):
    missing=[]
    for key,spans in reference['lane_to_road_intervals'].items():
        intervals=interval_difference(interval_union(spans),interval_union(mapping['lane_to_road_intervals'].get(key,[])))
        length=sum(b-a for a,b in intervals)
        if length>1e-5:missing.append(dict(source_lane_id=key,missing_intervals_m=intervals,missing_length_m=length))
    return dict(passed=not missing,source_count=len(reference['lane_to_road_intervals']),
        missing_extent_count=len(missing),missing_length_m=sum(r['missing_length_m']for r in missing),missing=missing,
        scope='Union of original source arc intervals versus all current native spans; topology alone does not establish source extent retention.')


def restore_terminal_extents(root,mapping,reference,lanes):
    """Keep evidenced open prefixes/suffixes as exact source road segments.

    Split each retained tail into an external road and a junction segment so a
    packed lane can keep its other lanes' existing endpoint container. The split
    is native representation only; no source successor or permission is added.
    Interior gaps and missing nonterminal source paths remain failures.
    """
    from scenariogeneration import xodr
    from map_generation.road_builder import fit_lane,make_road
    gaps=review_source_extents(reference,mapping)['missing']
    predecessors={key:[]for key in lanes}
    for key,lane in lanes.items():
        for target in lane['successors']:
            if target in predecessors:predecessors[target].append(key)
    elements={int(e.get('id')):e for e in root.findall('road')}
    next_id=max(elements)+1;records=[]
    for gap in gaps:
        key=gap['source_lane_id'];lane=lanes[key];spans=mapping['lane_to_road_intervals'].get(key,[])
        source_intervals=interval_union(reference['lane_to_road_intervals'][key])
        for start,end in gap['missing_intervals_m']:
            if end-start<=1e-5:continue
            prefix=abs(start-source_intervals[0][0])<1e-5 and not predecessors[key]
            suffix=abs(end-source_intervals[-1][1])<1e-5 and not lane['successors']
            if prefix==suffix:raise ValueError('Missing interval is not an observed open source tail: '+key)
            side='start'if prefix else'end';cut=end if prefix else start
            adjacent=[s for s in spans if abs(s['source_s_start_m'if prefix else'source_s_end_m']-cut)<1e-5]
            if len(adjacent)!=1:raise ValueError('Open tail needs a unique retained source section: '+key)
            span=adjacent[0];rid,lid=span['road_id'],span['lane_id'];element=elements[rid]
            road_station=span['road_s_start_m'if prefix else'road_s_end_m']
            if abs(road_station-(0. if prefix else float(element.get('length'))))>1e-4:
                raise ValueError('Open source tail meets an interior native section: '+key)
            link_name='predecessor'if prefix else'successor';link=element.find('link/'+link_name)
            if link is None:
                jid=max(g['id']for g in mapping['physical_junctions'])+1
                links=element.find('link')
                if links is None:links=ET.Element('link');element.insert(0,links)
                ET.SubElement(links,link_name,elementType='junction',elementId=str(jid))
                group=dict(id=jid,source_lane_ids=[key],incoming_lane_ids=[],outgoing_lane_ids=[],movement_road_ids=[],source_sumo_junction_ids=[],uncovered_internal_lane_ids=[],absorbed_unmarked_lane_ids=[])
                mapping['physical_junctions'].append(group)
                junction=ET.SubElement(root,'junction',name='source_terminal_'+str(jid),id=str(jid),type='default')
            else:
                if link.get('elementType')!='junction':raise ValueError('Open tail endpoint has a direct native road link: '+key)
                jid=int(link.get('elementId'));group=next(g for g in mapping['physical_junctions']if g['id']==jid)
                junction=next(j for j in root.findall('junction')if int(j.get('id'))==jid)
            external_id,movement_id=next_id,next_id+1;next_id+=2
            middle=(start+end)/2
            ext_interval=(start,middle)if prefix else(middle,end)
            movement_interval=(middle,end)if prefix else(start,middle)
            geometry=fit_lane(lane,1.)[0];geometries={key:geometry}
            external=make_road(external_id,[[key,*ext_interval]],geometries,lane['allowed_classes'],lane['speed_mps'])
            movement=make_road(movement_id,[[key,*movement_interval]],geometries,lane['allowed_classes'],lane['speed_mps'],jid)
            if prefix:
                external.add_successor(xodr.ElementType.junction,jid)
                movement.add_predecessor(xodr.ElementType.road,external_id,xodr.ContactPoint.end)
                movement.add_successor(xodr.ElementType.road,rid,xodr.ContactPoint.start)
                before,after=(external_id,-1),(rid,lid)
            else:
                external.add_predecessor(xodr.ElementType.junction,jid)
                movement.add_predecessor(xodr.ElementType.road,rid,xodr.ContactPoint.end)
                movement.add_successor(xodr.ElementType.road,external_id,xodr.ContactPoint.start)
                before,after=(rid,lid),(external_id,-1)
            movement.lanes.lanesections[0].rightlanes[0].add_link('predecessor',before[1]).add_link('successor',after[1])
            position=next((i for i,e in enumerate(root)if e.tag=='junction'),len(root))
            root.insert(position,external.get_element());root.insert(position+1,movement.get_element())
            connection=ET.SubElement(junction,'connection',id=str(len(junction.findall('connection'))),incomingRoad=str(before[0]),connectingRoad=str(movement_id),contactPoint='start')
            ET.SubElement(connection,'laneLink',attrib={'from':str(before[1]),'to':'-1'})
            mapping['roads'][str(external_id)]=dict(kind='lane',scene_lane_id=key,source_interval_m=list(ext_interval),allowed_classes=lane['allowed_classes'],speed_mps=lane['speed_mps'],junction_id=None,source_terminal_tail=True)
            mapping['roads'][str(movement_id)]=dict(kind='movement',from_lane_id=key,to_lane_id=key,source_lane_path=[key],source_intervals=[[key,*movement_interval]],allowed_classes=lane['allowed_classes'],speed_mps=lane['speed_mps'],junction_id=jid,source_terminal_tail=True)
            mapping['road_lane_ids'][str(external_id)]=key
            group['movement_road_ids'].append(movement_id)
            for native_id,interval,native_jid in [(external_id,ext_interval,None),(movement_id,movement_interval,jid)]:
                spans.append(dict(road_id=native_id,lane_id=-1,section_id=0,road_s_start_m=0.,road_s_end_m=interval[1]-interval[0],source_s_start_m=interval[0],source_s_end_m=interval[1],junction_id=native_jid))
            spans.sort(key=lambda s:(s['source_s_start_m'],s['road_id'],s['lane_id']))
            records.append(dict(source_lane_id=key,side=side,source_interval_m=[start,end],native_road_ids=[external_id,movement_id],
                source_graph_changed=False,expected_native_edges=[[list(before),[movement_id,-1]],[[movement_id,-1],list(after)]]))
    mapping.setdefault('source_terminal_tails',[]).extend(records)
    return records
