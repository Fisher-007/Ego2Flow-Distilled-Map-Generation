"""Normalize packed-road endpoint containers while retaining source intervals."""
from collections import defaultdict
import xml.etree.ElementTree as ET


def normalize_endpoints(root, mapping, groups):
    elements={int(e.get('id')):e for e in root.findall('road')}
    ids=mapping['key_to_road_id'];parent={};active=set();endpoint_nodes={};direct={}
    def find(a):
        parent.setdefault(a,a)
        if parent[a]!=a:parent[a]=find(parent[a])
        return parent[a]
    def join(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[b]=a
    for group in mapping['physical_junctions']:
        token=('junction',group['id']);find(token);active.add(token)
        for side,field in [('end','incoming_lane_ids'),('start','outgoing_lane_ids')]:
            for key in group[field]:
                node=(key,side);endpoint_nodes[node]=node;join(token,node)
    for supplied in groups:
        chains=[[k]if isinstance(k,str)else k for k in supplied]
        for side,link in [('start','predecessor'),('end','successor')]:
            nodes=[(chain[0 if side=='start'else -1],side)for chain in chains]
            for node in nodes:
                endpoint_nodes[node]=node;find(node);join(nodes[0],node)
                key=node[0];edge=elements[ids[key]].find('link/'+link)
                if edge is None or edge.get('elementType')!='road':continue
                rid=int(edge.get('elementId'));meta=mapping['roads'][str(rid)]
                other=meta.get('scene_lane_id')
                if meta['kind']!='lane' or other is None:
                    raise ValueError('Direct packed endpoint is not an original source lane: '+key)
                opposite='end'if side=='start'else 'start'
                neighbor=(other,opposite);endpoint_nodes[neighbor]=neighbor
                join(node,neighbor);active.add(node)
                a,b=(other,key)if side=='start'else(key,other)
                direct[a,b]=dict(source=a,target=b,old_from_road_id=ids[a],old_to_road_id=ids[b])
    old_by_root=defaultdict(list)
    for group in mapping['physical_junctions']:old_by_root[find(('junction',group['id']))].append(group)
    active_roots={find(n)for n in active}
    next_jid=max(g['id']for g in mapping['physical_junctions'])+1
    allocated={};merged=[]
    for token in sorted(active_roots,key=str):
        old=old_by_root[token]
        jid=min(g['id']for g in old)if old else next_jid
        if not old:next_jid+=1
        allocated[token]=jid
        combined=dict(id=jid,source_lane_ids=[],incoming_lane_ids=[],outgoing_lane_ids=[],
            movement_road_ids=[],source_sumo_junction_ids=[],uncovered_internal_lane_ids=[],absorbed_unmarked_lane_ids=[])
        for field in list(combined):
            if field=='id':continue
            combined[field]=sorted({value for g in old for value in g.get(field,[])})
        combined['native_container_source_junction_ids']=[g['id']for g in old]
        merged.append(combined)
    jid_map={g['id']:allocated[find(('junction',g['id']))]for g in mapping['physical_junctions']}
    ends={node:allocated[find(node)]for node in endpoint_nodes if find(node)in allocated}
    by_id={g['id']:g for g in merged}
    for (key,side),jid in ends.items():
        field='incoming_lane_ids'if side=='end'else'outgoing_lane_ids'
        if key not in by_id[jid][field]:by_id[jid][field].append(key)
    original_junctions=list(root.findall('junction'));containers={}
    for junction in original_junctions:
        jid=jid_map[int(junction.get('id'))]
        if jid not in containers:
            junction.set('id',str(jid));junction.set('name','source_container_'+str(jid));containers[jid]=junction
        else:
            containers[jid].extend(list(junction));root.remove(junction)
    for jid in by_id:
        if jid not in containers:containers[jid]=ET.SubElement(root,'junction',name='source_container_'+str(jid),id=str(jid),type='default')
    for junction in containers.values():
        for i,connection in enumerate(junction.findall('connection')):connection.set('id',str(i))
    for element in elements.values():
        value=int(element.get('junction','-1'))
        if value in jid_map:element.set('junction',str(jid_map[value]))
        for link in element.findall('link/*'):
            if link.get('elementType')=='junction':link.set('elementId',str(jid_map[int(link.get('elementId'))]))
    direct_nodes={(a,'end')for a,b in direct}|{(b,'start')for a,b in direct}
    for (key,side),jid in ends.items():
        if (key,side)not in direct_nodes:continue
        if key not in ids:continue
        element=elements[ids[key]];links=element.find('link')
        if links is None:links=ET.Element('link');element.insert(0,links)
        name='successor'if side=='end'else'predecessor';old=links.find(name)
        if old is not None:links.remove(old)
        ET.SubElement(links,name,elementType='junction',elementId=str(jid))
    for meta in mapping['roads'].values():
        if meta.get('junction_id')in jid_map:meta['junction_id']=jid_map[meta['junction_id']]
    for spans in mapping['lane_to_road_intervals'].values():
        for span in spans:
            if span.get('junction_id')in jid_map:span['junction_id']=jid_map[span['junction_id']]
    mapping['physical_junctions']=merged
    rows=[]
    for (a,b),row in direct.items():
        row['junction_id']=ends[a,'end'];rows.append(row)
    previous=mapping.get('source_endpoint_containers',{})
    old_edges=previous.get('direct_source_edges',[])
    for edge in old_edges:edge['junction_id']=jid_map.get(edge['junction_id'],edge['junction_id'])
    mapping['source_endpoint_containers']=dict(junction_id_map={**previous.get('junction_id_map',{}),**{str(k):v for k,v in jid_map.items()if k!=v}},
        direct_source_edges=old_edges+rows,scope='Native containers merge only. All original source movements and direct edge directions stay distinct.')
    return ends,rows


def split_retained_return_roads(root, mapping, lanes):
    # Split native entrance and exit IDs for roads joining the same container at both ends.
    from copy import deepcopy
    from scenariogeneration import xodr
    from map_generation.road_builder import fit_lane,make_road
    elements={int(e.get('id')):e for e in root.findall('road')}
    next_id=max(elements)+1;records=[]
    for rid,element in list(elements.items()):
        pred,succ=element.find('link/predecessor'),element.find('link/successor')
        if pred is None or succ is None or pred.get('elementType')!='junction' or pred.attrib!=succ.attrib:continue
        meta=mapping['roads'][str(rid)]
        if meta['kind']!='lane':raise ValueError('Unsupported retained return-road kind: '+meta['kind'])
        key=meta['scene_lane_id'];geometry=fit_lane(lanes[key],1.)[0]
        start,end=meta['source_interval_m'];middle=(start+end)/2;cut=middle-start
        second=next_id;next_id+=1;junction=int(pred.get('elementId'))
        roads=[]
        for native_id,low,high in [(rid,start,middle),(second,middle,end)]:
            roads.append(make_road(native_id,[[key,low,high]],{key:geometry},meta['allowed_classes'],meta['speed_mps']))
        roads[0].add_predecessor(xodr.ElementType.junction,junction)
        roads[0].add_successor(xodr.ElementType.road,second,xodr.ContactPoint.start)
        roads[1].add_predecessor(xodr.ElementType.road,rid,xodr.ContactPoint.end)
        roads[1].add_successor(xodr.ElementType.junction,junction)
        roads[0].lanes.lanesections[0].rightlanes[0].add_link('successor',-1)
        roads[1].lanes.lanesections[0].rightlanes[0].add_link('predecessor',-1)
        position=list(root).index(element);root.remove(element)
        for i,road in enumerate(roads):root.insert(position+i,road.get_element())
        for other in root.findall('road'):
            link=other.find('link/predecessor')
            if int(other.get('id'))!=second and link is not None and link.get('elementType')=='road' and link.get('elementId')==str(rid):
                link.set('elementId',str(second))
        for connection in root.findall('junction/connection'):
            if connection.get('incomingRoad')==str(rid):connection.set('incomingRoad',str(second))
        mapping['roads'][str(second)]=dict(deepcopy(meta),source_interval_m=[middle,end])
        meta['source_interval_m']=[start,middle]
        if str(rid)in mapping['road_lane_ids']:mapping['road_lane_ids'][str(second)]=deepcopy(mapping['road_lane_ids'][str(rid)])
        revised=[]
        for span in mapping['lane_to_road_intervals'][key]:
            if span['road_id']!=rid:revised.append(span);continue
            for native_id,low,high in [(rid,0.,cut),(second,cut,end-start)]:
                a,b=max(low,span['road_s_start_m']),min(high,span['road_s_end_m'])
                if b<=a:continue
                revised.append(dict(span,road_id=native_id,road_s_start_m=a-low,road_s_end_m=b-low,
                    source_s_start_m=start+a,source_s_end_m=start+b))
        mapping['lane_to_road_intervals'][key]=revised
        records.append(dict(source_road_id=rid,native_road_ids=[rid,second],native_lane_ids=[-1],
            cut_s_m=cut,junction_id=junction,source_lane_id=key,
            reason='Distinct native entrance/exit IDs avoid stock CARLA reverse junction entry; exact source interval and original movements retained.'))
    mapping.setdefault('return_road_splits',[]).extend(records)
    return records
