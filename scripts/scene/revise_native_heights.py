"""Update native Z profiles after source-height feedback, preserving the road graph.

Reuse accepted XY, lane widths, permissions, IDs and source station mappings.
Only elevation rows and corresponding physical cross-section heights change.
"""
import argparse,json
from copy import deepcopy
from pathlib import Path
from lxml import etree
import numpy as np
from shapely.geometry import LineString,Point
from map_generation.road_builder import fit_lane,bridge_geometry,stable_elevation_rows


def retain_inactive_compile_spans(before,after,inactive_ids,source_lanes=None):
    """Carry current endpoint heights through withdrawn native placeholders.

    Their internal lanes remain withdrawn from the published source graph.
    Native movement roads still include their station spans between live caps,
    so these temporary spans follow the current approach height corrections.
    """
    old={l['id']:l for l in before['fusion_map']['lanes']}
    new={l['id']:l for l in after['fusion_map']['lanes']}
    links={l['id']:l for l in (source_lanes or before['fusion_map']['lanes'])}
    records=[];restored=[]
    for key in sorted(inactive_ids):
        lane=deepcopy(old[key]);points=np.asarray(lane['centerline'],float)
        previous=[k for k,l in links.items()if key in l['successors']and k in new]
        following=[k for k in links[key]['successors']if k in new]
        if not previous or not following:
            raise ValueError('Withdrawn compile span needs reviewed active approach endpoints: '+key)
        shifts=[[new[k]['centerline'][at][2]-old[k]['centerline'][at][2]for k in keys]
                for keys,at in ((previous,-1),(following,0))]
        endpoints=[float(np.mean(v))for v in shifts]
        stations=np.r_[0.,np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1).cumsum()]
        points[:,2]+=np.interp(stations,[0.,stations[-1]],endpoints)
        lane['centerline']=points.tolist();restored.append(lane)
        records.append(dict(source_lane_id=key,endpoint_height_changes_m=endpoints,
                            endpoint_change_spread_m=[float(np.ptp(v))for v in shifts]))
    result=dict(after,fusion_map=dict(after['fusion_map'],lanes=after['fusion_map']['lanes']+restored))
    return result,dict(spans=records,scope='Compilation-only inactive spans; source withdrawal and final native lane restrictions remain required.')


def retained_profile_interval(rows, low, high):
    """Slice an existing native polynomial profile without refitting its height."""
    starts=np.asarray([r[0] for r in rows],float)
    cuts=np.r_[low,starts[(starts>low+1e-10)&(starts<high-1e-10)]]
    result=[]
    for at in cuts:
        i=max(0,int(np.searchsorted(starts,at,side='right')-1))
        station,a,b,c,d=rows[i];t=float(at-station)
        result.append((float(at),a+b*t+c*t*t+d*t*t*t,b+2*c*t+3*d*t*t,c+3*d*t,d))
    return result


def add_profile_delta(original, before, after, low, high):
    """Add the exact source polynomial change to the existing native profile.

    Sum coefficients at common breakpoints to preserve contact corrections.
    """
    tables=[np.asarray(rows,float) for rows in (original,before,after)]
    cuts=np.unique(np.concatenate([[low]]+[t[(t[:,0]>low+1e-10)&(t[:,0]<high-1e-10),0] for t in tables]))
    output=[]
    for at in cuts:
        coefficients=[]
        for table in tables:
            i=max(0,int(np.searchsorted(table[:,0],at,side='right')-1))
            station,a,b,c,d=table[i];x=at-station
            coefficients.append(np.array([a+x*(b+x*(c+x*d)),b+x*(2*c+3*x*d),c+3*x*d,d]))
        value=coefficients[0]+coefficients[2]-coefficients[1]
        output.append((float(at),*map(float,value)))
    return output


def revise(before,after,roads,output):
    old={l['id']:l for l in before['fusion_map']['lanes']}
    new={l['id']:l for l in after['fusion_map']['lanes']}
    if set(old)!=set(new):raise ValueError('Height feedback changed source identities')
    deltas={}
    for key,lane in new.items():
        a=np.asarray(old[key]['centerline']);b=np.asarray(lane['centerline'])
        if not np.array_equal(a[:,:2],b[:,:2])or old[key]['successors']!=lane['successors']:
            raise ValueError('Height feedback changed source XY/topology: '+key)
        s=np.r_[0.,np.linalg.norm(np.diff(a[:,:2],axis=0),axis=1).cumsum()]
        dense=lane.get('native_height_projection',{}).get('source_station_heights_m')
        if dense:
            dense=np.asarray(dense,float);grid=dense[:,0]
            previous_dense=old[key].get('native_height_projection',{}).get('source_station_heights_m')
            if previous_dense:
                previous_dense=np.asarray(previous_dense,float)
                previous_height=np.interp(grid,previous_dense[:,0],previous_dense[:,1])
            else:previous_height=np.interp(grid,s,a[:,2])
            dz=dense[:,1]-previous_height
        else:
            grid=s;dz=b[:,2]-a[:,2]
        if abs(dz).max()>1e-7:
            if lane['source']=='av2':raise ValueError('Observed AV2 Z changed: '+key)
            deltas[key]=(grid,dz)
    mapping=json.loads((roads/'road_lane_ids.json').read_text())
    profiles={};geometries={};old_geometries={};section_changes=[]
    def geometry(key):
        if key not in geometries:
            if key.startswith('bridge:'):
                a,b=key[len('bridge:'):].split('->');geometries[key]=bridge_geometry(geometry(a),geometry(b))
            else:geometries[key]=fit_lane(new[key],1.)[0]
        return geometries[key]
    def old_geometry(key):
        if key not in old_geometries:
            if key.startswith('bridge:'):
                a,b=key[len('bridge:'):].split('->')
                old_geometries[key]=bridge_geometry(old_geometry(a),old_geometry(b))
            else:
                old_geometries[key]=fit_lane(old[key],1.)[0]
        return old_geometries[key]
    def affected(key):
        return key in deltas or (key.startswith('bridge:')and any(k in deltas for k in key[len('bridge:'):].split('->')))
    def record_delta(record,at):
        at=np.asarray(at);result=np.zeros(at.shape)
        for part in record['source_chain']:
            key=part['source_lane_id']
            if key not in deltas:continue
            low,high=part['chord_start_m'],part['chord_end_m'];selected=(at>=low-1e-7)&(at<=high+1e-7)
            grid,delta=deltas[key];result[selected]=np.interp(at[selected]-low,grid,delta)
        return result
    for section in mapping['grouped_sections']:
        records=section['source_lane_records']
        if not any(p['source_lane_id']in deltas for r in records for p in r['source_chain']):continue
        samples=section['samples'];widths=np.asarray([s['widths_m']for s in samples]);corrections=[]
        reference_record=next(r for r in records if r['id']==section['reference_source_lane_id'])
        for i,record in enumerate(records):
            at=np.asarray([s['source_center_chord_stations_m'][i]for s in samples]);corrections.append(record_delta(record,at))
            xyz=np.asarray(record['centerline']);line=LineString(xyz[:,:2]);grid=np.r_[0.,np.linalg.norm(np.diff(xyz[:,:2],axis=0),axis=1).cumsum()]
            for side in ['left_boundary','right_boundary']:
                points=np.asarray(record[side]);ss=np.array([line.project(Point(p))for p in points[:,:2]])
                points[:,2]+=record_delta(record,ss);record[side]=points.tolist()
            xyz[:,2]+=record_delta(record,grid);record['centerline']=xyz.tolist()
            chain=record['source_chain']
            if len(chain)==1:
                key=chain[0]['source_lane_id']
                if key in deltas and new[key].get('native_height_projection',{}).get('source_station_heights_m'):
                    if not np.array_equal(np.asarray(record['centerline'])[:,:2],np.asarray(new[key]['centerline'])[:,:2]):
                        raise ValueError('Dense grouped height record needs exact source planar representation: '+key)
                    record['parametric_elevation_rows']=deepcopy(new[key]['parametric_elevation_rows'])
                    record['native_height_projection']=deepcopy(new[key]['native_height_projection'])
            elif any(new[p['source_lane_id']].get('native_height_projection',{}).get('source_station_heights_m') for p in chain if p['source_lane_id'] in deltas):
                raise ValueError('Dense grouped height projection requires explicit chained source station composition')
        dz=np.array(corrections).T
        shared=(dz[:,:-1]*widths[:,1:]+dz[:,1:]*widths[:,:-1])/(widths[:,:-1]+widths[:,1:])
        boundary_delta=np.c_[dz[:,0],shared,dz[:,-1]]
        reference=fit_lane(reference_record,1.)[0];start,end=section['reference_source_interval_m']
        for sample,change in zip(samples,boundary_delta):
            sample['reference_xyz'][2]=float(reference.elevations(start+sample['s_m']))
            sample['boundary_z_m']=(np.asarray(sample['boundary_z_m'])+change).tolist()
        parts=section.get('native_sections',[dict(road_id=section['source_lanes'][0]['road_id'],start_m=0.,end_m=end-start)])
        for part in parts:
            rid=part['road_id'];low,high=start+part['start_m'],start+part['end_m']
            profiles[rid]=stable_elevation_rows(reference.elevation_rows(low,high,0),high-low)[0]
        section_changes.append(dict(source_road_id=section['source_lanes'][0]['road_id'],maximum_boundary_z_change_m=float(abs(boundary_delta).max())))
    affected_pieces={}
    for rid,meta in mapping['roads'].items():
        if meta['kind']=='lane_group':continue
        pieces=([(meta['scene_lane_id'],*meta['source_interval_m'])]if meta['kind']=='lane'else meta['source_intervals'])
        if any(affected(k)for k,_,_ in pieces):affected_pieces[int(rid)]=pieces
    original_profiles={}
    for _,element in etree.iterparse(str(roads/'roads.xodr'),events=('end',),tag='road',huge_tree=True):
        rid=int(element.get('id'))
        if rid in affected_pieces:
            original_profiles[rid]=[tuple(float(row.get(k))for k in ('s','a','b','c','d'))
                for row in element.findall('elevationProfile/elevation')] or [(0.,0.,0.,0.,0.)]
        element.clear()
        while element.getprevious()is not None:del element.getparent()[0]
    retained_intervals=[]
    for rid,pieces in affected_pieces.items():
        cursor=0.;rows=[]
        for key,start,end in pieces:
            length=end-start
            if affected(key):
                previous_source=old_geometry(key).elevation_rows(start,end,cursor)
                revised_source=geometry(key).elevation_rows(start,end,cursor)
                rows.extend(add_profile_delta(original_profiles[rid],previous_source,
                                              revised_source,cursor,cursor+length))
            else:
                rows.extend(retained_profile_interval(original_profiles[rid],cursor,cursor+length))
                retained_intervals.append(dict(road_id=rid,source_id=key,start_m=cursor,end_m=cursor+length))
            cursor+=length
        profiles[rid]=stable_elevation_rows(rows,cursor)[0]
    output.mkdir(parents=True,exist_ok=True)
    iterator=etree.iterparse(str(roads/'roads.xodr'),events=('start','end'),huge_tree=True);_,root=next(iterator);written=set()
    with etree.xmlfile(str(output/'roads.xodr'),encoding='utf-8')as writer:
        writer.write_declaration()
        with writer.element(root.tag,root.attrib):
            for event,element in iterator:
                if event!='end'or element.getparent()is not root:continue
                if element.tag=='road'and int(element.get('id'))in profiles:
                    rid=int(element.get('id'));profile=element.find('elevationProfile')
                    for row in list(profile):profile.remove(row)
                    for row in profiles[rid]:etree.SubElement(profile,'elevation',dict(zip(('s','a','b','c','d'),map(str,row))))
                    written.add(rid)
                writer.write(element);element.clear()
                while element.getprevious()is not None:del root[0]
    if written!=set(profiles):raise ValueError('Some revised native road IDs were not written')
    (output/'road_lane_ids.json').write_text(json.dumps(mapping,separators=(',',':'))+'\n')
    report=dict(changed_source_lane_ids=sorted(deltas),changed_native_road_ids=sorted(written),
        changed_group_sections=section_changes,changed_sources=len(deltas),changed_native_roads=len(written),
        native_graph_XY_widths_and_permissions_unchanged=True,
        unchanged_native_source_intervals_retained=retained_intervals,
        individual_profile_update="Existing native polynomial plus exact revised-minus-previous source polynomial; prior contact corrections retained.",
        method=__doc__,scope='Z encoding update. Re-sample affected native surfaces and verify physical joins and driving.')
    (output/'height_revision.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items()if k not in ['changed_source_lane_ids','changed_native_road_ids','changed_group_sections','unchanged_native_source_intervals_retained','method']},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['previous-scene','scene','roads','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();revise(json.loads(a.previous_scene.read_text()),json.loads(a.scene.read_text()),a.roads,a.output)

if __name__=='__main__':main()
