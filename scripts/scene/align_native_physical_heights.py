"""Align OpenDRIVE elevation profiles to the selected physical road surface.

Packed lanes share one reference elevation; transverse residuals are reported.
"""
import argparse,json,shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
from lxml import etree


def align(xodr,mapping,samples_file,conditions_file,output):
    condition=json.loads(conditions_file.read_text())
    setting=condition['native_height_alignment']
    spacing=float(setting['station_spacing_m'])
    threshold=float(setting['minimum_reference_change_m'])
    agreement=float(setting['unowned_candidate_agreement_m'])
    samples=json.loads(samples_file.read_text());by_road=defaultdict(list)
    for row in samples:by_road[int(row['road_id'])].append(row)
    lengths={}
    for _,e in etree.iterparse(str(xodr),events=('end',),tag='road',huge_tree=True):
        lengths[int(e.get('id'))]=float(e.get('length'));e.clear()
        while e.getprevious()is not None:del e.getparent()[0]
    output.mkdir(parents=True,exist_ok=True)
    profiles={};reviews=[];retained=[]
    for rid,rows in sorted(by_road.items()):
        unresolved=[r for r in rows if r.get('surface_z_m')is None or
            (r['selection']!='source_owner'and r['candidate_z_spread_m']>agreement)]
        if unresolved:
            retained.append(dict(road_id=rid,samples=len(rows),unresolved_samples=unresolved,
                action='Retain previous native elevation; source-owner or single-floor evidence is required.'))
            continue
        length=lengths[rid];grid=np.linspace(0.,length,max(2,int(np.ceil(length/spacing))+1))
        lanes=defaultdict(list)
        for row in rows:lanes[row['lane_id']].append(row)
        floors=[];native=[]
        for lane,values in sorted(lanes.items()):
            values=sorted(values,key=lambda r:r['s_m'])
            s=np.array([r['s_m']for r in values]);z=np.array([r['surface_z_m']for r in values])
            floors.append(np.interp(grid,s,z))
            native.append(np.interp(grid,s,[r['native_xyz_m'][2]for r in values]))
        floors=np.asarray(floors)
        # CARLA uses a common reference Z for packed lanes. The center of their
        # height range minimizes the largest transverse waypoint residual.
        target=(floors.min(0)+floors.max(0))/2
        old=np.mean(native,axis=0);change=float(abs(target-old).max())
        if change<=threshold:continue
        slope=np.diff(target)/np.diff(grid)
        profile=np.c_[grid[:-1],target[:-1],slope,np.zeros((len(slope),2))]
        profiles[rid]=profile.tolist()
        predicted=[float(np.interp(r['s_m'],grid,target))for r in rows]
        residual=np.array([r['surface_z_m']-z for r,z in zip(rows,predicted)])
        reviews.append(dict(road_id=rid,samples=len(rows),lane_ids=sorted(lanes),
            source_lane_ids=sorted({key for r in rows for key in r['source_lane_ids']}),
            max_reference_change_m=change,max_encoded_grade=float(abs(slope).max()),
            max_physical_residual_m=float(abs(residual).max()),
            transverse_height_spread_max_m=float(np.ptp(floors,axis=0).max()),
            source_owned_samples=sum(r['selection']=='source_owner'for r in rows),
            single_floor_samples=sum(r['selection']!='source_owner'for r in rows)))
    iterator=etree.iterparse(str(xodr),events=('start','end'),huge_tree=True);_,root=next(iterator);written=set()
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
    assert written==set(profiles)
    shutil.copyfile(mapping,output/'road_lane_ids.json')
    report=dict(input_xodr=str(xodr),mapping=str(mapping),samples=str(samples_file),conditions=str(conditions_file),
        changed_roads=reviews,retained_ambiguous_roads=retained,changed_native_roads=len(reviews),
        retained_ambiguous_native_roads=len(retained),physical_geometry_modified=False,
        native_XY_topology_permissions_widths_unchanged=True,
        runtime_validation_pending=True,method=__doc__,
        scope='Native reference-height representation of reviewed physical support. Original source heights remain in source artifacts; no physical road is raised to a competing surface.')
    (output/'alignment.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k not in ['changed_roads','retained_ambiguous_roads','method']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['xodr','mapping','samples','conditions','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();align(a.xodr,a.mapping,a.samples,a.conditions,a.output)


if __name__=='__main__':main()
