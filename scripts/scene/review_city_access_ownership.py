"""Measure inferred-access ownership against directed AV2 paths."""
import argparse,heapq,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from shapely.geometry import LineString
from map_generation.access_fusion import restore_access,section
from scripts.scene.realize_city_source_components import observed_source_ids


def observed_paths(lanes,start,targets,maximum=150.):
    if start not in lanes or lanes[start]['source']!='av2':return []
    queue=[(0.,start,[start])];seen=set();result=[]
    while queue:
        length,key,path=heapq.heappop(queue)
        if key in seen or length>maximum:continue
        seen.add(key)
        if key in targets and len(path)>1:
            modes=sorted(set.intersection(*(set(lanes[k]['allowed_classes'])for k in path)))
            result.append(dict(source=start,target=key,path=path,length_m=length,permissions=modes))
        for target in lanes[key]['successors']:
            if target not in lanes or lanes[target]['source']!='av2':continue
            cost=LineString(np.asarray(lanes[target]['centerline'])[:,:2]).length
            heapq.heappush(queue,(length+cost,target,path+[target]))
    return result


def review(scene,batch,selection):
    lanes={l['id']:l for l in scene['fusion_map']['lanes']}
    observed=observed_source_ids(scene)
    source_lanes={k:dict(v,source='av2')for k,v in lanes.items()if k in observed}
    accesses=scene['fusion_map']['source_access_review']['connections']
    regions=[];ports={}
    for name in selection['regions']:
        condition=json.loads((batch/name/'condition.json').read_text())
        parts={v['lane_id']for p in condition['paths']for v in p['segments']}
        selected=[a for a in accesses if a['seam_id']in parts]
        for a in selected:ports[a['coarse_lane_id'],a['incoming']]=dict(coarse_lane_id=a['coarse_lane_id'],incoming=a['incoming'])
        fixed=[b for b in condition['boundaries']if b['fixed']]
        regions.append(dict(region=name,condition=str(batch/name/'condition.json'),
            source_parts=sorted(parts),source_kinds=sorted({lanes[k]['source']for k in parts}),
            fixed_boundaries=fixed,current_accesses=selected,bbox=condition['region']['bbox_local_xy']))
    proposals=restore_access(scene,dict(source_ports=list(ports.values()),include_fine_junctions=True,
        max_path_m=80.,alignment_distance_m=3.,minimum_transition_m=selection.get("minimum_transition_m",0.),propose_only=True))
    network=Path(scene['coarse_map']['network'])
    connections=[c.attrib for c in ET.parse(network).getroot().findall('connection')]
    coarse={l['id']:l for l in scene['coarse_map']['lanes']}
    for row in regions:
        identities={(a['coarse_lane_id'],a['incoming'])for a in row['current_accesses']}
        row['proposed_accesses']=[a for a in proposals['connections']if(a['coarse_lane_id'],a['incoming'])in identities]
        starts={b['neighbor_id']for b in row['fixed_boundaries']if b['end']==0}
        goals={b['neighbor_id']for b in row['fixed_boundaries']if b['end']==1}
        starts.update(a['fine_lane_id']for a in row['proposed_accesses']if not a['incoming'])
        goals.update(a['fine_lane_id']for a in row['proposed_accesses']if a['incoming'])
        row['observed_paths']=[p for k in starts for p in observed_paths(source_lanes,k,goals)]
        context=(starts|goals)&observed;frontier=set(context)
        for _ in range(selection.get('context_depth',0)):
            next_ids={t for k in frontier for t in lanes[k]['successors']if t in observed}
            next_ids.update(k for k,l in source_lanes.items()if any(t in frontier for t in l['successors']))
            frontier=next_ids-context;context.update(frontier)
        row['source_context']=[dict(id=k,source=lanes[k]['source'],successors=lanes[k]['successors'],
            start=lanes[k]['centerline'][0],end=lanes[k]['centerline'][-1],allowed_classes=lanes[k]['allowed_classes'])for k in sorted(context)]
        source_ids={k for a in row['current_accesses']for k,_,_ in a['source_path']}
        row['source_connections']=[c for c in connections if c.get('via')in source_ids]
        row['source_road_tags']={k:coarse[k].get('source_tags',{})for k in source_ids if not coarse[k].get('internal')}
    return dict(regions=regions,proposals=proposals,plot_options={k:selection[k]for k in ['context_depth','context_padding_m','label_context','draw_proposed_source_paths','minimum_transition_m']if k in selection},interpretation='Minimum transition is zero only for source ownership evidence. Proposals are not generated road geometry. The Agent must verify complete observed movement, permissions and source coverage before any replacement.')


def plot(scene,report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lanes={l['id']:l for l in scene['fusion_map']['lanes']}
    arrays={k:np.asarray(l['centerline'])for k,l in lanes.items()}
    originals={l['id']:l for l in scene['coarse_map']['lanes']}
    def draw(ax,k,color,width=1.,label=False):
        p=arrays[k];ax.plot(p[:,0],p[:,1],color=color,lw=width,alpha=.85)
        middle=max(1,len(p)//2);v=p[middle,:2]-p[middle-1,:2];length=np.linalg.norm(v)
        if length>0:ax.arrow(*p[middle-1,:2],*(v/length*1.8),head_width=.65,color=color,length_includes_head=True)
        if label:ax.text(*p[middle,:2],k,fontsize=5,color=color,clip_on=True)
    for page in range((len(report['regions'])+3)//4):
        fig,axes=plt.subplots(2,2,figsize=(16,14))
        for ax,row in zip(axes.flat,report['regions'][page*4:page*4+4]):
            x0,y0,x1,y1=row['bbox'];pad=report.get('plot_options',{}).get('context_padding_m',15.)
            for k,p in arrays.items():
                if p[:,0].max()<x0-pad or p[:,0].min()>x1+pad or p[:,1].max()<y0-pad or p[:,1].min()>y1+pad:continue
                if lanes[k]['source']=='av2':draw(ax,k,'silver',.7)
            if report.get('plot_options',{}).get('label_context'):
                for item in row.get('source_context',[]):draw(ax,item['id'],'C2',1.1,True)
            for k in row['source_parts']:draw(ax,k,'C0',2)
            for path in row['observed_paths']:
                for k in path['path']:draw(ax,k,'C2',1.5)
            for a in row['proposed_accesses']:
                draw(ax,a['fine_lane_id'],'C3',2,True)
                if report.get('plot_options',{}).get('draw_proposed_source_paths'):
                    for source,start,end in a['source_path']:
                        if abs(end-start)<1e-6:continue
                        points=section(originals[source]['centerline'],min(start,end),max(start,end))
                        if end<start:points=points[::-1]
                        ax.plot(points[:,0],points[:,1],color='C1',lw=1.5,ls='--')

            for b in row['fixed_boundaries']:
                k=b['neighbor_id']
                if k in lanes:draw(ax,k,'black',1.5,True)
            ax.set_xlim(x0-pad,x1+pad);ax.set_ylim(y0-pad,y1+pad);ax.set_aspect('equal');ax.grid()
            ax.set_title(row['region']+': inferred blue; observed path green; proposed source red')
        for ax in list(axes.flat)[len(report['regions'][page*4:page*4+4]):]:ax.axis('off')
        fig.tight_layout();fig.savefig(output/f'ownership_{page:02}.png',dpi=125);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['scene','batch-root','selection','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();scene=json.loads(a.scene.read_text())
    result=review(scene,a.batch_root,json.loads(a.selection.read_text()))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(scene,result,a.output)
    for row in result['regions']:
        print(dict(region=row['region'],accesses=len(row['current_accesses']),proposals=len(row['proposed_accesses']),
            observed_paths=len(row['observed_paths']),turns=[c for c in row['source_connections']if c.get('dir')=='t']),flush=True)


if __name__=='__main__':main()
