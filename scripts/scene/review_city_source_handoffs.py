"""Measure declared Area reachability and remaining observed-source port conflicts."""
import argparse,json
from pathlib import Path
import numpy as np
import networkx as nx
from shapely.geometry import LineString,box
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def review(scene,xy_limit,z_limit):
    lanes={l['id']:l for l in scene['fusion_map']['lanes']}
    lengths={k:LineString(np.asarray(l['centerline'])[:,:2]).length for k,l in lanes.items()}
    passenger={k for k,l in lanes.items()if 'passenger'in l.get('allowed_classes',[])}
    g=nx.DiGraph();g.add_nodes_from(passenger)
    for k in passenger:
        for t in lanes[k]['successors']:
            if t in passenger:g.add_edge(k,t,weight=lengths[k])
    area_lanes={}
    for a in scene['areas']:
        bounds=box(*a['bbox_local_xy'])
        area_lanes[a['id']]=sorted(k for k in passenger if LineString(np.asarray(lanes[k]['centerline'])[:,:2]).intersects(bounds))
    witnesses=[]
    for source,ids in area_lanes.items():
        distances,paths=nx.multi_source_dijkstra(g,ids,weight='weight') if ids else ({},{})
        for target,targets in area_lanes.items():
            if source==target:continue
            reachable=[k for k in targets if k in distances]
            end=min(reachable,key=lambda k:distances[k])if reachable else None
            witnesses.append(dict(source_area=source,target_area=target,reachable=end is not None,path=paths[end]if end else [],source_lane_length_m=float(distances[end])if end else None))
    components=list(nx.strongly_connected_components(g));total=sum(lengths[k]for k in passenger)
    connectivity=dict(area_count=len(area_lanes),area_candidate_counts={k:len(v)for k,v in area_lanes.items()},passenger_lanes=len(passenger),strong_components=len(components),largest_strong_component_length_fraction=max((sum(lengths[k]for k in c)for c in components),default=0)/max(total,1e-8),ordered_area_pairs=len(witnesses),reachable_pairs=sum(v['reachable']for v in witnesses),witnesses=witnesses,scope='Existence of a permitted directed passenger-lane path between each Area pair, not all-lane reachability or native driving validation.')
    ports=nx.Graph();joins=[]
    for k,lane in lanes.items():
        if lane.get('source')!='av2':continue
        for target in lane['successors']:
            if target not in lanes or lanes[target].get('source')!='av2':continue
            p=np.asarray(lane['centerline'])[-1];q=np.asarray(lanes[target]['centerline'])[0];delta=p-q
            row=dict(source=k,target=target,source_point=p.tolist(),target_point=q.tolist(),xy_m=float(np.linalg.norm(delta[:2])),z_m=float(abs(delta[2])))
            joins.append(row);ports.add_edge((k,1),(target,0))
    conflict_components=[]
    for component in nx.connected_components(ports):
        evidence=[v for v in joins if (v['source'],1)in component and(v['xy_m']>xy_limit or v['z_m']>z_limit)]
        if not evidence:continue
        points=[np.asarray(lanes[k]['centerline'])[0 if e==0 else -1]for k,e in component]
        conflict_components.append(dict(id='source_port_conflict_'+str(len(conflict_components)).zfill(3),ports=[dict(lane_id=k,end=e)for k,e in sorted(component)],center_local_xyz=np.mean(points,axis=0).tolist(),findings=evidence,maximum_xy_m=max(v['xy_m']for v in evidence),maximum_z_m=max(v['z_m']for v in evidence)))
    source=dict(joins=joins,thresholds=dict(xy_m=xy_limit,z_m=z_limit),conflict_components=conflict_components,xy_p95_m=float(np.percentile([v['xy_m']for v in joins],95)),xy_max_m=max(v['xy_m']for v in joins),z_p95_m=float(np.percentile([v['z_m']for v in joins],95)),z_max_m=max(v['z_m']for v in joins),scope='Measurements only. Agent must distinguish source-version conflicts, lane-width transitions and physical branching before deciding construction conditions.')
    return connectivity,source

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--xy-limit',type=float,default=.3);p.add_argument('--z-limit',type=float,default=.15)
    a=p.parse_args();s=json.loads(a.scene.read_text());c,v=review(s,a.xy_limit,a.z_limit);a.output.mkdir(parents=True,exist_ok=True)
    for name,d in [('area_connectivity.json',c),('source_handoffs.json',v)]:
        d['source_scene']=str(a.scene.resolve());(a.output/name).write_text(json.dumps(d,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(16,12))
    for source,color,width in [('other','#bbbbbb',.25),('av2','#2465a3',.5)]:
        shapes=[np.asarray(l['centerline'])[:,:2]for l in s['fusion_map']['lanes']if(l.get('source')=='av2')==(source=='av2')]
        ax.add_collection(LineCollection(shapes,colors=color,linewidths=width))
    for area in s['areas']:
        x0,y0,x1,y1=area['bbox_local_xy'];ax.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],color='green',lw=.7);ax.text((x0+x1)/2,(y0+y1)/2,area['id'],fontsize=7,color='green')
    for row in v['conflict_components']:
        x,y,z=row['center_local_xyz'];ax.plot(x,y,'ro',ms=3);ax.text(x,y,row['id'].split('_')[-1],fontsize=6,color='red')
    ax.autoscale();ax.set_aspect('equal');ax.set_title('Complete declared city: observed roads, 20 Areas and measured source port conflicts');fig.tight_layout();fig.savefig(a.output/'source_handoffs.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(connectivity={k:x for k,x in c.items()if k not in ['witnesses','area_candidate_counts']},source={k:x for k,x in v.items()if k not in ['joins','conflict_components']},source_conflict_components=len(v['conflict_components'])),indent=2),flush=True)
if __name__=='__main__':main()
