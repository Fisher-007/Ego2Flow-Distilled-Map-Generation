"""Measure long terrain edges and expose their nearby source-road constraints."""
import argparse,json
from pathlib import Path
import numpy as np
import shapely
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('terrain','scene','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--minimum-length-m',type=float,default=.25)
    p.add_argument('--report-grade',type=float,default=.3)
    a=p.parse_args();terrain=json.loads(a.terrain.read_text());scene=json.loads(a.scene.read_text())
    v=np.asarray(terrain['vertices']);f=np.asarray(terrain['faces'])
    e=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
    distance=np.linalg.norm(v[e[:,0],:2]-v[e[:,1],:2],axis=1)
    e=e[distance>a.minimum_length_m];distance=distance[distance>a.minimum_length_m]
    grade=np.abs(v[e[:,0],2]-v[e[:,1],2])/distance
    lanes=scene['fusion_map']['lanes'];lines=[shapely.LineString(x['centerline']) for x in lanes]
    tree=shapely.STRtree(lines)
    order=np.argsort(-grade);rows=[]
    for i in order[:200]:
        xyz=v[e[i]];mid=xyz[:,:2].mean(0);near=tree.query(shapely.Point(mid).buffer(8))
        roads=[]
        for j in near:
            lane=lanes[j];roads.append(dict(id=lane['id'],distance_m=lines[j].distance(shapely.Point(mid)),
                centerline=lane['centerline'],width_m=lane['width_m'],source_tags=lane.get('source_tags',{}),
                source=lane.get('source'),internal=lane.get('internal'),is_intersection=lane.get('is_intersection')))
        rows.append(dict(grade=float(grade[i]),length_m=float(distance[i]),xyz=xyz.tolist(),roads=roads))
    centers=[]
    for row in rows:
        center=np.asarray(row['xyz'])[:,:2].mean(0)
        if all(np.linalg.norm(center-c)>20 for c in centers):centers.append(center)
        if len(centers)==6:break
    fig,axs=plt.subplots(2,3,figsize=(18,11))
    for ax,center in zip(axs.ravel(),centers):
        nearby=np.linalg.norm(v[:,:2]-center,axis=1)<18
        ax.scatter(v[nearby,0],v[nearby,1],c=v[nearby,2],s=5,cmap='terrain')
        for j in tree.query(shapely.Point(center).buffer(18)):
            xyz=np.asarray(lanes[j]['centerline']);ax.plot(xyz[:,0],xyz[:,1],lw=1,alpha=.55)
        for row in rows:
            xyz=np.asarray(row['xyz'])
            if np.linalg.norm(xyz[:,:2].mean(0)-center)<18:
                ax.plot(xyz[:,0],xyz[:,1],color='red',lw=2)
        ax.set_xlim(center[0]-18,center[0]+18);ax.set_ylim(center[1]-18,center[1]+18)
        ax.set_aspect('equal');ax.set_title('Terrain edge region '+str(np.round(center,1)))
    a.output.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(a.output/'edges.png',dpi=160);plt.close(fig)
    report=dict(terrain=str(a.terrain),scene=str(a.scene),minimum_length_m=a.minimum_length_m,
        long_edge_count=len(e),grade_quantiles=np.quantile(grade,[.5,.95,.99,1]).tolist(),
        over_report_grade=int((grade>a.report_grade).sum()),report_grade=a.report_grade,
        worst_edges=rows,scope='Diagnostic geometry, not a universal terrain grade acceptance threshold.')
    (a.output/'review.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:x for k,x in report.items() if k!='worst_edges'}))
    print('WORST',json.dumps([{k:x for k,x in row.items() if k!='roads'} for row in rows[:10]]),flush=True)


if __name__=='__main__':main()
