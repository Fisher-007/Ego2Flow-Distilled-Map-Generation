"""Inspect actual ground slopes and compare retained diagnostic positions."""
from pathlib import Path
import orjson,json,numpy as np,shapely
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import argparse
p=argparse.ArgumentParser(description='Locate and plot actual steep terrain edges beside source roads')
for key in ('terrain','scene','output'):p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--reference-review',type=Path)
a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
terrain=orjson.loads(a.terrain.read_bytes());v=np.asarray(terrain['vertices']);f=np.asarray(terrain['faces']);del terrain
edges=np.unique(np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
delta=v[edges[:,1]]-v[edges[:,0]];length=np.linalg.norm(delta[:,:2],axis=1);grade=np.abs(delta[:,2])/np.maximum(length,1e-9)
eligible=length>.25
scene=orjson.loads(a.scene.read_bytes());lanes=scene['fusion_map']['lanes'];lines=np.array([shapely.LineString(l['centerline'])for l in lanes]);tree=shapely.STRtree(lines)
centers=(v[edges[:,0],:2]+v[edges[:,1],:2])/2;selected=[]
for i in np.flatnonzero(eligible)[np.argsort(-grade[eligible])]:
 if grade[i]<.5:break
 if all(np.linalg.norm(centers[i]-centers[j])>15 for j in selected):selected.append(int(i))
 if len(selected)==12:break
rows=[]
for i in selected:
 p=shapely.Point(centers[i]);ids=tree.query(p,predicate='dwithin',distance=15)
 near=sorted(ids,key=lambda k:lines[k].distance(p))[:6]
 rows.append(dict(edge_id=i,xyz=v[edges[i]].tolist(),length_m=float(length[i]),grade=float(grade[i]),nearby_lanes=[dict(id=lanes[k]['id'],distance_m=float(lines[k].distance(p)),centerline=lanes[k]['centerline'],source_tags=lanes[k].get('source_tags'),source=lanes[k]['source'])for k in near]))
summary=dict(edge_count=len(edges),length_over_quarter_m_count=int(eligible.sum()),grade_quantiles=np.quantile(grade[eligible],[.5,.95,.99,.999,1]).tolist(),edges_over_half_grade=int((eligible&(grade>.5)).sum()),samples=rows)
if a.reference_review:
 from scipy.spatial import cKDTree
 previous=json.loads(a.reference_review.read_text());vtree=cKDTree(v[:,:2]);comparisons=[]
 for row in previous['samples']:
  q=np.asarray(row['xyz']);dist,ix=vtree.query(q[:,:2]);current=v[ix]
  comparisons.append(dict(previous_xyz=row['xyz'],current_xyz=current.tolist(),xy_match_max_m=float(max(dist)),previous_grade=row['grade'],current_grade=float(abs(current[1,2]-current[0,2])/row['length_m'])))
 summary['reference_comparison']=comparisons
(out/'grade_review.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axes=plt.subplots(3,4,figsize=(20,15))
centroid=v[f,:2].mean(1)
for ax,row in zip(axes.flat,rows):
 xy=np.mean(row['xyz'],axis=0)[:2];mask=np.linalg.norm(centroid-xy,axis=1)<18;tri=v[f[mask]]
 collection=PolyCollection(tri[:,:,:2],array=tri[:,:,2].mean(1),cmap='terrain',edgecolors='none');ax.add_collection(collection)
 for lane in row['nearby_lanes']:
  c=np.asarray(lane['centerline']);ax.plot(c[:,0],c[:,1],lw=1,label=lane['id'][-24:])
 q=np.asarray(row['xyz']);ax.plot(q[:,0],q[:,1],'r.-',lw=3);ax.set_xlim(xy[0]-15,xy[0]+15);ax.set_ylim(xy[1]-15,xy[1]+15);ax.set_aspect('equal');ax.set_title('grade %.2f dz %.2fm'%(row['grade'],abs(q[1,2]-q[0,2])));ax.legend(fontsize=5);fig.colorbar(collection,ax=ax)
fig.tight_layout();fig.savefig(out/'steep_edges.png',dpi=130)
print({k:v for k,v in summary.items()if k!='samples'},flush=True)
print('SAMPLES',[{k:v for k,v in row.items()if k!='nearby_lanes'}for row in rows],flush=True)


