"""Measure clipped junction ports against retained source roads for review."""
import argparse,json,math
from pathlib import Path
from collections import defaultdict
import numpy as np
from shapely.geometry import LineString,Point
from shapely.strtree import STRtree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def geometry(lane):
    p=np.asarray(lane['centerline'],float)
    return p,LineString(p[:,:2])

def at(p,s):
    d=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1))]
    return np.array([np.interp(s,d,p[:,i])for i in range(3)])

def measure(scene,ports,radius):
    lanes={l['id']:l for l in scene['fusion_map']['lanes']}
    layers={'current':lanes,'coarse':{l['id']:l for l in scene['coarse_map']['lanes']}}
    indices={}
    for name,items in layers.items():
        ids=list(items);g=[geometry(items[k])[1]for k in ids]
        indices[name]=(ids,g,STRtree(g))
    out=[]
    for row in ports:
        lane=lanes[row['id']];p,line=geometry(lane);start=row['side']=='start'
        position=p[0 if start else -1];s=0 if start else line.length
        direction=at(p,min(line.length,s+1))-at(p,max(0,s-1));direction=direction[:2]/np.linalg.norm(direction[:2])
        candidates={}
        for name,items in layers.items():
            ids,lines,tree=indices[name];found=[]
            for n in tree.query(Point(position[:2]).buffer(radius)):
                key=ids[n]
                if name=='current'and key==row['id']:continue
                target=items[key];q,other=geometry(target);t=other.project(Point(position[:2]));hit=at(q,t)
                v=at(q,min(other.length,t+1))-at(q,max(0,t-1));norm=np.linalg.norm(v[:2])
                if norm<1e-7:continue
                angle=math.degrees(math.acos(np.clip(direction@v[:2]/norm,-1.,1.)))
                distance=float(np.linalg.norm(hit[:2]-position[:2]))
                if angle>50 or distance>radius:continue
                progress=t if start else other.length-t
                found.append(dict(id=key,source=target.get('source'),internal=target.get('internal'),
                    station_m=t,length_m=other.length,distance_m=distance,z_difference_m=float(hit[2]-position[2]),
                    heading_difference_deg=angle,available_direction_m=progress,point=hit.tolist(),
                    width_m=target.get('width_m'),allowed_classes=target.get('allowed_classes'),
                    source_lane_id=target.get('source_lane_id'),source_tags=target.get('source_tags')))
            candidates[name]=sorted(found,key=lambda x:(x['distance_m']+.025*x['heading_difference_deg']))[:10]
        out.append(dict(row,position=position.tolist(),candidates=candidates))
    return out

def plot(scene,rows,output):
    lanes={l['id']:l for l in scene['fusion_map']['lanes']}
    coarse={l['id']:l for l in scene['coarse_map']['lanes']}
    geoms={k:geometry(l)for k,l in lanes.items()}
    ids=list(geoms);tree=STRtree([geoms[k][1]for k in ids])
    groups=defaultdict(list)
    for row in rows:groups[row['group']].append(row)
    for group,ports in groups.items():
        positions=np.asarray([p['position']for p in ports]);lo=positions[:,:2].min(0)-23;hi=positions[:,:2].max(0)+23
        fig,ax=plt.subplots(figsize=(12,10));box=Point((lo+hi)/2).buffer(float(np.linalg.norm(hi-lo)/2))
        for n in tree.query(box):
            k=ids[n];p,line=geoms[k]
            if not line.intersects(box):continue
            av2=lanes[k].get('source')=='av2';ax.plot(p[:,0],p[:,1],color='#388db5'if av2 else '#c1c1c1',lw=1.6 if av2 else .65,alpha=.65)
        for index,row in enumerate(ports):
            k=row['id'];p,_=geoms[k];color=plt.get_cmap('tab10')(index%10)
            ax.plot(p[:,0],p[:,1],color=color,lw=2.6)
            pos=np.asarray(row['position']);ax.scatter(*pos[:2],marker='s'if row['side']=='start'else'o',color=color,s=50,zorder=8)
            ax.annotate(str(index)+' '+k+' '+row['side'],pos[:2],xytext=(4,5+index%3*8),textcoords='offset points',fontsize=8,color=color)
            ax.annotate('',xy=p[-1,:2],xytext=p[max(0,len(p)-3),:2],arrowprops=dict(arrowstyle='->',color=color))
            for j,c in enumerate(row['candidates']['current'][:3]):
                q=np.asarray(c['point']);ax.plot([pos[0],q[0]],[pos[1],q[1]],'--',color=color,lw=.6)
                ax.text(q[0],q[1],str(index)+'.'+str(j)+' '+c['id'],fontsize=6,color=color)
            for c in row['candidates']['coarse'][:1]:
                q=np.asarray(coarse[c['id']]['centerline']);ax.plot(q[:,0],q[:,1],':',color=color,lw=1.3)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),title=f'Clipped observed junction group {group}: {len(ports)} missing ports; blue observed, grey retained inferred, dotted original coarse')
        ax.set_aspect('equal');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/f'group_{group:02d}.png',dpi=145);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['scene','ports','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--radius',type=float,default=15)
    a=p.parse_args();s=json.loads(a.scene.read_text());ports=json.loads(a.ports.read_text())
    rows=measure(s,ports,a.radius);a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'candidates.json').write_text(json.dumps(rows,indent=2)+'\n')
    plot(s,rows,a.output)
    for row in rows:
        print(row['group'],row['id'],row['side'],{k:[(c['id'],round(c['station_m'],3),round(c['distance_m'],3),round(c['heading_difference_deg'],1),round(c['z_difference_m'],3))for c in v[:3]]for k,v in row['candidates'].items()},flush=True)
if __name__=='__main__':main()
