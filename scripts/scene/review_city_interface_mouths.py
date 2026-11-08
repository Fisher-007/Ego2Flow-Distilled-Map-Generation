"""Measure and draw observed-source interfaces for semantic road review."""
import argparse,json
from pathlib import Path
import numpy as np
from shapely.geometry import LineString,Polygon
from shapely.ops import nearest_points
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def measure(lanes,component):
    rows=[]
    for v in component['findings']:
        a,b=lanes[v['source']],lanes[v['target']]
        mouths=[np.asarray([a['left_boundary'][-1],a['right_boundary'][-1]]),
                np.asarray([b['left_boundary'][0],b['right_boundary'][0]])]
        ca,cb=[LineString(p[:,:2])for p in mouths];pa,pb=nearest_points(ca,cb)
        za=np.interp(ca.project(pa)/max(ca.length,1e-9),[0,1],mouths[0][:,2])
        zb=np.interp(cb.project(pb)/max(cb.length,1e-9),[0,1],mouths[1][:,2])
        ac=np.asarray(a['centerline']);bc=np.asarray(b['centerline'])
        da=ac[-1,:2]-ac[max(0,len(ac)-3),:2];db=bc[min(len(bc)-1,2),:2]-bc[0,:2]
        da/=max(np.linalg.norm(da),1e-9);db/=max(np.linalg.norm(db),1e-9)
        delta=bc[0,:2]-ac[-1,:2]
        rows.append(dict(v,mouth_gap_xy_m=ca.distance(cb),mouth_nearest_z_m=abs(float(za-zb)),
            boundary_widths_m=[ca.length,cb.length],signed_forward_gap_m=float(delta@da),
            lateral_offset_m=float(abs(da[0]*delta[1]-da[1]*delta[0])),heading_cosine=float(da@db)))
    return dict(component_id=component['id'],ports=component['ports'],center_local_xyz=component['center_local_xyz'],joins=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['scene','handoffs','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--page-size',type=int,default=12);a=p.parse_args()
    s=json.loads(a.scene.read_text());d=json.loads(a.handoffs.read_text());lanes={x['id']:x for x in s['fusion_map']['lanes']}
    rows=[measure(lanes,c)for c in d['conflict_components']];a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'mouth_measurements.json').write_text(json.dumps(dict(source_scene=str(a.scene),source_handoffs=str(a.handoffs),components=rows,scope='All conflicting observed successor mouths; signed overlap, heading and boundary gaps are measurements, not automatic permission to alter observed geometry.'),indent=2)+'\n')
    for page in range(0,len(rows),a.page_size):
        fig,axs=plt.subplots(4,3,figsize=(18,20))
        for ax,row in zip(axs.flat,rows[page:page+a.page_size]):
            center=np.asarray(row['center_local_xyz'])[:2];keys=sorted({p['lane_id']for p in row['ports']})
            for n,k in enumerate(keys):
                lane=lanes[k];col=plt.get_cmap('tab10')(n%10);q=np.asarray(lane['centerline'])
                ax.plot(q[:,0],q[:,1],'.-',color=col,ms=2,lw=1.2)
                if len(q)>1:ax.annotate('',xy=q[-1,:2],xytext=q[-2,:2],arrowprops=dict(arrowstyle='->',color=col))
                for f in ['left_boundary','right_boundary']:
                    p=np.asarray(lane[f]);ax.plot(p[:,0],p[:,1],color=col,lw=.7,alpha=.65)
                ends=[p['end']for p in row['ports']if p['lane_id']==k]
                for end in ends:
                    pt=q[0 if end==0 else -1,:2];ax.plot(*pt,'o',color=col,ms=4)
                    ax.text(pt[0],pt[1],k.split(':')[1][-5:]+('S'if end==0 else 'E'),fontsize=6,color=col)
            for j in row['joins']:
                q=np.asarray([j['source_point'],j['target_point']]);ax.plot(q[:,0],q[:,1],'k--',lw=.7)
            worst=max(row['joins'],key=lambda x:max(x['mouth_gap_xy_m']/.3,x['mouth_nearest_z_m']/.15))
            ax.set_title(row['component_id']+'  joins '+str(len(row['joins']))+'\nmouth XY %.2f Z %.2f; forward %.2f; cos %.2f'%(worst['mouth_gap_xy_m'],worst['mouth_nearest_z_m'],worst['signed_forward_gap_m'],worst['heading_cosine']),fontsize=9)
            ax.set_xlim(center[0]-12,center[0]+12);ax.set_ylim(center[1]-12,center[1]+12);ax.set_aspect('equal');ax.grid(alpha=.3)
        for ax in list(axs.flat)[len(rows[page:page+a.page_size]):]:ax.axis('off')
        fig.tight_layout();fig.savefig(a.output/f'ports_{page//a.page_size:02d}.png',dpi=130);plt.close(fig)
    flat=[j for c in rows for j in c['joins']]
    print(json.dumps(dict(components=len(rows),joins=len(flat),pages=(len(rows)+a.page_size-1)//a.page_size,
        gap_components=sum(any(j['mouth_gap_xy_m']>.3 for j in c['joins'])for c in rows),
        height_components=sum(any(j['mouth_nearest_z_m']>.15 for j in c['joins'])for c in rows))),flush=True)
if __name__=='__main__':main()
