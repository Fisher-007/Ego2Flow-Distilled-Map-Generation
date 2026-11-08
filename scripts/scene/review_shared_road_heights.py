"""Measure and display local shared road-floor height feedback."""
import argparse,json
from pathlib import Path
import numpy as np
import shapely
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from map_generation.av2_surfaces import mesh_triangles


def stats(tri):
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    area=abs(normal[:,2])/2;valid=area>1e-10
    slope=np.linalg.norm(normal[valid,:2],axis=1)/abs(normal[valid,2])
    area=area[valid]
    return dict(triangles=len(tri),area_m2=float(area.sum()),height_range_m=[float(tri[:,:,2].min()),float(tri[:,:,2].max())],
        slope_max=float(slope.max()),area_grade_over_15_percent_m2=float(area[slope>.15].sum()),
        area_grade_over_30_percent_m2=float(area[slope>.3].sum()),
        area_grade_over_100_percent_m2=float(area[slope>1].sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['previous-roads','roads','review','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();before=json.loads(a.previous_roads.read_text());after=json.loads(a.roads.read_text());review=json.loads(a.review.read_text())
    old={m['id']:m for m in before['meshes']};new={m['id']:m for m in after['meshes']}
    fig,axes=plt.subplots(len(review['patches']),2,figsize=(12,5*len(review['patches'])),squeeze=False);rows=[]
    for i,r in enumerate(review['patches']):
        selected=r['selection'];current=new[r['id']]
        previous_ids=selected.get('feedback_source_patches')
        if previous_ids is None:previous_ids=[selected['feedback_source_patch']]
        previous_meshes=[old[key]for key in previous_ids]
        previous=dict(id=' + '.join(previous_ids))
        t0=np.concatenate([mesh_triangles(m)for m in previous_meshes]);t1=mesh_triangles(current)
        if len(previous_meshes)==1:
            prior_selection=previous_meshes[0]['selection']
            poly=shapely.Polygon(prior_selection['polygon_xy_m'],prior_selection.get('holes_xy_m',[]))
        else:
            poly=shapely.union_all(shapely.polygons(t0[:,:,:2]))
        keep=shapely.intersects(shapely.points(t1[:,:,:2].mean(1)),poly)
        row=dict(previous_id=previous['id'],current_id=current['id'],previous_scope=stats(t0),new_same_scope=stats(t1[keep]),
            new_expanded_scope=stats(t1),source_xy_area_error_m2=r['source_xy_coverage_area_error_m2'],
            observed_contacts=r['observed_contact_vertices'],retained_isolated_support=r['retained_isolated_support'])
        rows.append(row)
        low=min(t0[:,:,2].min(),t1[:,:,2].min());high=max(t0[:,:,2].max(),t1[:,:,2].max())
        for ax,t,title in zip(axes[i],[t0,t1],['Previous '+previous['id'],'Local FEM '+current['id']]):
            col=PolyCollection(t[:,:,:2],array=t[:,:,2].mean(1),cmap='terrain',clim=(low,high),edgecolors='none')
            ax.add_collection(col);ax.autoscale();ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('East (m)');ax.set_ylabel('North (m)')
            fig.colorbar(col,ax=ax,label='Physical height (m)')
    a.output.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(a.output/'local_height_feedback.png',dpi=140);plt.close(fig)
    report=dict(patches=rows,observed_AV2_meshes_exactly_retained=review['observed_AV2_meshes_exactly_retained'],scope='Physical gradient statistics; source geometry, native driving and native contact acceptance are separate.')
    (a.output/'review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
