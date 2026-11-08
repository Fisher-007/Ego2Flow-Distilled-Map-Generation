"""Replace selected paint strokes while retaining complete installed paint cells."""
import argparse,json
from pathlib import Path
import numpy as np,orjson,shapely
from shapely.ops import substring
from map_generation.source_paint import PATTERNS
from map_generation.shared_road_ground import clip_outside

def source_strokes(row,width):
    line=shapely.LineString(row['centerline_xy_m']);pieces=[]
    for stripe in row['paint_intervals_m']:
        for low,high in stripe['intervals_m']:
            part=substring(line,low,high)
            if part.length<=1e-8:continue
            if stripe['lateral_offset_m']:
                part=shapely.offset_curve(part,stripe['lateral_offset_m'],join_style='mitre')
            pieces.append(shapely.buffer(part,width/2,cap_style='flat',join_style='mitre'))
    return shapely.union_all(pieces)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('previous-paint','replacement-paint','conditions','tile-report','native-report','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--current-paint',type=Path,help='Complete currently installed cells; preserve their existing retained geometry instead of reconstructing old cells')
    a=p.parse_args();read=lambda p:orjson.loads(p.read_bytes())
    old=read(a.previous_paint);new=read(a.replacement_paint);conditions=read(a.conditions);tiles=read(a.tile_report);native=read(a.native_report)
    keys={(r['lane_id'],r['side'])for r in conditions['source_side_selection']}
    explicitly_retired=set(conditions.get('retire_boundary_ids', []))
    retired_sides={(r['lane_id'],r['side'])for r in conditions.get('retire_source_sides',[])}
    current={m['id']:m for m in read(a.current_paint)['meshes']}if a.current_paint else None
    target={};others={};chosen=[]
    width=old['report']['parameters']['width_m']
    for row in old['report']['boundary_records']:
        if 'id'not in row:continue
        color=PATTERNS[row['mark_type']][0]
        selected=row['id'] in explicitly_retired or any((r['lane_id'],r['side'])in keys|retired_sides for r in row['sources'])
        store=target if selected else others
        store.setdefault(color,[]).append(source_strokes(row,width))
        if selected:chosen.append(row['id'])
    domains={}
    for color,parts in target.items():
        # Retained nonselected paint can cross the selected stroke. Its original
        # footprint stays even when the selected pattern leaves a new gap there.
        domain=shapely.union_all(parts)
        other=shapely.union_all(others.get(color,[]))
        domains[color]=shapely.difference(domain,other)
    a.output.mkdir(parents=True,exist_ok=True);meshes=[];selection=[];retired=[]
    tile_by_name={r['name']:r for r in tiles['objects']}
    for before in old['meshes']:
        color=before['color'];domain=domains.get(color)
        after=next((m for m in new['meshes']if m['color']==color),None)
        if domain is None or after is None:continue
        # Reproduce the actual installed Blender float32 whole-triangle cells.
        v=np.asarray(before['vertices'],np.float32);f=np.asarray(before['faces'],int);tri=v[f]
        cells=np.floor(tri[:,:,:2].mean(1)/tiles['tile_size_m']).astype(int)
        unique,group=np.unique(cells,axis=0,return_inverse=True)
        order=np.argsort(group,kind='stable');cuts=np.r_[0,np.flatnonzero(np.diff(group[order]))+1,len(order)]
        parts=[order[lo:min(lo+250000,high)]for low,high in zip(cuts[:-1],cuts[1:])for lo in range(low,high,250000)]
        revised=np.asarray(after['vertices'])[np.asarray(after['faces'],int)]
        revised_cells=np.floor(revised[:,:,:2].mean(1)/tiles['tile_size_m']).astype(int)
        assigned=np.zeros(len(revised),bool)
        for index,ids in enumerate(parts):
            name=before['id']+'_tile_'+str(index);record=tile_by_name[name]
            if record['triangles']!=len(ids):raise ValueError('Installed paint partition mismatch: '+name)
            cell=tuple(cells[ids[0]]);actual=tri[ids].astype(float)
            if current is not None:
                if name not in current:continue
                existing=current[name]
                actual=np.asarray(existing['vertices'])[np.asarray(existing['faces'],int)]
            bounds=shapely.box(*actual[:,:,:2].min((0,1)),*actual[:,:,:2].max((0,1)))
            additions=(revised_cells==cell).all(1)
            if not bounds.intersects(domain) and not additions.any():continue
            if assigned[additions].any():raise ValueError('Multiple original chunks in replacement cell require explicit repartition')
            assigned|=additions
            intersects=shapely.intersects(shapely.polygons(actual[:,:,:2]),domain)
            # Normalize winding on selected faces before clipping.
            affected=actual[intersects]
            if len(affected):
                piece=clip_outside(dict(vertices=affected.reshape(-1,3).tolist(),faces=np.arange(affected.size//3).reshape(-1,3).tolist()),domain)
                clipped=np.asarray(piece['vertices']).reshape(-1,3,3)
                retained=np.concatenate([actual[~intersects],clipped])
            else:
                retained=actual
            combined=np.concatenate([retained,revised[additions]])
            meshes.append(dict(after,id=name,vertices=combined.reshape(-1,3).tolist(),faces=np.arange(combined.size//3).reshape(-1,3).tolist()))
            matches=[r for r in native['paint']if r['mesh'].split('/')[-1].split('.')[0]=='markings_'+name]
            if len(matches)!=1:raise ValueError('Expected one recorded native tile: '+name)
            retired.extend(matches)
            selection.append(dict(id=name,cell_xy=[int(v) for v in cell],old_triangles=len(ids),current_triangles=len(actual),retained_triangles=len(retained),new_pattern_triangles=int(additions.sum()),complete_tile=True))
        if not assigned.all():raise ValueError('Replacement paint extends into a cell without installed paint; declare this new cell explicitly')
    report=dict(previous_paint=str(a.previous_paint),current_paint=str(a.current_paint)if a.current_paint else None,retired_source_sides=conditions.get('retire_source_sides',[]),replacement_paint=str(a.replacement_paint),conditions=str(a.conditions),selected_old_boundary_ids=chosen,
        explicitly_retired_boundary_ids=sorted(explicitly_retired),
        selected_cells=selection,retired_native_meshes=[r['mesh']for r in retired],all_unselected_native_assets_retained=True,
        method='Remove only selected old paint footprints excluding any unchanged crossing paint; clip retained triangles on original planes; replace complete affected native cells with retained plus revised paint.',
        scope='Paint-only topology and display change. Road source geometry, navigation, collision, unrelated paint and other native asset cells retained. Native image/reopen validation remains required.')
    (a.output/'markings.json').write_bytes(orjson.dumps(dict(frame=new.get('frame','SceneIR ENU metres'),meshes=meshes,report=report)))
    (a.output/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'previous_native_subset.json').write_text(json.dumps(dict(paint=retired),indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
