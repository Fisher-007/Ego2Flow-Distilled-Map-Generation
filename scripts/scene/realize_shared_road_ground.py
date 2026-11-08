"""Realize graph-selected same-level physical road support patches."""
import argparse
import json
from pathlib import Path

import numpy as np
import shapely

from map_generation.shared_road_ground import realize_shared_ground, reconstructed_support_selected
from map_generation.av2_surfaces import mesh_triangles
from map_generation.junction_surfaces import plane_z
from scripts.scene.build_assets import surface_heights, BOOLEAN_GRID_M
from scripts.scene.rebase_physical_road_heights import source_ids as physical_source_ids
from scripts.scene.prepare_city_ground import at_grade_roads



def reviewed_approach_extent_review(before, after, scopes):
    """Measure unchanged physical planes outside explicitly selected portals."""
    measurements=[]
    for key, polygons in scopes.items():
        old=mesh_triangles(before[key])
        current=mesh_triangles(after[key]) if key in after else np.empty((0,3,3))
        domain=shapely.union_all(polygons)
        original=shapely.union_all(shapely.polygons(old[:,:,:2]))
        retained=shapely.union_all(shapely.polygons(current[:,:,:2]))
        expected=original.difference(domain)
        area_error=expected.symmetric_difference(retained).area
        error=0.
        if len(current):
            centers=current.mean(1)
            tree=shapely.STRtree(shapely.polygons(old[:,:,:2]))
            # Match retained triangles to source planes at all three vertices.
            pairs=tree.query(shapely.points(centers[:,:2]),predicate='intersects')
            plane_errors=np.zeros(pairs.shape[1])
            for vertex in range(3):
                expected_z=plane_z(old[pairs[1]],current[pairs[0],vertex,:2])
                plane_errors=np.maximum(plane_errors,abs(expected_z-current[pairs[0],vertex,2]))
            per_triangle=np.full(len(current),np.inf)
            np.minimum.at(per_triangle,pairs[0],plane_errors)
            missing=~np.isfinite(per_triangle)
            if missing.any():
                height=surface_heights(old,tree,centers[missing,:2],'retained portal exterior')
                per_triangle[missing]=abs(height-centers[missing,2])
            error=float(per_triangle.max())
        tolerance=2*BOOLEAN_GRID_M*max(original.length,1.)
        if area_error>tolerance or error>2*BOOLEAN_GRID_M:
            raise ValueError('Reviewed portal changed physical geometry outside its selected extent: '+key)
        measurements.append(dict(mesh_id=key,exterior_area_difference_m2=area_error,
            exterior_plane_height_difference_m=error,boolean_area_tolerance_m2=tolerance,
            height_assignment='Actual covering source triangle plane, compared at all retained vertices; nearest boundary fallback only when no covering triangle exists'))
    return measurements


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('scene','roads','conditions','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--workers',type=int,default=1,help='Concurrent solves for already merged, disjoint source junction regions')
    a=p.parse_args();read=lambda path:json.loads(path.read_text())
    scene,roads,conditions=read(a.scene),read(a.roads),read(a.conditions)
    selected=next(row for row in conditions['native_objects'] if row['choice']=='enhance'
        and row.get('native_construction',{}).get('kind')=='shared_inferred_road_ground')
    settings=selected['native_construction']['settings']
    buffer_m=float(settings.get('separate_level_buffer_m',4.))
    spacing_m=float(settings.get('spacing_m',settings.get('grid_spacing_m',.5)))
    _,protected,separate,excluded=at_grade_roads(scene,roads,buffer_m)
    by_id={m['id']:m for m in roads['meshes']}
    contacts=[];approach_scopes={};reconstructed_scopes={}
    for region in settings['regions']:
        for key in region.get('reconstructed_support_mesh_ids',[]):
            reconstructed_support_selected(by_id[key],region)
            reconstructed_scopes.setdefault(key,[]).append(shapely.Polygon(
                region['polygon_xy_m'],region.get('holes_xy_m',[])))
        if not shapely.Polygon(region['polygon_xy_m'],region.get('holes_xy_m',[])).intersects(protected):
            continue
        contact=region.get('separate_level_contact')
        if not contact or not region.get('mesh_ids'):
            raise ValueError('Selected same-level road patch intersects a tagged separate level')
        # Use source ownership for approach selection inside bridge buffers.
        allowed=set(contact['fixed_source_lane_ids'])
        fixed=set(region.get('fixed_support_mesh_ids',[]))
        selected_ids=set(region['mesh_ids'])
        touched=selected_ids & set(excluded)
        # A source-tagged tunnel or bridge can end inside a same-level portal.
        # Only an explicit source-backed approach condition makes that local
        # part mutable; true crossing decks and all other tagged roads stay fixed.
        mutable=set(contact.get('mutable_approach_mesh_ids',[]))
        mutable_sources=set(contact.get('mutable_source_lane_ids',[]))
        if not touched<=fixed|mutable or not mutable<=touched:
            raise ValueError('Selected tagged surface lacks fixed or reviewed portal ownership')
        for key in mutable:
            ids=physical_source_ids(by_id[key])
            if not ids or not ids<=mutable_sources:
                raise ValueError('Mutable portal lacks exact reviewed source ownership: '+key)
            approach_scopes.setdefault(key,[]).append(shapely.Polygon(
                region['polygon_xy_m'],region.get('holes_xy_m',[])))
        for key in touched-mutable:
            mesh=by_id[key]
            source_ids={mesh.get('source_lane_id')}|set(mesh.get('source_lane_ids',[]))
            if mesh.get('kind')=='connection_closure':
                source_ids.update(mesh[side+'_source_lane_id'] if side+'_source_lane_id' in mesh
                    else by_id[mesh[side+'_mesh_id']].get('source_lane_id') for side in ('from','to'))
            if not source_ids.intersection(allowed):
                raise ValueError('Fixed separate-level surface lacks reviewed source ownership: '+key)
        contacts.append(dict(selection=region.get('id'),condition=contact,
            fixed_separate_level_mesh_ids=sorted(touched-mutable),
            reviewed_mutable_approach_mesh_ids=sorted(mutable)))
    result,review=realize_shared_ground(roads,settings['regions'],spacing_m,a.workers)
    before={m['id']:m for m in roads['meshes']
        if m.get('kind') in ('av2_boundary_surface','av2_grounded_junction')
        and m['id'] not in reconstructed_scopes}
    after={m['id']:m for m in result['meshes'] if m['id'] in before}
    if before!=after:raise ValueError('Observed AV2 geometry changed during inferred support realization')
    separate_before={m['id']:m for m in roads['meshes'] if m['id'] in set(excluded)}
    separate_after={m['id']:m for m in result['meshes'] if m['id'] in separate_before}
    unchanged_before={k:v for k,v in separate_before.items()if k not in approach_scopes}
    unchanged_after={k:v for k,v in separate_after.items()if k not in approach_scopes}
    if unchanged_before!=unchanged_after:
        raise ValueError('Unselected tagged physical geometry changed during approach realization')
    approach_review=reviewed_approach_extent_review(
        by_id,{m['id']:m for m in result['meshes']},approach_scopes)
    reconstructed_review=reviewed_approach_extent_review(
        by_id,{m['id']:m for m in result['meshes']},reconstructed_scopes)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'road_meshes.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    report=dict(condition=selected,input_roads=str(a.roads),patches=review,observed_AV2_meshes_exactly_retained=True,
        observed_support_definition='Raw AV2 boundary surfaces and grounded junctions without explicitly selected recorded reconstruction.',
        reviewed_reconstructed_support_exterior=reconstructed_review,
        native_navigation_file_modified=False,tagged_separate_level_sources=separate,
        separate_level_contacts=contacts,
        tagged_separate_level_meshes_exactly_retained=separate_before==separate_after,
        unselected_tagged_meshes_exactly_retained=True,reviewed_approach_exterior=approach_review,
        scope='Inferred physical support under unchanged source/native traffic graph. Native driving remains a separate check.')
    (a.output/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(patches=len(review),area_m2=sum(r['area_m2']for r in review),
        max_height_change_m=max(r['max_height_change_m']for r in review))),flush=True)


if __name__=='__main__':main()
