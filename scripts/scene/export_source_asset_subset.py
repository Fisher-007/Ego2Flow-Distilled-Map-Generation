"""Export explicitly triangulated source building subsets from a saved asset blend."""
import argparse,json,sys
from pathlib import Path

def triangulate_object(obj):
    import bpy,numpy as np
    before=np.asarray([tuple(v.co) for v in obj.data.vertices])
    original_faces=len(obj.data.polygons)
    bpy.context.view_layer.objects.active=obj
    modifier=obj.modifiers.new('NativeTriangles','TRIANGULATE')
    modifier.quad_method='FIXED';modifier.ngon_method='CLIP'
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    after=np.asarray([tuple(v.co) for v in obj.data.vertices])
    if len(before)!=len(after) or not np.array_equal(before,after):
        raise ValueError('Native triangulation changed source vertices: '+obj.name)
    if any(len(f.vertices)!=3 for f in obj.data.polygons):
        raise ValueError('Untriangulated source face: '+obj.name)
    ids=np.asarray([tuple(f.vertices) for f in obj.data.polygons],dtype=int)
    xyz=after[ids]
    areas=np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1)*.5
    return dict(mesh=obj.name,vertices=len(after),source_faces=original_faces,
        native_triangles=len(ids),zero_area_triangles=int(np.sum(areas==0)),coordinates_unchanged=True)

def main():
    import bpy
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','selection','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    spec=json.loads(a.assets.read_text());selection=json.loads(a.selection.read_text())
    selected={r['source_id'] for r in selection['missing_source_masses']}
    rows=[r for r in spec['objects'] if r['kind']=='building' and r['source_id'] in selected]
    if len(rows)!=len(selected):raise ValueError('Missing source building record')
    bpy.ops.wm.open_mainfile(filepath=str(Path(spec['blend']).resolve()))
    bpy.ops.object.select_all(action='DESELECT');results=[]
    for row in rows:
        obj=bpy.data.objects[row['name']];obj.select_set(True)
        results.append(dict(source_id=row['source_id'],**triangulate_object(obj)))
    a.output.mkdir(parents=True,exist_ok=True)
    fbx=a.output/'map_assets.fbx'
    bpy.ops.export_scene.fbx(filepath=str(fbx.resolve()),use_selection=True,
        object_types={'MESH'},**spec['fbx_settings'])
    (a.output/'subset.json').write_text(json.dumps(dict(source_assets=str(a.assets.resolve()),
        selection=str(a.selection.resolve()),fbx=str(fbx.resolve()),objects=rows,triangulation=results,
        scope='Source building vertices/transforms/materials retained; only face representation triangulated.'),indent=2)+'\n')
    print(dict(buildings=len(rows),coordinates_unchanged=all(x['coordinates_unchanged']for x in results)))

if __name__=='__main__':main()
