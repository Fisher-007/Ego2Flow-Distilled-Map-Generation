"""Export measured round signal faces with explicit native-state glyph geometry."""
import argparse,json,math,sys
from pathlib import Path

def build(condition, output):
    import bpy
    from mathutils import Vector
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system='METRIC'
    bpy.context.scene.unit_settings.scale_length=1.
    q=json.loads(Path(condition).read_text())
    g=q['geometry'];centers=g['lens_centers_local_m'];radii=g['lens_radius_m']
    front=Vector(g['front_normal_local']);front.normalize()
    tangent=Vector(g['viewer_left_local']);tangent.normalize()
    up=Vector((0,0,1))
    materials=[]
    for name,color in [('SignalBlack',(.008,.009,.011,1)),('SignalLamp',(.8,.3,.03,1)),('SignalHousing',(.58,.37,.035,1))]:
        m=bpy.data.materials.new(name);m.diffuse_color=color;materials.append(m)
    records=[]
    def object_mesh(name, vertices, faces, indices):
        mesh=bpy.data.meshes.new(name);mesh.from_pydata(vertices,[],faces);mesh.update()
        obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj)
        for m in materials:mesh.materials.append(m)
        for p,k in zip(mesh.polygons,indices):p.material_index=k
        # Recalculate selected closed-shell and face normals consistently.
        bpy.context.view_layer.objects.active=obj;obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False);bpy.ops.object.mode_set(mode='OBJECT');obj.select_set(False)
        # Disconnected planar lamp faces have a source-selected viewing side.
        if name.startswith('SignalFace_'):
            for polygon in mesh.polygons:
                if polygon.normal.dot(front)<0:polygon.flip()
            mesh.update()

        low=[min(v[i]for v in vertices)for i in range(3)]
        high=[max(v[i]for v in vertices)for i in range(3)]
        records.append(dict(name=name,local_bounds_m=[low,high],vertices=len(vertices),faces=len(faces)))
        return obj
    def point(c,u,v,d):return list(Vector(c)+tangent*u+up*v+front*d)
    def disc(c,r,vertices,faces,indices,depth=0.,material_index=0):
        n=len(vertices);vertices.append(point(c,0,0,depth))
        vertices.extend(point(c,r*math.cos(a*math.tau/64),r*math.sin(a*math.tau/64),depth)for a in range(64))
        for i in range(64):faces.append([n,n+1+i,n+1+(i+1)%64]);indices.append(material_index)
    def ribbon(c,r,a,b,vertices,faces,indices):
        dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy)
        w=g['arrow_stroke_radius_fraction']*.5
        normal=(-dy/length*w,dx/length*w)
        corners=[(a[0]+normal[0],a[1]+normal[1]),(b[0]+normal[0],b[1]+normal[1]),
                 (b[0]-normal[0],b[1]-normal[1]),(a[0]-normal[0],a[1]-normal[1])]
        n=len(vertices);vertices.extend(point(c,x*r,y*r,.001)for x,y in corners)
        faces.append(list(range(n,n+4)));indices.append(1)
    active=set()
    for state,position in q['native_state_positions'].items():
        i=int(position)-1;active.add(i);vertices=[];faces=[];indices=[]
        disc(centers[i],radii[i],vertices,faces,indices)
        glyph=q.get('native_state_glyphs',{}).get(state,'arrow')
        if glyph=='round':
            disc(centers[i],radii[i]*.94,vertices,faces,indices,depth=.001,material_index=1)
        elif glyph=='arrow':
            for a,b in g['arrow_segments_viewer_left_up']:
                ribbon(centers[i],radii[i],a,b,vertices,faces,indices)
        else:
            raise ValueError('Unknown explicit native signal glyph: '+str(glyph))
        object_mesh('SignalFace_'+state,vertices,faces,indices)
    vertices=[];faces=[];indices=[]
    for i,(c,r)in enumerate(zip(centers,radii)):
        if i not in active:disc(c,r,vertices,faces,indices)
        # Source-visible shallow upper hood; exact depth is explicitly inferred.
        n=len(vertices);count=32
        for d in (0.,g['hood_depth_m']):
            vertices.extend(point(c,r*math.cos(j*math.pi/count),r*math.sin(j*math.pi/count),d)for j in range(count+1))
        for j in range(count):faces.append([n+j,n+j+1,n+count+2+j,n+count+1+j]);indices.append(0)
    corners=g['housing_corners_local_m'];n=len(vertices)
    vertices.extend(list(Vector(c)-front*.012)for c in corners)
    vertices.extend(list(Vector(c)-front*g['housing_depth_m'])for c in corners)
    faces.extend([[n,n+1,n+2,n+3],[n+7,n+6,n+5,n+4]]);indices.extend([2,2])
    for j in range(4):faces.append([n+j,n+(j+1)%4,n+4+(j+1)%4,n+4+j]);indices.append(2)
    object_mesh('SignalHousing',vertices,faces,indices)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fbx=output/'signal_display.fbx'
    bpy.ops.export_scene.fbx(filepath=str(fbx),use_selection=False,object_types={'MESH'},
        apply_unit_scale=True,apply_scale_options='FBX_SCALE_UNITS',
        axis_forward='-Y',axis_up='Z',use_space_transform=True)
    report=dict(condition=str(Path(condition).resolve()),fbx=str(fbx.resolve()),objects=records,
        material_slots=['SignalBlack','SignalLamp','SignalHousing'],geometry=g,
        native_state_positions=q['native_state_positions'],state_scope=q['state_scope'],
        native_state_glyphs=q.get('native_state_glyphs',{}))
    (output/'mesh_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(output/'mesh_manifest.json',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--condition',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);build(a.condition,a.output)

if __name__=='__main__':main()
