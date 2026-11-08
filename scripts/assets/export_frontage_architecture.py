"""Build Blender facade details on source building footprints."""
import argparse,json,sys
from pathlib import Path


def main():
    import bpy
    import numpy as np
    from mathutils import Vector
    from mathutils.geometry import tessellate_polygon
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene'))
    from preview_coarse import building_mesh, material, ring
    from map_generation.export_geometry import footprint_area
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.output.mkdir(parents=True,exist_ok=True)
    plan=json.loads(a.plan.read_text());bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh_prefix=plan.get('mesh_prefix','frontage')
    bpy.context.scene.unit_settings.system='METRIC';bpy.context.scene.unit_settings.scale_length=1
    names=['wall','roof','trim','glass','door','foundation']
    colors=[(.6,.58,.48),(.18,.16,.13),(.72,.71,.64),(.10,.15,.18),(.24,.20,.16),(.4,.4,.36)]
    materials=[material(n,c) for n,c in zip(names,colors)];records=[];omitted=[]
    def finish_object(obj,b,mats):
        mesh=obj.data
        for mat in materials:mesh.materials.append(mat)
        uv=mesh.uv_layers.new(name='UVMap')
        for face,mat in zip(mesh.polygons,mats):
            face.material_index=mat
            direction=np.array([-face.normal.y,face.normal.x,0.]);base=np.asarray(mesh.vertices[face.vertices[0]].co)
            for loopid in face.loop_indices:
                pt=np.asarray(mesh.vertices[mesh.loops[loopid].vertex_index].co)
                uv.data[loopid].uv=((pt-base)@direction/2.,(pt[2]-b['base_z_m'])/2.) if abs(face.normal.z)<.5 else (pt[0]/2.,pt[1]/2.)
    for index,b in enumerate(plan['buildings']):
        area=footprint_area(b)
        if area<=1e-8:
            omitted.append(dict(source_id=b['id'],footprint_area_m2=area,
                reason='Numerically zero source footprint; no solid architectural mass can be formed'))
            continue
        if b.get('solid_parts') or b.get('source_mesh'):
            # Retain source openings when adding facade details.
            vertices=[];faces=[];mats=[]
            if b.get('source_mesh'):
                source=b['source_mesh']
                vertices=[list(v) for v in source['vertices_xyz_m']]
                faces=[list(f) for f in source['faces']]
                mats=[names.index(name) for name in source['face_materials']]
                if len(mats)!=len(faces):
                    raise ValueError('Explicit source mesh requires one material per face')
            for part in b.get('solid_parts',[]):
                piece=building_mesh(part,materials[0],bpy.context.collection)
                offset=len(vertices);vertices.extend(list(v.co)for v in piece.data.vertices)
                faces.extend([offset+i for i in f.vertices]for f in piece.data.polygons)
                mats.extend(1 if abs(f.normal.z)>.5 else 0 for f in piece.data.polygons)
                data=piece.data;bpy.data.objects.remove(piece,do_unlink=True);bpy.data.meshes.remove(data)
            from map_generation.observed_facades import append_source_facades
            facade_counts=append_source_facades(vertices,faces,mats,b.get('source_facades',[]),
                {name:i for i,name in enumerate(names)})
            mesh=bpy.data.meshes.new(mesh_prefix+'_'+str(index)+'_access');mesh.from_pydata(vertices,[],faces);mesh.update()
            obj=bpy.data.objects.new(mesh_prefix+'_'+str(index),mesh);bpy.context.collection.objects.link(obj)
            finish_object(obj,b,mats)
            records.append(dict(mesh=obj.name,source_id=b['id'],style=b['style'],windows=facade_counts['windows'],doors=0,
                source_facade_counts=facade_counts,material_overrides=b.get('material_overrides',{}),
                material_slots=names,source_solid_parts=len(b.get('solid_parts',[])),
                explicit_source_mesh=bool(b.get('source_mesh')),
                previous_source_ids=b.get('previous_source_ids',[b['id']]),
                previous_mesh_names=b.get('previous_mesh_names'),
                retain_primitive_material=b.get('retain_primitive_material',False),
                old_bounds_center_xy_m=((np.min(b['exterior'],axis=0)+np.max(b['exterior'],axis=0))/2).tolist()))
            continue
        style=b['style'];eave=b['roof_z_m']-style['roof_rise_m']
        mass=dict(b,height_m=eave-b['base_z_m']);obj=building_mesh(mass,materials[0],bpy.context.collection)
        obj.name=mesh_prefix+'_'+str(index)
        vertices=[list(v.co) for v in obj.data.vertices];faces=[list(f.vertices) for f in obj.data.polygons]
        mats=[1 if abs(f.normal.z)>.5 else 0 for f in obj.data.polygons]
        # Remove the flat roof for a low hip; a clipped distance-to-eave profile
        # follows concave footprints too, without an arbitrary centroid fan.
        if style['roof'] in ('low_hip', 'gable'):
            keep=[i for i,f in enumerate(obj.data.polygons) if not (f.normal.z>.5)]
            faces=[faces[i] for i in keep];mats=[mats[i] for i in keep]
            loops=[ring(b['exterior'])]+[ring(h,False) for h in b.get('holes',[])]
            flat=[(x,y,eave) for loop in loops for x,y in loop]
            triangles=tessellate_polygon([[Vector((x,y,eave)) for x,y in loop] for loop in loops])
            boundary=np.asarray([(p,q) for loop in loops for p,q in zip(loop,loop[1:]+loop[:1])])
            start,end=boundary[:,0],boundary[:,1];direction=end-start;denom=(direction**2).sum(1)
            footprint=np.asarray(b['exterior'])
            edge_vectors=np.roll(footprint,-1,axis=0)-footprint
            ridge=edge_vectors[np.linalg.norm(edge_vectors,axis=1).argmax()]
            ridge=ridge/np.linalg.norm(ridge);across=np.array([-ridge[1],ridge[0]])
            across_values=footprint@across;mid=(across_values.min()+across_values.max())/2
            half_width=(across_values.max()-across_values.min())/2
            def roof_point(x):
                if style['roof']=='gable':
                    rise=style['roof_rise_m']*max(0.,1.-abs(x[:2]@across-mid)/half_width)
                    return [*x[:2],eave+rise]
                t=np.clip(((x[:2]-start)*direction).sum(1)/denom,0,1)
                distance=np.linalg.norm(x[:2]-start-t[:,None]*direction,axis=1).min()
                return [*x[:2],eave+min(style['roof_rise_m'],float(distance)*.27)]
            def subdivide(tri):
                if max(np.linalg.norm(tri[(i+1)%3,:2]-tri[i,:2]) for i in range(3))>2.:
                    x,y,z=tri;xy=(x+y)/2;yz=(y+z)/2;zx=(z+x)/2
                    for child in [[x,xy,zx],[xy,y,yz],[zx,yz,z],[xy,yz,zx]]:subdivide(np.array(child))
                else:
                    offset=len(vertices);vertices.extend(roof_point(x) for x in tri)
                    faces.append([offset,offset+1,offset+2]);mats.append(1)
            def gable_half(triangle,side):
                clipped=[]
                for p0,p1 in zip(triangle,triangle[1:]+triangle[:1]):
                    d0=side*(p0[:2]@across-mid);d1=side*(p1[:2]@across-mid)
                    if d0>=0:clipped.append(p0)
                    if (d0<0<d1) or (d1<0<d0):
                        clipped.append(p0+(p1-p0)*(d0/(d0-d1)))
                return clipped
            for tri in triangles:
                points=[np.asarray(flat[i]) for i in tri]
                if style['roof']=='gable':
                    # Split planar roof sides at the ridge before triangulation.
                    for side in (-1,1):
                        part=gable_half(points,side)
                        for j in range(1,len(part)-1):
                            offset=len(vertices);vertices.extend(roof_point(x) for x in [part[0],part[j],part[j+1]])
                            faces.append([offset,offset+1,offset+2]);mats.append(1)
                else:subdivide(np.asarray(points))
            if style['roof']=='gable':
                # Gable end walls follow the same clipped roof field. The
                # source footprint and maximum roof Z remain unchanged.
                for loop in loops:
                    for start_xy,end_xy in zip(loop,loop[1:]+loop[:1]):
                        count=max(1,int(np.ceil(np.linalg.norm(np.asarray(end_xy)-start_xy)/2.)))
                        p=np.asarray(start_xy);q=np.asarray(end_xy)
                        d0=p@across-mid;d1=q@across-mid
                        fractions=list(np.linspace(0.,1.,count+1))
                        if d0*d1<0:fractions.append(float(d0/(d0-d1)))
                        samples=[p+(q-p)*u for u in sorted(set(fractions))]
                        for p0,p1 in zip(samples[:-1],samples[1:]):
                            q0=roof_point(np.array([*p0,eave]));q1=roof_point(np.array([*p1,eave]))
                            if max(q0[2],q1[2])<=eave+1e-8:continue
                            points=[[*p0,eave],[*p1,eave]]
                            if q1[2]>eave+1e-8:points.append(q1)
                            if q0[2]>eave+1e-8:points.append(q0)
                            offset=len(vertices);vertices.extend(points)
                            faces.append(list(range(offset,offset+len(points))));mats.append(0)
        def panel(points,mat):
            offset=len(vertices);vertices.extend(np.asarray(points).tolist());faces.append(list(range(offset,offset+len(points))));mats.append(mat)
        windows=doors=0;regional_facades=[]
        ground=b['entrance_z_m'];floors=style['floors']
        if floors>1 and 'ground_floor_height_m' in style:
            first=min(style['ground_floor_height_m'],(eave-ground)-1.8*(floors-1))
            first=max(1.8,first)
            floor_levels=[ground]+np.linspace(ground+first,eave,floors).tolist()
        else:floor_levels=np.linspace(ground,eave,floors+1).tolist()
        loops=[ring(b['exterior'])]+[ring(h,False) for h in b.get('holes',[])]
        for loop in loops:
            for p0,p1 in zip(loop,loop[1:]+loop[:1]):
                p0=np.asarray(p0);p1=np.asarray(p1);length=np.linalg.norm(p1-p0);along=(p1-p0)/length
                outward=np.array([along[1],-along[0]])
                def rect(u0,u1,z0,z1,mat,offset=.02):
                    lo=p0+along*u0+outward*offset;hi=p0+along*u1+outward*offset
                    panel([[*lo,z0],[*hi,z0],[*hi,z1],[*lo,z1]],mat)
                rect(0,length,eave-.15,eave+.04,1,.04)
                features=True
                if style.get('feature_face_policy')=='frontage':
                    direction=np.asarray(b['frontage_direction_xy']);direction=direction/np.linalg.norm(direction)
                    features=float(outward@direction)>.5
                ground_spec=style.get('ground_floor_facade',{}) if features else {}
                facade=dict(start_xy_m=p0.tolist(),end_xy_m=p1.tolist(),outward_xy=outward.tolist(),
                    z_range_m=[ground,eave],piers=[],balconies=[])
                pier_spacing=style.get('vertical_pier_spacing_m',0) if features else 0
                if pier_spacing>0:
                    for u in np.arange(pier_spacing/2,length,pier_spacing):
                        facade['piers'].append(dict(u_m=float(u),width_m=style.get('vertical_pier_width_m',.18)))
                # A restrained base band and sill line break repeated full-wall
                # textures. Foundation extends below the final sampled surface.
                rect(0,length,b['base_z_m'],ground+.25,5,.025)
                for floor in range(floors):
                    bottom,top=floor_levels[floor:floor+2]
                    if floor and style.get('floor_bands',True):rect(0,length,bottom-.08,bottom+.08,2,.045)
                    ground_feature=bool(floor==0 and ground_spec)
                    spacing=ground_spec.get('bay_spacing_m',style['bay_spacing_m']) if ground_feature else style['bay_spacing_m']
                    count=int(length/spacing)
                    for bay in range(count):
                        center=(bay+.5)*length/count
                        width=min(ground_spec.get('opening_width_m',style['window_width_m']) if ground_feature else style['window_width_m'],length/count-(.35 if ground_feature else .6))
                        if width<.5:continue
                        sill=ground_spec.get('sill_height_m',.12) if ground_feature else .85
                        opening_height=ground_spec.get('opening_height_m',top-bottom-.4) if ground_feature else style['window_height_m']
                        z0=bottom+sill;z1=min(z0+opening_height,top-.25 if ground_feature else top-.45)
                        if z1-z0<.5:continue
                        u0,u1=center-width/2,center+width/2
                        rect(u0-.055,u1+.055,z0-.055,z1+.055,2,.035)
                        material_index=4 if ground_feature and ground_spec.get('mode')=='loading_bays' else 3
                        rect(u0,u1,z0,z1,material_index,.045)
                        if material_index==3:rect(center-.025,center+.025,z0,z1,2,.055)
                        rect(u0-.1,u1+.1,z0-.09,z0-.04,2,.09)
                        windows+=1
                # Door uses a remaining bay near an edge, away from corners.
                if length>6 and doors==0 and not ground_spec:
                    u0=.4;u1=1.35;z0=ground+.03;z1=min(z0+2.1,eave-.2)
                    rect(u0-.08,u1+.08,z0,z1+.08,2,.06);rect(u0,u1,z0,z1,4,.07);doors+=1
                balcony=style.get('balcony') if features else None
                if balcony:
                    count=int(length/balcony.get('bay_spacing_m',style['bay_spacing_m']))
                    for floor in range(1,floors):
                        for bay in range(count):
                            center=(bay+.5)*length/count;width=min(balcony.get('width_m',2.),length/count-.4)
                            if width<=.3:continue
                            facade['balconies'].append(dict(u_range_m=[center-width/2,center+width/2],
                                floor_z_m=floor_levels[floor],depth_m=balcony['depth_m'],
                                rail_height_m=balcony.get('rail_height_m',1.05),rail_spacing_m=balcony.get('rail_spacing_m',.4)))
                if facade['piers'] or facade['balconies']:regional_facades.append(facade)
        from map_generation.observed_facades import append_source_facades
        regional_counts=append_source_facades(vertices,faces,mats,regional_facades,{name:i for i,name in enumerate(names)})
        mesh=bpy.data.meshes.new(obj.name+'_detailed');mesh.from_pydata(vertices,[],faces);mesh.update()
        old=obj.data;obj.data=mesh;bpy.data.meshes.remove(old)
        finish_object(obj,b,mats)
        records.append({'mesh':obj.name,'source_id':b['id'],'style':style,'windows':windows,'doors':doors,
                        'material_slots':names,'material_overrides':b.get('material_overrides',{}),'regional_facade_counts':regional_counts,
                        'floor_levels_m':floor_levels,'frontage_direction_xy':b.get('frontage_direction_xy'),
                        'previous_source_ids':b.get('previous_source_ids',[b['id']]),'previous_mesh_names':b.get('previous_mesh_names'),
                        'retain_primitive_material':b.get('retain_primitive_material',False),'old_bounds_center_xy_m':((np.min(b['exterior'],axis=0)+np.max(b['exterior'],axis=0))/2).tolist()})
    from scripts.scene.export_source_asset_subset import triangulate_object
    triangulation=[triangulate_object(obj) for obj in list(bpy.context.scene.objects) if obj.type=='MESH']
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.export_scene.fbx(filepath=str(a.output/'frontage.fbx'),use_selection=True,axis_forward='-Y',axis_up='Z',global_scale=1,
                            apply_unit_scale=True,object_types={'MESH'},use_mesh_modifiers=True,add_leaf_bones=False)
    (a.output/'architecture.json').write_text(json.dumps({'buildings':records,'plan':str(a.plan.resolve()),'omitted_degenerate_geometry':omitted,'explicit_triangulation':triangulation,
        'scope':plan.get('scope', 'Native source-footprint architectural completion with explicitly inferred facade details.')},indent=2)+'\n')


if __name__=='__main__':main()
