"""Import source-selected display meshes into existing native lamp components."""
import json,math
from pathlib import Path

def install_display_meshes(unreal, settings, components):
    from scripts.build.install_observed_separations import native_material
    from scripts.build.install_observed_signal_heads import emitter_position
    manifest=json.loads(Path(settings['display_mesh_manifest']).read_text())
    lib=unreal.EditorAssetLibrary
    root=settings['display_mesh_root'].rstrip('/')
    task=unreal.AssetImportTask();task.filename=manifest['fbx'];task.destination_path=root
    task.automated=True;task.replace_existing=True;task.save=True
    options=unreal.FbxImportUI();options.import_mesh=True;options.import_materials=False
    options.import_textures=False;options.import_as_skeletal=False
    options.mesh_type_to_import=unreal.FBXImportType.FBXIT_STATIC_MESH
    data=options.static_mesh_import_data
    data.combine_meshes=False;data.convert_scene=True;data.convert_scene_unit=True
    data.transform_vertex_to_absolute=True;data.generate_lightmap_u_vs=False;data.auto_generate_collision=False
    task.options=options;unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    imported=[lib.load_asset(p)for p in task.imported_object_paths]
    imported=[m for m in imported if isinstance(m,unreal.StaticMesh)]
    black=native_material(unreal,root,'SignalBlack',(.008,.009,.011,1),True)
    housing=native_material(unreal,root,'SignalHousing',(.58,.37,.035,1),True)
    mappings={'SignalFace_Red':'RedLight','SignalFace_Yellow':'YellowLight',
              'SignalFace_Green':'GreenLight','SignalHousing':'Roadsign_TrafficLights_BotCover'}
    selected={}
    records=[]
    for entry in manifest['objects']:
        matches=[m for m in imported if m.get_name()==entry['name']or m.get_name().endswith('_'+entry['name'])]
        if len(matches)!=1:raise RuntimeError('Expected one source display mesh: '+entry['name'])
        mesh=matches[0];low,high=entry['local_bounds_m'];b=mesh.get_bounds()
        expected=[(low[i]+high[i])*50*(1 if i!=1 else -1)for i in range(3)]
        expected +=[(high[i]-low[i])*50 for i in range(3)]
        actual=[b.origin.x,b.origin.y,b.origin.z,b.box_extent.x,b.box_extent.y,b.box_extent.z]
        error=max(abs(a-b)for a,b in zip(actual,expected))
        if error>.1:raise ValueError('Source display FBX bounds error >1mm: '+entry['name'])
        name=mappings[entry['name']];component=components[name]
        emission=component.get_material(1)if entry['name']!='SignalHousing'else black
        # Native Blueprint changes lamp slot1; all other slots remain source material.
        selected[name]=(mesh,emission,entry)
        records.append(dict(component=name,mesh=mesh.get_path_name(),bounds_error_cm=error,
                            source_bounds_m=entry['local_bounds_m']))
    def depth(name):
        c=components[name].get_attach_parent();value=0
        while c is not None:value+=1;c=c.get_attach_parent()
        return value
    for name in sorted(selected,key=depth):
        mesh,emission,entry=selected[name];c=components[name];c.set_static_mesh(mesh)
        c.set_world_location_and_rotation(unreal.Vector(0,0,0),unreal.Rotator(),False,True)
        c.set_world_scale3d(unreal.Vector(1,1,1))
        # FBX removes unused slots (housing has Black/Housing, no Lamp slot).
        named={'SignalBlack':black,'SignalLamp':emission,'SignalHousing':housing}
        slot_names=[str(s.material_slot_name) for s in mesh.get_editor_property('static_materials')]
        if entry['name']!='SignalHousing' and slot_names.index('SignalLamp')!=1:
            raise ValueError('Native lamp event requires emission at slot1')
        for i,slot in enumerate(slot_names):
            c.set_material(i,named[slot]);mesh.set_material(i,named[slot])
        c.set_visibility(True,False);c.set_hidden_in_game(False);c.set_collision_profile_name('NoCollision')
        records.append(dict(component=name,material_slots=slot_names,
                            bound_materials=[named[n].get_path_name() for n in slot_names]))
        if not lib.save_loaded_asset(mesh,only_if_is_dirty=False):raise RuntimeError('Cannot save source display mesh')
    for name in ('Roadsign_TrafficLights_TopCover',):
        components[name].set_visibility(False,False);components[name].set_hidden_in_game(True)
    center_checks=[]
    for state,pos in manifest['native_state_positions'].items():
        c=components[state+'Light'];actual=emitter_position(c)
        target=manifest['geometry']['lens_centers_local_m'][int(pos)-1]
        desired=[target[0]*100,-target[1]*100,target[2]*100]
        error=math.sqrt(sum((a-b)**2 for a,b in zip(actual,desired)))/100
        center_checks.append(dict(state=state,position=pos,expected_local_m=target,
            actual_local_m=[actual[0]/100,-actual[1]/100,actual[2]/100],bounds_center_error_m=error))
    if max(r['bounds_center_error_m']for r in center_checks)>.002:
        raise RuntimeError('Source display native centers changed: '+str(center_checks))
    return dict(meshes=records,centers=center_checks,source_manifest=settings['display_mesh_manifest'],
        state_scope=manifest['state_scope'],native_controller_logic_changed=False,
        scope='Finite state-driven glyph geometry; source flashing/yield and unseen face roles are not established.')
