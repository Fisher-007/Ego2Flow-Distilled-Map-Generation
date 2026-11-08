"""Place source-measured signal displays using the native CARLA controller Blueprint."""
import json
import math
from pathlib import Path


def emitter_position(component):
    import unreal
    transform = component.get_world_transform()
    bounds = component.get_editor_property('static_mesh').get_bounds()
    point = unreal.MathLibrary.transform_location(transform, bounds.origin)
    return [point.x, point.y, point.z]


def orient_upright_lamp_hoods(unreal, components, names, yaw):
    """Keep native lamp hoods upright without moving their observed centers.

    Horizontal head rows still use horizontal backing geometry. The selected
    native lamp meshes retain the downward hood opening used by upright heads.
    """
    records = []
    for name in names[:3]:
        component = components[name]
        center = emitter_position(component)
        location = component.get_world_location()
        component.set_world_location_and_rotation(
            location, unreal.Rotator(yaw=yaw), False, True)
        rotated = emitter_position(component)
        component.set_world_location(location + unreal.Vector(
            *[center[i]-rotated[i] for i in range(3)]), False, True)
        actual = emitter_position(component)
        error = math.sqrt(sum((a-b)**2 for a,b in zip(actual,center)))/100
        if error > .002:
            raise RuntimeError('Native lamp hood rotation moved its center: '+name)
        records.append(dict(component=name, center_before_carla_cm=center,
            center_after_carla_cm=actual, center_error_m=error,
            upright_yaw_carla_deg=yaw))
    return records


def place_heads(unreal, settings):
    level, lib = unreal.EditorLevelLibrary, unreal.EditorAssetLibrary
    asset = '/Game/Carla/Static/TrafficLight/Streetlights_01/BP_TrafficLight'
    bp = lib.load_asset(asset)
    yaw = 90.-math.degrees(settings['approach_heading_local_rad'])
    angle = math.radians(yaw)
    records = []
    for head in settings['heads']:
        actor = level.spawn_actor_from_object(bp, unreal.Vector(0,0,0))
        label = 'MAP_ObservedHead_'+settings['signal_id']+'_'+str(head['head'])
        actor.set_actor_label(label)
        components = {c.get_name():c for c in actor.get_components_by_class(unreal.StaticMeshComponent)}
        green = components['GreenLight']
        original = emitter_position(green)
        target = head['green_emitter_local_m']
        desired = [100*target[0],-100*target[1],100*target[2]]
        offset = [math.cos(angle)*original[0]-math.sin(angle)*original[1],
                  math.sin(angle)*original[0]+math.cos(angle)*original[1],original[2]]
        actor.set_actor_location_and_rotation(unreal.Vector(*[desired[i]-offset[i]for i in range(3)]),
                                              unreal.Rotator(yaw=yaw),False,False)
        pole = components['Cylinder']
        pole.set_visibility(False,True)
        pole.set_hidden_in_game(True)
        pole.set_collision_profile_name('NoCollision')
        for name, component in components.items():
            if name!='Cylinder':
                component.set_collision_profile_name('NoCollision')
        actual = emitter_position(green)
        component = actor.get_traffic_light_component()
        records.append(dict(label=label,blueprint=asset,signal_id=settings['signal_id'],
                            head_index=head['head'],green_emitter_local_m=[actual[0]/100,-actual[1]/100,actual[2]/100],
                            measured_green_emitter_local_m=target,
                            error_m=math.sqrt(sum((actual[i]-desired[i])**2 for i in range(3)))/100,
                            actor_origin_carla_cm=[actor.get_actor_location().x,actor.get_actor_location().y,actor.get_actor_location().z],
                            yaw_carla_deg=yaw,traffic_component=component.get_name(),
                            hidden_stock_pole=True,controller_binding='pending same source controller binding',
                            source_observation=head))
    return records




def prepare_emissive_blueprint(unreal, stock_asset, material_root):
    """Copy native state logic with its colored material references using Unreal."""
    lib=unreal.EditorAssetLibrary
    root=material_root.rstrip('/')+'/NativeDisplay'
    source_root=stock_asset.rsplit('/',1)[0]
    packages={stock_asset:root+'/'+stock_asset.rsplit('/',1)[1]}
    names={'Red':'MI_Streetlights_2','Yellow':'MI_Streetlights_1','Green':'MI_Streetlights_3'}
    packages.update({source_root+'/'+name:root+'/'+name for name in names.values()})
    # Load the source Blueprint before AdvancedCopyPackages.
    for source in packages:
        if lib.load_asset(source) is None:
            raise RuntimeError('Cannot load native signal copy source: '+source)
    # Reuse an already persisted complete set. Replacing referenced materials
    # after a level loads can fail duplication and retain stock Blueprint links.
    if not all(lib.does_asset_exist(destination) for destination in packages.values()):
        if not unreal.MapBuilderAssetLibrary.copy_asset_packages(packages):
            raise RuntimeError('Native reference-preserving signal asset copy failed')
    materials={}
    for color,name in names.items():
        material=lib.load_asset(root+'/'+name)
        unreal.MaterialEditingLibrary.set_material_instance_scalar_parameter_value(material,'On/Off',1.)
        if not lib.save_loaded_asset(material):raise RuntimeError('Cannot save city signal material')
        materials[color]=material.get_path_name()
    blueprint=lib.load_asset(packages[stock_asset])
    if not lib.save_loaded_asset(blueprint,only_if_is_dirty=False):
        raise RuntimeError('Cannot save copied native signal Blueprint')
    return packages[stock_asset],materials



def repeat_display_geometry(unreal, settings, components, names):
    """Repeat lamp meshes within the existing state-driven component and material slots."""
    extra = settings['heads'][2:]
    if not extra:
        return []
    lib = unreal.EditorAssetLibrary
    root = settings['display_mesh_root'].rstrip('/')
    base = settings['heads'][0]['green_emitter_local_m']
    offsets = [[100*(h['green_emitter_local_m'][0]-base[0]),
                -100*(h['green_emitter_local_m'][1]-base[1]),
                100*(h['green_emitter_local_m'][2]-base[2])] for h in extra]
    assets = []
    for name in names:
        component = components[name]
        source = component.get_editor_property('static_mesh')
        transform = component.get_world_transform()
        origin = transform.translation
        local_origin = unreal.MathLibrary.inverse_transform_location(transform, origin)
        copies = [unreal.Transform()]
        for offset in offsets:
            target = origin + unreal.Vector(*offset)
            local = unreal.MathLibrary.inverse_transform_location(transform, target)-local_origin
            # UE4 Python keyword construction otherwise zeroes unspecified scale.
            copy = unreal.Transform()
            copy.translation = local
            copies.append(copy)
        path = root + '/' + settings['signal_id'] + '_' + name
        mesh = lib.load_asset(path) if lib.does_asset_exist(path) else lib.duplicate_asset(source.get_path_name(), path)
        if mesh is None or not unreal.MapBuilderAssetLibrary.repeat_static_mesh(source, mesh, copies):
            raise RuntimeError('Cannot compose additional source head geometry: '+path)
        component.set_static_mesh(mesh)
        if not lib.save_loaded_asset(mesh, only_if_is_dirty=False):
            raise RuntimeError('Cannot persist repeated source head geometry: '+path)
        assets.append(dict(component=name, source_mesh=source.get_path_name(),
                           native_mesh=mesh.get_path_name(), copies=len(copies),
                           material_slots=component.get_num_materials()))
    return [dict(head=h['head'], green_emitter_local_m=h['green_emitter_local_m'],
                 source_observation=h, native_meshes=assets,
                 placement_method='Existing first-head meshes translated in component space; material slots and native state event retained',
                 native_visibility='pending actual source-view and state verification')
            for h in extra]


def place_control(unreal, settings, anchor, controller_id, group=None):
    """Keep the native control anchor while positioning observed displays."""
    level,lib=unreal.EditorLevelLibrary,unreal.EditorAssetLibrary
    asset='/Game/Carla/Static/TrafficLight/Streetlights_01/BP_TrafficLight02'
    emission_materials={}
    if settings.get('emission_material_root'):
        asset,emission_materials=prepare_emissive_blueprint(unreal,asset,settings['emission_material_root'])
    origin=unreal.Vector(*[100*x for x in anchor['location_carla_m']])
    yaw=90.-math.degrees(settings['approach_heading_local_rad'])
    actor=level.spawn_actor_from_object(lib.load_asset(asset),origin,unreal.Rotator(yaw=yaw))
    actor.set_actor_label('MAP_ObservedSignal_'+settings['signal_id'])
    components={c.get_name():c for c in actor.get_components_by_class(unreal.StaticMeshComponent)}
    groups=[
        ['GreenLight','YellowLight','RedLight','Roadsign_TrafficLights_BotCover','Roadsign_TrafficLights_TopCover'],
        ['GreenLight2','YellowLight2','RedLight2','basecover','StreetLight2']]
    visible=set(sum(groups[:min(len(settings['heads']),2)],[]));records=[]
    # Serialize the original colored materials used by Blueprint initialization.
    original_materials = {name: [c.get_material(i) for i in range(c.get_num_materials())]
                          for name,c in components.items()}
    for name,c in components.items():
        c.set_collision_profile_name('NoCollision')
        if name not in visible:
            c.set_visibility(False,True);c.set_hidden_in_game(True)
    # Stock mast visibility propagates through its child hierarchy. Restore only
    # the selected head components after hiding unwanted stock geometry.
    for name in visible:
        components[name].set_visibility(True,False)
        components[name].set_hidden_in_game(False)
    targets={};rotations={}
    for head,names in zip(settings['heads'],groups):
        color=head.get('emitter_color','Green')
        emitter=names[{'Green':0,'Yellow':1,'Red':2}[color]]
        current=emitter_position(components[emitter])
        target=head.get('emitter_local_m',head.get('green_emitter_local_m'))
        desired=[100*target[0],-100*target[1],100*target[2]]
        offset=[desired[i]-current[i]for i in range(3)]
        for name in names:
            p=components[name].get_world_location()
            if head.get('layout') == 'horizontal':
                # Stock heads share upright mesh axes. Pitch around local Y:
                # top/red -> viewer-left, while the original facing is retained.
                delta=unreal.MathLibrary.inverse_transform_direction(actor.get_actor_transform(),
                    p-unreal.Vector(*current))
                turned=unreal.Vector(-delta.z,delta.y,delta.x)
                world=unreal.MathLibrary.transform_direction(actor.get_actor_transform(),turned)
                targets[name]=[desired[0]+world.x,desired[1]+world.y,desired[2]+world.z]
                rotations[name]=unreal.Rotator(pitch=90.,yaw=yaw)
            else:
                targets[name]=[p.x+offset[0],p.y+offset[1],p.z+offset[2]]
    def depth(name):
        parent=components[name].get_attach_parent();value=0
        while parent is not None:
            value+=1;parent=parent.get_attach_parent()
        return value
    for name in sorted(targets,key=depth):
        if name in rotations:
            components[name].set_world_location_and_rotation(unreal.Vector(*targets[name]),rotations[name],False,True)
        else:
            components[name].set_world_location(unreal.Vector(*targets[name]),False,True)
    upright_hood_records=[]
    for head,names in zip(settings['heads'],groups):
        if head.get('upright_lamp_hoods', False):
            upright_hood_records.extend(orient_upright_lamp_hoods(unreal,components,names,yaw))
    for head,names in zip(settings['heads'],groups):
        color=head.get('emitter_color','Green')
        emitter=names[{'Green':0,'Yellow':1,'Red':2}[color]]
        actual=emitter_position(components[emitter]);green=emitter_position(components[names[0]])
        target=head.get('emitter_local_m',head.get('green_emitter_local_m'))
        desired=[100*target[0],-100*target[1],100*target[2]]
        records.append(dict(head=head['head'],layout=head.get('layout','vertical'),components=names,green_emitter_local_m=[green[0]/100,-green[1]/100,green[2]/100],
                            measured_emitter_color=color,emitter_local_m=[actual[0]/100,-actual[1]/100,actual[2]/100],
                            error_m=math.sqrt(sum((actual[i]-desired[i])**2 for i in range(3)))/100))
    if max(r['error_m']for r in records)>.002:raise RuntimeError('Observed signal display placement exceeds 2 mm: '+json.dumps(records))
    records.extend(repeat_display_geometry(unreal, settings, components, groups[0]))
    display_mesh_review=None
    if settings.get('display_mesh_manifest'):
        from scripts.build.observed_signal_meshes import install_display_meshes
        display_mesh_review=install_display_meshes(unreal,settings,components)
        original_materials = {name: [c.get_material(i) for i in range(c.get_num_materials())]
                              for name,c in components.items()}
    if group is None:
        group_path='/Game/MAPDevelopment/SourceControls/BP_SourceTrafficLightGroup'
        group_asset=lib.load_asset(group_path) if lib.does_asset_exist(group_path) else None
        if group_asset is None:
            factory=unreal.BlueprintFactory()
            factory.set_editor_property('parent_class',unreal.TrafficLightGroup)
            group_asset=unreal.AssetToolsHelpers.get_asset_tools().create_asset('BP_SourceTrafficLightGroup','/Game/MAPDevelopment/SourceControls',unreal.Blueprint,factory)
            if group_asset is None:raise RuntimeError('Cannot create native traffic-group Blueprint asset')
            lib.save_loaded_asset(group_asset)
        group=level.spawn_actor_from_object(group_asset,origin)
        group.set_actor_label('MAP_SourceSignalGroup_'+settings['source_junction_id'])
        group.set_editor_property('junction_id',int(settings['source_junction_id']))
    controller=next((c for c in group.get_controllers()if c.get_controller_id()==str(controller_id)),None)
    if controller is None:
        controller=unreal.TrafficLightController(outer=group,name='MAPController_'+str(controller_id))
        controller.set_controller_id(str(controller_id));group.add_controller(controller)
    component=actor.get_traffic_light_component()
    component.set_sign_id(settings['signal_id']);controller.add_traffic_light(component)
    states=[]
    for state in [unreal.TrafficLightState.RED,unreal.TrafficLightState.YELLOW,unreal.TrafficLightState.GREEN]:
        controller.set_traffic_lights_state(state)
        states.append(dict(requested=str(state),observed=str(component.get_light_state()),
            lamp_materials={color: components[color+"Light"].get_material(1).get_path_name() for color in ("Red","Yellow","Green")}))
    if emission_materials:
        for color,row in zip(('Red','Yellow','Green'),states):
            if row['lamp_materials'][color]!=emission_materials[color]:
                raise RuntimeError('Native state event did not select copied emission material: '+str(row))
    controller.reset_state()
    for name, materials in original_materials.items():
        for index, material in enumerate(materials):
            components[name].set_material(index, material)
    return actor,dict(upright_lamp_hoods=upright_hood_records,display_mesh_review=display_mesh_review,emission_materials=emission_materials,original_display_materials_restored=True,
label=actor.get_actor_label(),blueprint=asset,signal_id=component.get_sign_id(),
        controller_id=component.get_controller().get_controller_id(),junction_id=component.get_group().get_junction_id(),
        heads=records,logical_anchor_carla_m=anchor['location_carla_m'],
        controller_light_count=len(controller.get_traffic_lights()),group_controller_count=len(group.get_controllers()),
        state_checks=states,scope='Native serialized component/controller link. Full map runtime trigger and phase bindings still require verification.')




def preserve_group_companions(unreal, group, members):
    """Serialize unchanged companion controls in the same native junction group."""
    level,lib=unreal.EditorLevelLibrary,unreal.EditorAssetLibrary
    records=[]
    for member in members:
        controller_id=str(member['controller_id'])
        controller=next((c for c in group.get_controllers() if c.get_controller_id()==controller_id),None)
        if controller is None:
            controller=unreal.TrafficLightController(outer=group,name='MAPController_'+controller_id)
            controller.set_controller_id(controller_id);group.add_controller(controller)
        existing=next((c for c in controller.get_traffic_lights() if c.get_sign_id()==member['signal_id']),None)
        if existing is not None:continue
        asset=member.get('blueprint','/Game/Carla/Blueprints/TrafficLight/BP_TLOpenDrive_RHT')
        pitch,yaw,roll=member['rotation_carla_deg']
        actor=level.spawn_actor_from_object(lib.load_asset(asset),
            unreal.Vector(*[100*x for x in member['location_carla_m']]),
            unreal.Rotator(pitch=pitch,yaw=yaw,roll=roll))
        actor.set_actor_label('MAP_SourceSignalCompanion_'+member['signal_id'])
        component=actor.get_traffic_light_component()
        component.set_sign_id(member['signal_id']);controller.add_traffic_light(component)
        records.append(dict(signal_id=component.get_sign_id(),controller_id=controller_id,
                            blueprint=asset,location_carla_m=member['location_carla_m'],
                            rotation_carla_deg=member['rotation_carla_deg'],
                            junction_id=component.get_group().get_junction_id()))
    return records


def preserve_native_groups(unreal, groups):
    """Persist selected neighboring controls at their original anchors."""
    level,lib=unreal.EditorLevelLibrary,unreal.EditorAssetLibrary
    records=[]
    for settings in groups:
        junction=int(settings['junction_id'])
        group=next((a for a in level.get_all_level_actors()
                    if isinstance(a,unreal.TrafficLightGroup)
                    and a.get_junction_id()==junction),None)
        if group is None:
            group_asset=lib.load_asset('/Game/MAPDevelopment/SourceControls/BP_SourceTrafficLightGroup')
            if group_asset is None:
                raise RuntimeError('Missing existing native traffic-group Blueprint')
            location=settings['members'][0]['location_carla_m']
            group=level.spawn_actor_from_object(group_asset,unreal.Vector(*[100*x for x in location]))
            group.set_actor_label('MAP_RetainedSignalGroup_'+str(junction))
            group.set_editor_property('junction_id',junction)
        added=preserve_group_companions(unreal,group,settings['members'])
        actual=[dict(controller_id=c.get_controller_id(),
                     signal_id=x.get_sign_id())
                for c in group.get_controllers() for x in c.get_traffic_lights()]
        expected={(str(m['controller_id']),str(m['signal_id'])) for m in settings['members']}
        if not expected.issubset({(m['controller_id'],m['signal_id']) for m in actual}):
            raise RuntimeError('Retained native signal identities were not serialized')
        records.append(dict(junction_id=junction,added=added,members=actual,
                            evidence=settings.get('evidence',[])))
    return records



def support_segments(settings):
    """Return explicit reviewed segments or the original source mast layout."""
    if settings.get('support_segments') is not None:
        return [(s['id'], list(s['from_local_xyz_m']), list(s['to_local_xyz_m']),
                 float(s['diameter_m'])) for s in settings['support_segments']]
    support=settings['mast_support'];join=list(support['join_local_xyz_m'])
    base=float(support['ground_z_m']);dims=support['inferred_dimensions']
    far_head=max([h for h in settings['heads'] if h.get('support_style') != 'independent'],key=lambda h:math.hypot(h['green_emitter_local_m'][0]-join[0],h['green_emitter_local_m'][1]-join[1]))['green_emitter_local_m']
    dx,dy=far_head[0]-join[0],far_head[1]-join[1];length=math.hypot(dx,dy)
    extension=dims['mast_extension_beyond_left_head_m']
    end=[far_head[0]+extension*dx/length,far_head[1]+extension*dy/length,join[2]]
    pole_top=join[2]+dims['pole_extension_above_join_m']
    segments=[('pole',[join[0],join[1],base],[join[0],join[1],pole_top],dims['pole_diameter_m']),
              ('arm',join,end,dims['mast_diameter_m'])]
    segments.extend((s['id'],s['from_local_xyz_m'],s['to_local_xyz_m'],s['diameter_m'])
                    for s in settings.get('additional_support_segments',[]))
    return segments


def place_mast(unreal, settings, label_prefix=None):
    """Realize the recorded source-constrained support with inferred dimensions."""
    support=settings['mast_support'];segments=support_segments(settings)
    level,lib=unreal.EditorLevelLibrary,unreal.EditorAssetLibrary
    mesh=lib.load_asset('/Engine/BasicShapes/Cylinder');bounds=mesh.get_bounds()
    records=[]
    for name,a,b,diameter in segments:
        ca=[100*a[0],-100*a[1],100*a[2]];cb=[100*b[0],-100*b[1],100*b[2]]
        delta=[y-x for x,y in zip(ca,cb)];distance=math.sqrt(sum(x*x for x in delta))
        rotation=unreal.MathLibrary.make_rot_from_z(unreal.Vector(*delta))
        actor=level.spawn_actor_from_object(mesh,unreal.Vector(*[(x+y)/2 for x,y in zip(ca,cb)]),rotation)
        actor.set_actor_label((label_prefix or 'MAP_ObservedSignal_'+settings['signal_id'])+'_'+name)
        actor.set_actor_scale3d(unreal.Vector(100*diameter/(2*bounds.box_extent.x),
            100*diameter/(2*bounds.box_extent.y),distance/(2*bounds.box_extent.z)))
        component=actor.get_component_by_class(unreal.StaticMeshComponent)
        component.set_collision_profile_name('BlockAll')
        records.append(dict(label=actor.get_actor_label(),from_local_xyz_m=a,to_local_xyz_m=b,
            diameter_m=diameter,ground_support_evidence=support.get('ground_support_evidence'),
            shape='Existing engine cylinder; dimensions inferred in source condition'))
    return records


def configure(map_path,conditions_file,output,source_map=None):
    import unreal
    level=unreal.EditorLevelLibrary
    if source_map is not None:
        if unreal.EditorAssetLibrary.does_asset_exist(map_path):
            raise ValueError('Use a new target map when cloning source controls')
        if not level.new_level_from_template(map_path,source_map):
            raise RuntimeError('Cannot clone source map for native controls')
    elif not level.load_level(map_path):raise RuntimeError('Cannot load target city map')
    q=json.loads(Path(conditions_file).read_text())
    selected=[r['native_construction']['settings']for r in q['native_objects']
        if r.get('native_construction',{}).get('kind')=='observed_signal_heads']
    records=[];by_junction={}
    for settings in selected:
        by_junction.setdefault(int(settings['source_junction_id']),[]).append(settings)
    for junction, selections in by_junction.items():
        members={member['signal_id']:member for settings in selections
                 for member in settings['native_group_members']}
        ids=set(members);selected_ids={s['signal_id']for s in selections}
        prefixes=['MAP_ObservedSignal_'+key for key in selected_ids]
        actors=list(level.get_all_level_actors())
        for actor in actors:
            if actor.get_actor_label().startswith('MAP_ObservedSignal_') and hasattr(actor,'get_traffic_light_component'):
                component=actor.get_traffic_light_component()
                if component is not None and component.get_sign_id()in ids-selected_ids:
                    raise ValueError('Include all observed controls of the affected junction in conditions: '+str(junction))
        for actor in actors:
            remove=any(actor.get_actor_label()==prefix or actor.get_actor_label().startswith(prefix+'_') for prefix in prefixes)
            if hasattr(actor,'get_traffic_light_component'):
                component=actor.get_traffic_light_component()
                remove=remove or (component is not None and component.get_sign_id()in ids)
            if remove:level.destroy_actor(actor)
        for actor in list(level.get_all_level_actors()):
            if isinstance(actor,unreal.TrafficLightGroup) and actor.get_junction_id()==junction:
                level.destroy_actor(actor)
        # Create all observed directions before restoring companion controls.
        group=None;group_records=[]
        for settings in selections:
            anchor=members[settings['signal_id']]
            actor,record=place_control(unreal,settings,anchor,anchor['controller_id'],group)
            group=actor.get_traffic_light_component().get_group()
            record['mast']=place_mast(unreal,settings)
            group_records.append(record)
        companions=preserve_group_companions(unreal,group,list(members.values()))
        actual=[dict(controller_id=c.get_controller_id(),signal_ids=[x.get_sign_id()for x in c.get_traffic_lights()])
                for c in group.get_controllers()]
        for record in group_records:
            record['companions']=companions;record['group_members']=actual
        records.extend(group_records)
    retained=preserve_native_groups(unreal,q.get('retained_native_signal_groups',[]))
    if not level.save_current_level():raise RuntimeError('Cannot save observed source controls')
    Path(output).write_text(json.dumps(dict(map=map_path,source_map=source_map,conditions=conditions_file,objects=records,
        retained_native_groups=retained,
        source_phase_programs_modified=False,source_traffic_permissions_modified=False,
        scope='Source-measured displays and source-constrained inferred support; actual full-map source-view and runtime review pending.'),indent=2)+'\n')


def rear_support_segment(segment, face_centers, rear_direction, clearance_m):
    """Shift an inferred cylinder behind fixed face planes, keeping its shape."""
    name, a, b, diameter = segment
    norm = math.sqrt(sum(x*x for x in rear_direction))
    direction = [x/norm for x in rear_direction]
    depth = min(sum((point[i]-face[i])*direction[i] for i in range(3))
                for point in (a,b) for face in face_centers)
    shift = max(0., diameter/2.+clearance_m-depth)
    return dict(id=name, from_local_xyz_m=[a[i]+shift*direction[i] for i in range(3)],
                to_local_xyz_m=[b[i]+shift*direction[i] for i in range(3)],
                diameter_m=diameter, inferred_rear_shift_m=shift,
                fixed_display_centers=face_centers, rear_direction=direction,
                face_plane_clearance_m=clearance_m)


def native_control_snapshot(unreal):
    """Record existing displays and native bindings without changing lamp state."""
    rows=[]
    for actor in unreal.EditorLevelLibrary.get_all_level_actors():
        if not hasattr(actor, 'get_traffic_light_component'):
            continue
        control=actor.get_traffic_light_component()
        if control is None:
            continue
        components=[]
        for c in actor.get_components_by_class(unreal.StaticMeshComponent):
            mesh=c.get_editor_property('static_mesh')
            t=c.get_world_transform()
            components.append(dict(name=c.get_name(),
                mesh=mesh.get_path_name() if mesh is not None else None,
                transform=[[float(getattr(v,k)) for k in keys]
                    for v,keys in [(t.translation,'xyz'),(t.rotation,'xyzw'),(t.scale3d,'xyz')]],
                materials=[c.get_material(i).get_path_name() if c.get_material(i) is not None else None
                           for i in range(c.get_num_materials())]))
        controller=control.get_controller();group=control.get_group()
        rows.append(dict(label=actor.get_actor_label(),signal_id=control.get_sign_id(),
            controller_id=controller.get_controller_id() if controller is not None else None,
            junction_id=group.get_junction_id() if group is not None else None,
            components=sorted(components,key=lambda x:x['name'])))
    return sorted(rows,key=lambda x:(x['signal_id'],x['label']))


def configure_supports(map_path, conditions_file, output, source_map=None):
    """Replace only source support actors; retain all existing display components."""
    import unreal
    if source_map is not None:
        raise ValueError('Clone the native map before a support-only update')
    level=unreal.EditorLevelLibrary
    if not level.load_level(map_path):
        raise RuntimeError('Cannot load support update target')
    q=json.loads(Path(conditions_file).read_text())
    selected=[r['native_construction']['settings'] for r in q['native_objects']
              if r.get('native_construction',{}).get('kind')=='observed_signal_heads']
    before=native_control_snapshot(unreal)
    records=[]
    for settings in selected:
        prefix='MAP_ObservedSignal_'+settings['signal_id']+'_'
        removed=[]
        for actor in list(level.get_all_level_actors()):
            if actor.get_actor_label().startswith(prefix):
                removed.append(actor.get_actor_label());level.destroy_actor(actor)
        records.append(dict(signal_id=settings['signal_id'],removed=removed,
                            mast=place_mast(unreal,settings)))
    after=native_control_snapshot(unreal)
    if before!=after:
        raise RuntimeError('Support-only update changed native display or control bindings')
    if not level.save_current_level():
        raise RuntimeError('Cannot save source support update')
    Path(output).write_text(json.dumps(dict(map=map_path,conditions=conditions_file,
        operation='supports_only',objects=records,controls_unchanged=True,
        preserved_controls=before,source_phase_programs_modified=False,
        source_traffic_permissions_modified=False,
        scope='Saved inferred supports only; independent reopen and actual source-view visibility remain required.'),
        indent=2)+'\n')


def main():
    import argparse,os,shutil,subprocess
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map',required=True);p.add_argument('--conditions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-map',help='Clone this source level into a new --map before installing controls')
    p.add_argument('--supports-only',action='store_true',help='Replace inferred supports while preserving display/control actors')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if a.source_map:
        content=Path(os.environ['CARLA_ROOT'])/'Unreal/CarlaUE4/Content'
        def navigation(name):
            package=content/name.removeprefix('/Game/')
            return package.parent/'OpenDrive'/(package.name+'.xodr')
        src,dst=navigation(a.source_map),navigation(a.map)
        if dst.exists():raise ValueError('Use a new map navigation destination')
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    script=out/'install_ue.py';report=out/'native_signals.json'
    entry='configure_supports' if a.supports_only else 'configure'
    script.write_text('import sys\nsys.path.insert(0,%r)\nfrom scripts.build.install_observed_signal_heads import %s\n%s(%r,%r,%r,%r)\n'%(
        str(Path.cwd()),entry,entry,a.map,str(a.conditions.resolve()),str(report),a.source_map))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    command=['taskset','-c',os.environ['MAP_CPUSET'],str(Path(os.environ['UE4_ROOT'])/'Engine/Binaries/Linux/UE4Editor'),
        str(Path(os.environ['CARLA_ROOT'])/'Unreal/CarlaUE4/CarlaUE4.uproject'),'-run=pythonscript','-script='+str(script),
        '-nullrhi','-unattended','-nosourcecontrol','-nosound','-nocrashreports']
    with(out/'editor.log').open('w')as log:run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    if not report.exists():raise RuntimeError('Observed signal import failed; inspect editor.log')
    data=json.loads(report.read_text());data['editor_returncode']=run.returncode
    if a.source_map:data['navigation_byte_identical']=src.read_bytes()==dst.read_bytes()
    report.write_text(json.dumps(data,indent=2)+'\n');print(report,flush=True)
    if run.returncode:raise RuntimeError('Unreal exited '+str(run.returncode)+'; saved report requires independent reopen verification')


if __name__=='__main__':main()
