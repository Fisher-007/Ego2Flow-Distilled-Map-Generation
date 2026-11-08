"""Install source-selected props with configured native meshes, size, and support."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess


def label_for(source_id):
    return 'MAP_SourceProp_' + source_id.replace(':', '_')


def measure(unreal, actor, spec):
    component = actor.get_component_by_class(unreal.StaticMeshComponent)
    mesh = component.get_editor_property('static_mesh')
    scale = actor.get_actor_scale3d()
    center, extent = actor.get_actor_bounds(False)
    bounds = mesh.get_bounds()
    dimensions = [2 * bounds.box_extent.x * scale.x / 100,
                  2 * bounds.box_extent.y * scale.y / 100,
                  2 * bounds.box_extent.z * scale.z / 100]
    actual = [center.x / 100, -center.y / 100, center.z / 100]
    target = spec['center_local_xyz_m']
    error = max(abs(a-b) for a,b in zip(actual,target))
    size_error = max(abs(a-b) for a,b in zip(dimensions,spec['dimensions_lwh_m']))
    collision = str(component.get_collision_profile_name())
    return dict(source_id=spec['id'], actor_label=actor.get_actor_label(),
                asset=mesh.get_path_name(), center_local_xyz_m=actual,
                local_dimensions_lwh_m=dimensions,
                world_aabb_dimensions_m=[2*extent.x/100,2*extent.y/100,2*extent.z/100],
                configured_center_error_m=error, configured_size_error_m=size_error,
                ground_gap_m=actual[2]-dimensions[2]/2-spec['ground_z_m'],
                collision_profile=collision,
                yaw_carla_deg=actor.get_actor_rotation().yaw,
                materials=[str(component.get_material(i).get_path_name())
                           for i in range(component.get_num_materials())
                           if component.get_material(i) is not None],
                passed=error < .001 and size_error < .001
                       and mesh.get_path_name() == spec['asset']
                       and collision == spec.get('collision_profile','BlockAll'),
                source_condition=spec)


def configure(map_path, conditions_file, output):
    import unreal
    level, lib = unreal.EditorLevelLibrary, unreal.EditorAssetLibrary
    if not level.load_level(map_path):
        raise RuntimeError('Cannot open source-prop target map')
    conditions = json.loads(Path(conditions_file).read_text())
    specs = [r['native_construction']['settings'] for r in conditions['native_objects']
             if r['choice'] in ('retain','enhance')
             and r.get('native_construction',{}).get('kind') == 'native_source_prop']
    labels = {label_for(s['id']) for s in specs}
    for actor in list(level.get_all_level_actors()):
        if actor.get_actor_label() in labels:
            level.destroy_actor(actor)
    records = []
    for spec in specs:
        mesh = lib.load_asset(spec['asset'])
        b = mesh.get_bounds()
        dimensions = spec['dimensions_lwh_m']
        scale = [100*dimensions[0]/(2*b.box_extent.x),
                 100*dimensions[1]/(2*b.box_extent.y),
                 100*dimensions[2]/(2*b.box_extent.z)]
        yaw = spec['yaw_carla_deg'];angle=math.radians(yaw)
        x,y,z = spec['center_local_xyz_m']
        offset = unreal.Vector(scale[0]*b.origin.x*math.cos(angle)-scale[1]*b.origin.y*math.sin(angle),
                               scale[0]*b.origin.x*math.sin(angle)+scale[1]*b.origin.y*math.cos(angle),
                               scale[2]*b.origin.z)
        actor = level.spawn_actor_from_object(mesh,unreal.Vector(100*x,-100*y,100*z)-offset,
                                             unreal.Rotator(yaw=yaw))
        actor.set_actor_label(label_for(spec['id']))
        actor.set_actor_scale3d(unreal.Vector(*scale))
        actor.get_component_by_class(unreal.StaticMeshComponent).set_collision_profile_name(
            spec.get('collision_profile','BlockAll'))
        records.append(measure(unreal,actor,spec))
    if not all(r['passed'] for r in records):
        raise RuntimeError('Source prop transform or asset differs from explicit conditions')
    if not level.save_current_level():
        raise RuntimeError('Cannot save source prop map')
    Path(output).write_text(json.dumps(dict(map=map_path,conditions=conditions_file,objects=records,
        passed=True,independent_reopen='pending',native_view='pending',
        scope='Explicit source-epoch props using unchanged existing native meshes and materials. '
              'Dimensions and ground binding are configured observations/approximations; '
              'native semantic class follows the existing asset folder. Road permissions are unchanged.'),
        indent=2)+'\n')


def verify(map_path, report_path, output):
    import unreal
    level=unreal.EditorLevelLibrary
    if not level.load_level(map_path):
        raise RuntimeError('Cannot reopen source-prop target map')
    expected=json.loads(Path(report_path).read_text())
    actors={}
    for actor in level.get_all_level_actors():
        actors.setdefault(actor.get_actor_label(),[]).append(actor)
    rows=[];errors=[]
    for item in expected['objects']:
        found=actors.get(item['actor_label'],[])
        if len(found)!=1:
            errors.append(dict(label=item['actor_label'],actor_count=len(found)))
            continue
        row=measure(unreal,found[0],item['source_condition']);rows.append(row)
        if not row['passed']:
            errors.append(dict(label=item['actor_label'],reason='Saved source-prop binding or transform differs'))
    data=dict(map=map_path,source_report=report_path,objects=rows,errors=errors,
              passed=not errors and len(rows)==len(expected['objects']),native_view='pending')
    Path(output).write_text(json.dumps(data,indent=2)+'\n')
    if not data['passed']:
        raise RuntimeError('Saved source props differ; inspect verification report')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--map',required=True)
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--conditions',type=Path)
    group.add_argument('--verify-report',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    entry='verify' if args.verify_report else 'configure'
    source=args.verify_report or args.conditions
    report=out/('verification.json' if args.verify_report else 'native_props.json')
    script=out/'editor.py'
    script.write_text('import sys\nsys.path.insert(0,%r)\nfrom scripts.build.install_source_props import %s\n%s(%r,%r,%r)\n' % (
        str(Path.cwd()),entry,entry,args.map,str(source.resolve()),str(report)))
    command=['taskset','-c',os.environ['MAP_CPUSET'],
        str(Path(os.environ['UE4_ROOT'])/'Engine/Binaries/Linux/UE4Editor'),
        str(Path(os.environ['CARLA_ROOT'])/'Unreal/CarlaUE4/CarlaUE4.uproject'),
        '-run=pythonscript','-script='+str(script),'-nullrhi','-unattended',
        '-nosourcecontrol','-nosound','-nocrashreports']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    with (out/'editor.log').open('w') as log:
        result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    if not report.exists():
        raise RuntimeError('Native source props did not produce a report; inspect editor.log')
    data=json.loads(report.read_text());data['editor_returncode']=result.returncode
    report.write_text(json.dumps(data,indent=2)+'\n');print(report,flush=True)
    if result.returncode:
        raise RuntimeError('Unreal exited %s; saved report requires independent reopen verification' % result.returncode)


if __name__=='__main__':
    main()
