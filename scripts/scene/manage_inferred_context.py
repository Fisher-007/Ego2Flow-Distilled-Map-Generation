"""Initialize and manage inferred parked vehicles as native CARLA actors.

Vehicles settle on support surfaces, then hold their poses until selected
for release or removal.
"""
import argparse
import json
from pathlib import Path
import math
import carla


def role_prefix(registry):
    return registry['role_name_prefix'] + ':'


def owned(world, registry):
    prefix = role_prefix(registry)
    return {a.attributes.get('role_name', '')[len(prefix):]: a
            for a in world.get_actors().filter('vehicle.*')
            if a.attributes.get('role_name', '').startswith(prefix)}


def readback(actor):
    t, v, b, c = actor.get_transform(), actor.get_velocity(), actor.bounding_box, actor.get_control()
    return dict(actor_id=actor.id, blueprint=actor.type_id,
        role_name=actor.attributes.get('role_name'), color=actor.attributes.get('color'),
        position_carla_m=[t.location.x,t.location.y,t.location.z],
        rotation_pyr_deg=[t.rotation.pitch,t.rotation.yaw,t.rotation.roll],
        velocity_mps=[v.x,v.y,v.z], bounding_box=dict(location=[b.location.x,b.location.y,b.location.z],extent=[b.extent.x,b.extent.y,b.extent.z]),
        bounding_vertices_carla_m=[[p.x,p.y,p.z] for p in b.get_world_vertices(t)],
        control=dict(throttle=c.throttle,brake=c.brake,hand_brake=c.hand_brake,steer=c.steer))


def spawn_context(world, registry):
    if world.get_map().name.split('/')[-1] != registry['map']:
        raise ValueError('Registry belongs to a different native map')
    previous = owned(world,registry)
    if previous:
        raise ValueError('Context already present; read status or explicitly remove before respawning')
    old=world.get_settings();settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05;world.apply_settings(settings)
    records=[];spawned={};failed=[];library=world.get_blueprint_library()
    try:
        for entry in registry['vehicles']:
            bp=library.find(entry['blueprint']);bp.set_attribute('role_name',role_prefix(registry)+entry['id'])
            if bp.has_attribute('color'):bp.set_attribute('color',entry['color'])
            x,y,z=entry['position_enu_m'];target=carla.Location(x,-y,z)
            hits=world.cast_ray(carla.Location(x,-y,z+3),carla.Location(x,-y,z-3))
            surfaces=[h for h in hits if str(h.label).split('.')[-1] in ['Roads','Sidewalks','Terrain','Ground']]
            if not surfaces:
                failed.append(dict(id=entry['id'],reason='No native ground support at proposed mapped-parking point',hits=[dict(label=str(h.label),z=h.location.z)for h in hits]));continue
            hit=min(surfaces,key=lambda h:abs(h.location.z-z));support_z=hit.location.z
            t=carla.Transform(carla.Location(x,-y,support_z+.6),carla.Rotation(yaw=-entry['yaw_enu_degrees']))
            actor=world.try_spawn_actor(bp,t)
            if actor is None:
                failed.append(dict(id=entry['id'],reason='Native collision prevented parking initialization'));continue
            actor.apply_control(carla.VehicleControl(brake=1.,hand_brake=True));spawned[entry['id']]=actor
            records.append(dict(id=entry['id'],inferred=True,source_entry=entry,native_support=dict(label=str(hit.label),position_carla_m=[hit.location.x,hit.location.y,hit.location.z]),spawn_transform=readback(actor)))
        for _ in range(100):world.tick()
        for row in records:
            actor=spawned[row['id']];row['settled_before_hold']=readback(actor);actor.set_simulate_physics(False);actor.apply_control(carla.VehicleControl(brake=1.,hand_brake=True))
        world.tick()
        for row in records:row['held']=readback(spawned[row['id']])
    finally:
        world.apply_settings(old)
    return dict(map=world.get_map().name,expected=len(registry['vehicles']),spawned=len(records),failed=failed,vehicles=records,
        mode='Native CARLA vehicles; brake and hand brake during five-second support settling, then physics held. Explicit release enables normal VehicleControl.',
        source_identity_scope='All IDs in this registry are inferred context; original source actors and trajectories remain separate.',settings_restored=True)


def release_context(world,registry,identities):
    actors=owned(world,registry);rows=[]
    for identity in identities:
        actor=actors[identity];actor.set_simulate_physics(True);actor.apply_control(carla.VehicleControl(brake=1.,hand_brake=False))
        rows.append(dict(id=identity,actor_id=actor.id,physics_enabled=True,initial_brake=1.,driving_permissions='Caller retains the existing road/parking permissions; no route or priority assigned here.'))
    return rows


def remove_context(world,registry):
    rows=[]
    for identity,actor in owned(world,registry).items():
        rows.append(dict(id=identity,actor_id=actor.id,destroyed=actor.destroy()))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registry',type=Path,required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--host',default='127.0.0.1');p.add_argument('--action',choices=['spawn','status','release','remove'],required=True);p.add_argument('--ids',nargs='*',default=[]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    registry=json.loads(a.registry.read_text());client=carla.Client(a.host,a.port);client.set_timeout(45);world=client.get_world()
    if a.action=='spawn':result=spawn_context(world,registry)
    elif a.action=='release':result=dict(released=release_context(world,registry,a.ids))
    elif a.action=='remove':result=dict(removed=remove_context(world,registry))
    else:result=dict(vehicles={k:readback(v)for k,v in owned(world,registry).items()})
    result.update(registry=str(a.registry),action=a.action);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k not in ['vehicles']}),flush=True)
    if a.action=='spawn' and result['failed']:raise RuntimeError('Some proposed inferred parking positions failed native initialization; inspect actual entries and revise conditions.')


if __name__=='__main__':main()
