"""Place native walker poses and approximate body interactions from source conditions.

Uses WalkerBoneControlIn, get_bones, and show_pose with configured local anchors.
"""
import numpy as np


def settle(world):
    if world.get_settings().synchronous_mode:
        world.tick()
    else:
        world.wait_for_tick(60.)


def _align(a, b):
    from scipy.spatial.transform import Rotation
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    a/=np.linalg.norm(a);b/=np.linalg.norm(b)
    axis=np.cross(a,b);s=np.linalg.norm(axis);c=np.clip(a@b,-1.,1.)
    if s<1e-8:
        if c>0:return np.eye(3)
        axis=np.cross(a,np.eye(3)[np.argmin(abs(a))]);axis/=np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi*axis).as_matrix()
    return Rotation.from_rotvec(np.arctan2(s,c)*axis/s).as_matrix()


def _limb(bones, names, target, pole):
    """Two-bone analytic pose with retained bone lengths and local offsets."""
    upper,lower,end=[bones[n]for n in names]
    c0,c1,c2=[np.asarray(x.component.get_matrix())for x in [upper,lower,end]]
    l0,l1,l2=[np.asarray(x.relative.get_matrix())for x in [upper,lower,end]]
    p0,p1,p2=[m[:3,3]for m in [c0,c1,c2]]
    a=np.linalg.norm(p1-p0);b=np.linalg.norm(p2-p1)
    delta=np.asarray(target)-p0;requested=np.linalg.norm(delta);direction=delta/requested
    distance=np.clip(requested,abs(a-b)+1e-5,a+b-1e-5)
    projected=np.asarray(pole,dtype=float)-direction*np.dot(pole,direction)
    projected/=np.linalg.norm(projected)
    along=(a*a-b*b+distance*distance)/(2*distance)
    joint=p0+along*direction+np.sqrt(max(0.,a*a-along*along))*projected
    endpoint=p0+distance*direction
    q0=_align(p1-p0,joint-p0);q1=_align(q0@(p2-p1),endpoint-joint)
    r0=q0@c0[:3,:3];r1=q1@q0@c1[:3,:3]
    parent=c0[:3,:3]@l0[:3,:3].T
    l0[:3,:3]=parent.T@r0;l1[:3,:3]=r0.T@r1
    # Preserve the reference foot/hand orientation; no finger articulation.
    l2[:3,:3]=r1.T@c2[:3,:3]
    return list(zip(names,[l0,l1,l2])),dict(bones=names,requested_component_xyz_m=list(target),reachable_component_xyz_m=endpoint.tolist(),reach_clamped_m=float(abs(requested-distance)))


def _box_matrix(actor):
    import carla
    return np.asarray(actor.get_transform().get_matrix())@np.asarray(carla.Transform(actor.bounding_box.location,actor.bounding_box.rotation).get_matrix())


def apply_interactions(world, actors, records, conditions):
    """Apply named native walker roles after ordinary source-ground fitting."""
    import carla
    from scripts.scene.render_dynamic_av2_view import transform
    by_actor={a.id:a for a in actors};by_record={r['track_uuid']:r for r in records}
    selected=[c for c in conditions if c['track_uuid']in by_record]
    if not selected:return []
    settle(world);results=[]
    for condition in selected:
        record=by_record[condition['track_uuid']];actor=by_actor[record['actor_id']]
        if not isinstance(actor,carla.Walker):raise TypeError('Interaction role requires native Walker')
        initial=np.asarray(actor.get_transform().get_matrix());actor.get_pose_from_animation();actor.show_pose();settle(world)
        parent=by_actor[by_record[condition['parent_track_uuid']]['actor_id']] if condition.get('parent_track_uuid') else None
        parent_matrix=_box_matrix(parent) if parent else None
        bones={b.name:b for b in actor.get_bones().bone_transforms}
        if condition.get('pelvis_parent_xyz_m') is not None:
            desired=parent_matrix@np.r_[condition['pelvis_parent_xyz_m'],1.]
            pelvis=np.asarray(bones['crl_hips__C'].component.get_matrix())[:3,3]
            pose=initial.copy();pose[:3,:3]=parent_matrix[:3,:3];pose[:3,3]=desired[:3]-pose[:3,:3]@pelvis
            actor.set_transform(transform(pose));settle(world)
            bones={b.name:b for b in actor.get_bones().bone_transforms}
        actor_matrix=np.asarray(actor.get_transform().get_matrix());inverse=np.linalg.inv(actor_matrix)
        changes=[];limbs=[]
        for item in condition.get('limbs',[]):
            if 'parent_xyz_m'in item:target=(inverse@parent_matrix@np.r_[item['parent_xyz_m'],1.])[:3]
            else:
                hip=np.asarray(bones['crl_hips__C'].component.get_matrix())[:3,3]
                target=hip+np.asarray(item['pelvis_delta_xyz_m'])
            transforms,measurement=_limb(bones,item['bones'],target,item['pole_component']);changes.extend(transforms);limbs.append(measurement)
        if changes:
            actor.set_bones(carla.WalkerBoneControlIn([(name,transform(matrix))for name,matrix in changes]));actor.show_pose();settle(world)
        actual={b.name:b for b in actor.get_bones().bone_transforms}
        for item in limbs:
            endpoint=np.asarray(actual[item['bones'][-1]].component.get_matrix())[:3,3]
            item['actual_component_xyz_m']=endpoint.tolist();item['reachable_target_error_m']=float(np.linalg.norm(endpoint-item['reachable_component_xyz_m']))
        final=np.asarray(actor.get_transform().get_matrix());box=_box_matrix(actor)
        record['render_box_pose']=dict(center_xyz_m=box[:3,3].tolist(),rotation_matrix=box[:3,:3].tolist())
        result=dict(track_uuid=condition['track_uuid'],role=condition['interaction_role'],blueprint=actor.type_id,condition=condition,initial_actor_matrix=initial.tolist(),actual_actor_matrix=final.tolist(),display_translation_m=(final[:3,3]-initial[:3,3]).tolist(),limbs=limbs,bones={name:dict(component=b.component.get_matrix(),world=b.world.get_matrix())for name,b in actual.items()if name in ['crl_hips__C','crl_Head__C','crl_hand__L','crl_hand__R','crl_foot__L','crl_foot__R']},scope='Coarse source-supported role and occupancy pose, original AV2 state retained; not continuous control or fine contact validation.')
        record['source_interaction']=result;results.append(result)
    return results


def readback_interactions(actors, records):
    by_id={a.id:a for a in actors}
    for record in records:
        if 'source_interaction'not in record:continue
        actor=by_id[record['actor_id']]
        bones={b.name:b for b in actor.get_bones().bone_transforms}
        record['source_interaction']['after_capture_bones']={name:dict(component=b.component.get_matrix(),world=b.world.get_matrix())for name,b in bones.items()if name in record['source_interaction']['bones']}


def spawn_controllable_walkers(world, snapshot, surface, native_blueprints):
    """Spawn selected pedestrian tracks for caller-managed control and cleanup."""
    from copy import deepcopy
    import carla
    from scripts.scene.render_dynamic_av2_view import spawn_snapshot
    selected=deepcopy(snapshot)
    selected['actors']=[r for r in selected['actors']if r['track_uuid']in native_blueprints and r['category']=='PEDESTRIAN']
    actors,records,skipped=spawn_snapshot(world,selected,front_only=False,native_blueprints=native_blueprints)
    sources={r['track_uuid']:r for r in selected['actors']};by_id={a.id:a for a in actors}
    try:
        for record in records:
            actor=by_id[record['actor_id']]
            pose,box,fit=surface.fit(sources[record['track_uuid']],actor)
            actor.set_transform(pose);actor.hide_pose();actor.set_enable_gravity(True);actor.set_simulate_physics(True)
            actor.apply_control(carla.WalkerControl(speed=0.))
            record.update(render_box_pose=box,asset_support_fit=fit,mode='native_walker_controller',source_motion_retained=True,
                scope='Native pedestrian initialized on retained source support; controlled motion is separate from original AV2 state, coupled interactions and whole-city traffic.')
        return actors,records,skipped
    except Exception:
        for actor in reversed(actors):actor.destroy()
        raise
