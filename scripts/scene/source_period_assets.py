"""Place removable fixed meshes at source-epoch poses using static collision."""


def spawn_fixed_period_meshes(world, pieces):
    import carla
    actors, records = [], []
    try:
        for piece in pieces:
            blueprint = world.get_blueprint_library().find("static.prop.mesh")
            blueprint.set_attribute("mesh_path", piece["mesh_path"])
            blueprint.set_attribute("mass", "0")
            if piece.get("distance_field_lighting") is False:
                blueprint.set_attribute("distance_field_lighting", "false")
            blueprint.set_attribute("scale", str(piece.get("uniform_scale", 1.0)))
            rotation = piece["rotation_carla_deg"]
            transform = carla.Transform(carla.Location(*piece["position_carla_m"]),
                carla.Rotation(pitch=rotation[0], yaw=rotation[1], roll=rotation[2]))
            actor = world.spawn_actor(blueprint, transform)
            actors.append(actor)
            records.append(dict(source_piece=piece, actor_id=actor.id,
                blueprint=blueprint.id, mass_attribute=0,
                scope="Fixed source-epoch piece at declared pose; remove after use. No moving-replay or road-permission change."))
        return actors, records
    except Exception:
        for actor in reversed(actors):
            actor.destroy()
        raise


def readback_period_meshes(actors, records):
    for actor, record in zip(actors, records):
        transform = actor.get_transform()
        box = actor.bounding_box
        record.update(actual_position_carla_m=[transform.location.x, transform.location.y, transform.location.z],
            actual_rotation_carla_deg=[transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll],
            actual_dimensions_m=[2*box.extent.x, 2*box.extent.y, 2*box.extent.z],
            semantic_tags=list(actor.semantic_tags))
    return records
