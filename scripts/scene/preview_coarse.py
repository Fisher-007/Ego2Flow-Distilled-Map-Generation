"""Create a CPU Blender preview of SceneIR's coarse roads, terrain and buildings.

Run with Blender --background --python-exit-code 1 --python this_file --
--scene scene.json --output-dir preview. Geometry stays in local ENU meters.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import textwrap


def mesh_object(name, vertices, faces, material, collection):
    import bpy

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def ring(points, ccw=True):
    """Normalize ring closure/winding for Blender's polygon tessellator."""
    points = list(points)
    if points[0] == points[-1]:
        points = points[:-1]
    area = sum(a[0] * b[1] - b[0] * a[1]
               for a, b in zip(points, points[1:] + points[:1]))
    return points if (area > 0) == ccw else points[::-1]


def polygon_mesh(rings):
    from mathutils.geometry import tessellate_polygon

    # Blender tessellation returns indices into the flattened rings.
    return [tuple(p) for loop in rings for p in loop], tessellate_polygon(rings)


def road_strip(lane, material, collection):
    from mathutils import Vector

    points = [Vector(p) for p in lane["centerline"]]
    vertices = []
    for i, point in enumerate(points):
        tangent = points[min(i + 1, len(points) - 1)] - points[max(i - 1, 0)]
        side = Vector((-tangent.y, tangent.x, 0)).normalized() * lane["width_m"] / 2
        # A small display lift avoids coplanar flicker; it is recorded below.
        lift = Vector((0, 0, 0.04))
        left, right = point + side, point - side
        vertices.extend(left.lerp(right, sample / 4) + lift for sample in range(5))
    faces = [(5*i+j, 5*i+j+1, 5*(i+1)+j+1, 5*(i+1)+j)
             for i in range(len(points) - 1) for j in range(4)]
    return mesh_object("lane_" + str(lane["id"]), vertices, faces, material, collection)


def drape_preview(obj, terrain_tree, ray_z, offset):
    """Blender-only display surface; SceneIR's sampled centerline is unchanged."""
    from mathutils import Vector

    down = Vector((0, 0, -1))
    for vertex in obj.data.vertices:
        hit, _, _, _ = terrain_tree.ray_cast(Vector((vertex.co.x, vertex.co.y, ray_z)), down)
        if hit is None:
            raise ValueError("Preview surface extends beyond terrain: %s" % obj.name)
        vertex.co.z = hit.z + offset
    obj.data.update()


def building_mesh(building, material, collection):
    base, height = building["base_z_m"], building["height_m"]
    loops = [ring(building["exterior"])]
    loops.extend(ring(hole, ccw=False) for hole in building.get("holes", []))
    vertices, roof = polygon_mesh([[(x, y, base + height) for x, y in loop] for loop in loops])
    count = len(vertices)
    vertices += [(x, y, base) for x, y, _ in vertices]
    faces = list(roof) + [tuple(i + count for i in face[::-1]) for face in roof]
    offset = 0
    for loop in loops:
        for i in range(len(loop)):
            a, b = offset + i, offset + (i + 1) % len(loop)
            faces.append((a, a + count, b + count, b))
        offset += len(loop)
    # Replace OSM namespace separators in export names and retain source IDs in metadata.
    name = "building_" + re.sub(r'[^A-Za-z0-9_]', '_', str(building['id']))
    obj = mesh_object(name, vertices, faces, material, collection)
    obj['source_building_id'] = str(building['id'])
    obj["height_source"] = str(building.get("height_source", "not supplied"))
    obj["ground_source"] = str(building.get("ground_source", building.get("base_z_source", "not supplied")))
    for key in ("base_z_m", "height_m", "nominal_height_m", "foundation_relief_m", "roof_z_m"):
        if key in building:
            obj[key] = building[key]
    return obj


def material(name, color, emission=False):
    import bpy

    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    if emission:
        nodes.clear()
        shader = nodes.new("ShaderNodeEmission")
        shader.inputs["Color"].default_value = (*color, 1)
        mat.node_tree.links.new(shader.outputs[0], nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
    else:
        shader = nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = (*color, 1)
        shader.inputs["Roughness"].default_value = 1
    return mat


def labels(camera, lines, collection, ink):
    import bpy
    from mathutils import Vector

    width = camera.data.ortho_scale
    scale = width / (4 / 3)
    rotation = camera.rotation_euler.to_quaternion()
    for i, (text, size, y) in enumerate(lines):
        curve = bpy.data.curves.new("annotation_%d" % i, "FONT")
        curve.body, curve.size = text, scale * size
        curve.space_line = 1.15
        obj = bpy.data.objects.new(curve.name, curve)
        collection.objects.link(obj)
        obj.rotation_euler = camera.rotation_euler
        obj.location = camera.location + rotation @ Vector((-0.46 * width, y * scale, -10))
        obj.data.materials.append(ink)
        obj.visible_shadow = obj.visible_diffuse = obj.visible_glossy = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    source_path, output = args.scene.resolve(), args.output_dir.resolve()
    source = json.loads(source_path.read_text())
    coarse, terrain, buildings = source["coarse_map"], source["terrain"], source["buildings"]
    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system, scene.unit_settings.scale_length = "METRIC", 1
    geometry = bpy.data.collections.new("Coarse geometry in ENU meters")
    scene.collection.children.link(geometry)
    annotations = bpy.data.collections.new("Preview annotations")
    scene.collection.children.link(annotations)
    colors = {
        "av2": (0.13, 0.39, 0.59), "osm": (0.80, 0.37, 0.12),
        "unknown": (0.36, 0.38, 0.40), "junction": (0.76, 0.64, 0.25),
        "terrain": (0.50, 0.57, 0.43), "building": (0.67, 0.56, 0.44),
    }
    materials = {key: material(key, color) for key, color in colors.items()}
    mesh_object("terrain", terrain["vertices"], terrain["faces"], materials["terrain"], geometry)
    terrain_tree = BVHTree.FromPolygons(terrain["vertices"], terrain["faces"], all_triangles=True)
    ray_z = max(point[2] for point in terrain["vertices"]) + 100
    road_colors = Counter()
    for lane in coarse["lanes"]:
        provenance = str(lane.get("source", coarse.get("source", "not supplied"))).lower()
        color = "av2" if "av2" in provenance else "osm" if "osm" in provenance else "unknown"
        obj = road_strip(lane, materials[color], geometry)
        obj["source"] = provenance
        road_colors[color] += 1
        drape_preview(obj, terrain_tree, ray_z, .04)
    junctions_without_area = []
    for junction in coarse["junctions"]:
        boundary = [[p[0], p[1], p[2] + 0.06] for p in ring(junction["boundary"])]
        # SUMO can represent nodes with a two-point shape, which has no fill area.
        if len(boundary) < 3:
            junctions_without_area.append(junction["id"])
            continue
        vertices, faces = polygon_mesh([boundary])
        obj = mesh_object("junction_" + str(junction["id"]), vertices, faces, materials["junction"], geometry)
        drape_preview(obj, terrain_tree, ray_z, .06)
    for building in buildings:
        building_mesh(building, materials["building"], geometry)

    points = [Vector(v.co) for obj in geometry.objects for v in obj.data.vertices]
    low = Vector([min(p[i] for p in points) for i in range(3)])
    high = Vector([max(p[i] for p in points) for i in range(3)])
    center = (low + high) / 2
    span = max(high.x - low.x, high.y - low.y, high.z - low.z)
    camera_data = bpy.data.cameras.new("overview")
    camera = bpy.data.objects.new("overview", camera_data)
    scene.collection.objects.link(camera)
    camera.location = center + Vector((0, -0.7 * span, 1.4 * span))
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    inverse = camera.rotation_euler.to_matrix().transposed()
    projected = [inverse @ (p - center) for p in points]
    width = max(p.x for p in projected) - min(p.x for p in projected)
    height = max(p.y for p in projected) - min(p.y for p in projected)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(width, height * (4 / 3)) * 1.42
    offset = Vector(((max(p.x for p in projected) + min(p.x for p in projected)) / 2,
                     (max(p.y for p in projected) + min(p.y for p in projected)) / 2, 0))
    camera.location += inverse.transposed() @ offset
    camera_data.clip_end = 10 * span + 100
    scene.camera = camera

    road_sources = Counter(str(lane.get("source", coarse.get("source", "not supplied"))) for lane in coarse["lanes"])
    height_sources = Counter(str(b.get("height_source", "not supplied")) for b in buildings)
    assumptions = {
        "coarse_map": coarse.get("assumptions", "not supplied"),
        "terrain": terrain.get("assumptions", "not supplied"),
        "preview": ["Road ribbons use centerline width, not final engineered boundaries.",
                    "Roads use five samples across each supplied centerline cross-section; junctions use their supplied polygon vertices.",
                    "Blender BVHTree vertically projects those preview vertices onto the same terrain, with road/junction display offsets of 0.04/0.06 m.",
                    "SUMO junction shapes with fewer than three vertices have no fill area and are omitted from the preview; their lanes remain.",
                    "SceneIR centerlines retain their original DEM height. Road surfaces and terrain seams require native contact checks.",
                    "Building base is sampled minimum ground; roof is sampled maximum ground plus nominal_height_m.",
                    "Nominal height comes from OSM tags or an explicit assumption; height_m also includes foundation_relief_m.",
                    "The added foundation relief is geometry, not a measured building height. Roofs are flat.",
                    "No height exaggeration. This preview does not validate collision or drivability."],
    }
    terrain_source = terrain.get("source", "not supplied")
    if isinstance(terrain_source, dict):
        dataset = terrain_source.get("metadata_details", {}).get("dataset", {})
        terrain_label = " | ".join(str(value) for value in (
            terrain_source.get("provider", "provider not supplied"),
            dataset.get("Name", dataset.get("title", "dataset not supplied")),
        ))
    else:
        terrain_label = str(terrain_source)
    source_labels = {"av2": "Blue: AV2 lanes", "osm": "Orange: OSM-derived lanes", "unknown": "Gray: source unknown"}
    if any("sumo" in key.lower() for key in road_sources):
        source_labels["osm"] = "Orange: OSM / SUMO inferred lanes"
    road_legend = " | ".join("%s (%d)" % (source_labels[key], count) for key, count in road_colors.items())
    road_legend += " | Yellow: junction areas (%d)" % (len(coarse["junctions"]) - len(junctions_without_area))
    text = [
        (str(source.get("map_id", source_path.stem)) + " | coarse geometry preview", .023, .45),
        (road_legend, .014, .408),
        ("Green: terrain | Tan: buildings | View from south; east right / north up; meters", .014, .38),
        ("Terrain: " + textwrap.shorten(terrain_label, width=130, placeholder="..."), .012, -.385),
        ("Building height sources: " + textwrap.shorten("; ".join(height_sources) or "no buildings", width=118, placeholder="..."), .012, -.409),
        ("Building base = minimum ground; roof = maximum ground + nominal height. Relief is added foundation geometry.", .012, -.433),
        ("Preview-only terrain drape; roads +4 cm / junctions +6 cm. SceneIR unchanged; no height exaggeration.", .012, -.457),
        ("Geometry inspection only; collision, driving and final surface construction are not checked.", .012, -.481),
    ]
    labels(camera, text, annotations, material("annotation ink", (.07, .09, .11), emission=True))
    scene.world = bpy.data.worlds.new("plain studio")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (.82, .86, .90, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = .8
    sun_data = bpy.data.lights.new("sun", "SUN")
    sun_data.energy, sun_data.angle = 1.5, 0.15
    sun = bpy.data.objects.new("sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (.35, -.45, -.3)
    scene.render.engine = "CYCLES"
    scene.cycles.device, scene.cycles.samples = "CPU", 16
    scene.cycles.use_denoising = True
    scene.render.threads_mode, scene.render.threads = "FIXED", 6
    scene.render.resolution_x, scene.render.resolution_y = 1600, 1200
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output / "overview.png")
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.render.film_transparent = False
    report = {
        "scene": str(source_path), "blender_version": bpy.app.version_string,
        "renderer": "Cycles CPU, 16 samples", "road_sources": dict(road_sources),
        "terrain_source": terrain_source, "terrain_label": terrain_label,
        "building_height_sources": dict(height_sources),
        "building_geometry": {
            "base": "minimum sampled ground",
            "nominal_height_m": "OSM tag or declared assumption",
            "height_m": "nominal_height_m + foundation_relief_m",
            "roof_z_m": "maximum sampled ground + nominal_height_m",
        },
        "assumptions": assumptions,
        "counts": {"lanes": len(coarse["lanes"]), "junctions": len(coarse["junctions"]),
                   "junction_surfaces": len(coarse["junctions"]) - len(junctions_without_area),
                   "buildings": len(buildings), "buildings_with_holes": sum(bool(b.get("holes")) for b in buildings),
                   "terrain_vertices": len(terrain["vertices"]), "terrain_faces": len(terrain["faces"])},
        "bounds_m": {"min": list(low), "max": list(high)},
        "junctions_without_area": junctions_without_area,
        "blend": str(output / "coarse.blend"), "png": scene.render.filepath,
    }
    notes = bpy.data.texts.new("SOURCES_AND_ASSUMPTIONS.json")
    notes.write(json.dumps(report, indent=2) + "\n")
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "coarse.blend"))
    bpy.ops.render.render(write_still=True)
    (output / "preview.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Coarse preview saved: %s" % output)


if __name__ == "__main__":
    main()
