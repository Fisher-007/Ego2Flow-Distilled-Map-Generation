"""Add source-conditioned facade components to a solid mesh.

Inputs specify world-space facade planes, floor levels, and component intervals.
"""
import math


def append_source_facades(vertices, faces, materials, facades, slots):
    counts = {"facades": 0, "balconies": 0, "windows": 0, "boxes": 0}

    def box(origin, along, outward, u0, u1, v0, v1, z0, z1, material):
        if u1 <= u0 or v1 <= v0 or z1 <= z0:
            raise ValueError("Source facade boxes require positive dimensions")
        offset = len(vertices)
        for z in (z0, z1):
            for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
                vertices.append([origin[0] + along[0]*u + outward[0]*v,
                                 origin[1] + along[1]*u + outward[1]*v, z])
        local = [(3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4),
                 (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        if along[0]*outward[1] - along[1]*outward[0] < 0:
            local = [tuple(reversed(f)) for f in local]
        faces.extend([[offset+i for i in f] for f in local])
        materials.extend([slots[material]]*6)
        counts["boxes"] += 1

    for facade in facades:
        p0, p1 = facade["start_xy_m"], facade["end_xy_m"]
        length = math.hypot(p1[0]-p0[0], p1[1]-p0[1])
        along = [(p1[i]-p0[i])/length for i in range(2)]
        outward = facade["outward_xy"]
        norm = math.hypot(*outward)
        outward = [x/norm for x in outward]
        def component(u0, u1, v0, v1, z0, z1, material="trim"):
            box(p0, along, outward, u0, u1, v0, v1, z0, z1, material)
        bottom, top = facade["z_range_m"]
        for pier in facade.get("piers", []):
            component(pier["u_m"]-pier["width_m"]/2,
                      pier["u_m"]+pier["width_m"]/2, .01, .12, bottom, top)
        for band in facade.get("bands", []):
            component(0, length, .01, .13, band["z_m"],
                      band["z_m"]+band["height_m"])
        for window in facade.get("windows", []):
            u0, u1 = window["u_range_m"]; z0, z1 = window["z_range_m"]
            component(u0-.055, u1+.055, .012, .045, z0-.055, z1+.055)
            component(u0, u1, .046, .06, z0, z1, "glass")
            component((u0+u1)/2-.025, (u0+u1)/2+.025, .061, .075, z0, z1)
            counts["windows"] += 1
        for balcony in facade.get("balconies", []):
            u0, u1 = balcony["u_range_m"]; z = balcony["floor_z_m"]
            depth = balcony["depth_m"]; thickness = balcony.get("slab_thickness_m", .18)
            rail_height = balcony.get("rail_height_m", 1.05)
            component(u0, u1, .015, depth, z-thickness, z)
            # Thin rail members preserve the visible openings and occlusion.
            component(u0, u1, depth-.04, depth, z+rail_height-.035, z+rail_height)
            for u in (u0, u1):
                component(u, u+.035, .04, depth, z+rail_height-.035, z+rail_height)
            divisions = max(1, math.ceil((u1-u0)/balcony.get("rail_spacing_m", .38)))
            for i in range(divisions+1):
                u = u0+(u1-u0)*i/divisions
                component(u-.015, u+.015, depth-.04, depth, z, z+rail_height)
            counts["balconies"] += 1
        counts["facades"] += 1
    return counts
