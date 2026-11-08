"""Local UTM coordinates shared by source maps, Blender and CARLA."""

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer
from av2.geometry.utm import CityName, UTM_ZONE_MAP, convert_city_coords_to_utm


@dataclass
class LocalFrame:
    city: str
    origin_city_xyz: list

    def __post_init__(self):
        self.origin = np.asarray(self.origin_city_xyz, dtype=float)
        self.epsg = 32600 + UTM_ZONE_MAP[CityName(self.city)]
        self.origin_utm = convert_city_coords_to_utm(self.origin[None, :2], CityName(self.city))[0]
        self.project = Transformer.from_crs(4326, self.epsg, always_xy=True)
        self.unproject = Transformer.from_crs(self.epsg, 4326, always_xy=True)

    def city_to_local(self, points):
        points = np.asarray(points, dtype=float)
        return points - self.origin[:points.shape[-1]]

    def local_to_city(self, points):
        points = np.asarray(points, dtype=float)
        return points + self.origin[:points.shape[-1]]

    def lonlat_to_local(self, lonlat):
        lonlat = np.asarray(lonlat, dtype=float)
        return np.stack(self.project.transform(lonlat[..., 0], lonlat[..., 1]), axis=-1) - self.origin_utm

    def local_to_lonlat(self, points):
        utm = np.asarray(points, dtype=float)[..., :2] + self.origin_utm
        return np.stack(self.unproject.transform(utm[..., 0], utm[..., 1]), axis=-1)

    def sumo_to_local_xy(self, points, net_offset):
        return np.asarray(points, dtype=float) - np.asarray(net_offset) - self.origin_utm

    def metadata(self):
        return {
            "city": self.city, "units": "m", "axes": ["grid_east", "grid_north", "up"],
            "handedness": "right", "horizontal_crs": f"EPSG:{self.epsg}",
            "origin_city_xyz": self.origin.tolist(), "origin_utm_xy": self.origin_utm.tolist(),
            "origin_lonlat": self.local_to_lonlat([0, 0]).tolist(),
            "vertical_reference": "AV2 city z minus origin z; no sea-level or ellipsoid conversion",
            "blender": {"axes": ["grid_east", "grid_north", "up"], "units": "m"},
            "carla": {"axes": ["grid_east", "grid_south", "up"], "units": "m",
                      "from_local_diagonal": [1, -1, 1], "yaw": "negative local heading in degrees"},
            "unreal": {"units": "cm", "from_local_diagonal": [100, -100, 100]},
            "geolocation": "Use local_to_lonlat (pyproj); CARLA built-in GNSS is not aligned to this UTM frame.",
        }


def local_to_carla(points):
    points = np.asarray(points, dtype=float)
    return points * np.array([1, -1, 1])[:points.shape[-1]]


def carla_to_local(points):
    return local_to_carla(points)


def heading_to_carla(heading_radians):
    return -np.degrees(heading_radians)


def register_local_pose(position, rotation, registration):
    """Apply one log-to-map rigid transform equally to cameras and objects."""
    basis = np.asarray(registration["rotation_matrix"])
    return (basis @ np.asarray(position) + registration["translation_local_m"],
            basis @ np.asarray(rotation))


def source_log_registration(scene, log_id):
    """Use the same reviewed log frame for roads, camera poses and actors."""
    rows=scene.get('source_info',{}).get('assembly_review',{}).get('log_registrations',[])
    row=next((r for r in rows if r['log_id']==log_id),None)
    if row is None:return None
    registration=row['registration'];basis=np.asarray(registration['rotation_matrix'])
    origin=np.asarray(scene['coordinates']['origin_city_xyz'])
    translation=basis@origin+np.asarray(registration['translation_city_m'])-origin
    return dict(log_id=log_id,map_id=scene['map_id'],reference_log=registration.get('reference_log'),
        registration_evidence=dict(registration),
        rotation_matrix=basis.tolist(),translation_local_m=translation.tolist(),
        source='Selected city source assembly; identical rigid transform for geometry, cameras and objects.')


def paired_observation_context(scene, log_id, city, city_xyz):
    """Resolve a source log in the declared city frame and XY coverage, retaining measured Z."""
    from pathlib import Path

    if city != scene["coordinates"]["city"]:
        raise ValueError("Observation and SceneIR must use the same AV2 city frame")
    local = np.asarray(city_xyz) - np.asarray(scene["coordinates"]["origin_city_xyz"])
    areas = [area["id"] for area in scene["areas"]
             if area["bbox_local_xy"][0] <= local[0] <= area["bbox_local_xy"][2]
             and area["bbox_local_xy"][1] <= local[1] <= area["bbox_local_xy"][3]]
    region = scene.get('map_region_bbox_local_xy')
    in_region = bool(region is not None and region[0] <= local[0] <= region[2]
                     and region[1] <= local[1] <= region[3])
    influence = [area['id'] for area in scene['areas']
                 if 'influence_bbox_local_xy' in area
                 and area['influence_bbox_local_xy'][0] <= local[0] <= area['influence_bbox_local_xy'][2]
                 and area['influence_bbox_local_xy'][1] <= local[1] <= area['influence_bbox_local_xy'][3]]
    if not areas and not in_region:
        raise ValueError("Observation ego position is outside the declared map region and Areas")
    source_map = scene["source_info"]["av2_map"]
    source_maps = scene["source_info"].get("av2_maps", [source_map])
    source_logs = sorted({Path(path).parts[-3] for path in source_maps})
    source_log = source_logs[0] if len(source_logs) == 1 else None
    return {"observation_log_id": log_id, "map_source_log_id": source_log,
            "map_source_av2_map": source_map, "map_source_log_ids": source_logs,
            "is_map_source_log": log_id in source_logs,
            "area_ids": areas, "influence_area_ids": influence,
            "inside_declared_map_region": in_region, "ego_local_xyz_m": local.tolist(),
            "registration": "Official shared city frame minus the existing map origin; no fitted XY or Z shift.",
            "scope": "Declared map/Area capture eligibility; core and influence membership recorded separately. Does not verify cross-log map geometry, road height or signal state."}
