"""Convert native GNSS through the calibrated simulator-to-source frame.

The conversion retains raw GNSS samples and reports planar source coordinates.
"""
from dataclasses import dataclass

import numpy as np
from pyproj import Proj

from map_generation.coordinates import LocalFrame


@dataclass
class NativeGnssFrame:
    source_frame: LocalFrame
    native_projection: str
    native_to_local_diagonal: list
    native_altitude_offset_m: float = 0.0

    def __post_init__(self):
        self.projection = Proj(self.native_projection)
        self.diagonal = np.asarray(self.native_to_local_diagonal, dtype=float)

    @classmethod
    def from_config(cls, config):
        return cls(
            LocalFrame(config["city"], config["origin_city_xyz"]),
            config["native_projection"],
            config["native_to_local_diagonal"],
            config.get("native_altitude_offset_m", 0.0),
        )

    def convert(self, latitude, longitude, altitude):
        """Convert a GNSS sample using the fixed source calibration."""
        x, y = self.projection(longitude, latitude)
        native = np.array([x, y, altitude - self.native_altitude_offset_m])
        local = native * self.diagonal
        lon, lat = self.source_frame.local_to_lonlat(local)
        return {
            "raw_native_gnss": {
                "latitude": float(latitude),
                "longitude": float(longitude),
                "altitude": float(altitude),
            },
            "native_xyz_m": native.tolist(),
            "local_xyz_m": local.tolist(),
            "latitude": float(lat),
            "longitude": float(lon),
            "height_local_m": float(local[2]),
            "height_av2_city_m": float(local[2] + self.source_frame.origin[2]),
            "vertical_reference": "AV2 city Z; ellipsoid and sea-level height unknown",
        }
