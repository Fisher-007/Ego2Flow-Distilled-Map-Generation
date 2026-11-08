"""Sample a USGS DEM aligned to AV2 lane elevations."""

import json
from pathlib import Path

import numpy as np
import rasterio
from scipy.interpolate import RegularGridInterpolator
from av2.map.map_api import ArgoverseStaticMap

from .coordinates import LocalFrame


def residual_summary(values):
    values = np.asarray(values)
    return {"samples": int(values.size), "rmse_m": float(np.sqrt(np.mean(values**2))),
            "median_m": float(np.median(values)),
            "absolute_p95_m": float(np.percentile(np.abs(values), 95)),
            "range_m": [float(values.min()), float(values.max())]}


class Terrain:
    """Query a DEM in local meters, returning NaN for unavailable heights.

    ``bounds`` is local [xmin, ymin, xmax, ymax]. The input GeoTIFF has a
    same-stem JSON containing ``extent_utm_xy``, ``epsg`` and source metadata.
    """

    def __init__(self, scene, config, data_root, bounds=None, spacing=None):
        self.frame = LocalFrame(config["city"], config["coordinates"]["origin_city_xyz"])
        coarse = config["coarse"]
        self.spacing = spacing if spacing is not None else coarse["terrain_spacing_m"]
        if bounds is None:
            bounds = np.asarray(scene.get("map_region_bbox_local_xy",scene["areas"][0]["bbox_local_xy"])) + np.array([-1, -1, 1, 1])*coarse["buffer_m"]
        self.bounds = np.asarray(bounds, dtype=float)
        path = Path(data_root) / config["sources"]["terrain_dem"]
        self.metadata = json.loads(path.with_suffix(".json").read_text())
        if self.metadata["epsg"] != self.frame.epsg:
            raise ValueError("DEM must use the configured UTM CRS")
        # GDAL recognizes sparse TIFF tile masks; Pillow can interpret a zero
        # tile offset as TIFF-header floats rather than unavailable elevation.
        with rasterio.open(path) as dataset:
            raster = dataset.read(1, masked=True).filled(np.nan).astype(float)
        self.allow_exterior_nodata = coarse.get("allow_missing_terrain_outside_map", False)
        self.protected_bounds = np.asarray(
            scene.get("map_region_bbox_local_xy", scene["areas"][0]["bbox_local_xy"]))
        xmin, ymin, xmax, ymax = self.metadata["extent_utm_xy"]
        ny, nx = raster.shape
        dx, dy = (xmax-xmin)/nx, (ymax-ymin)/ny
        x = xmin + (np.arange(nx)+0.5)*dx - self.frame.origin_utm[0]
        y = ymin + (np.arange(ny)+0.5)*dy - self.frame.origin_utm[1]
        self.extent = np.array([xmin, ymin, xmax, ymax]) - np.tile(self.frame.origin_utm, 2)
        self.centers = [x[0], y[0], x[-1], y[-1]]
        self.interpolate = RegularGridInterpolator((y, x), raster[::-1], bounds_error=False, fill_value=np.nan)
        self.source = {"dem": config["sources"]["terrain_dem"], "metadata": str(path.with_suffix(".json").relative_to(data_root)),
                       "provider": "USGS 3DEP", "vertical_datum": self.metadata["vertical_datum"],
                       "pixel_spacing_m": [dx, dy], "metadata_details": self.metadata}
        points = np.unique(np.concatenate([lane["centerline"] for lane in scene["lanes"]]), axis=0)
        dem_z = self._dem_at(points[:, :2])
        valid = np.isfinite(dem_z)
        if not valid.any():
            raise ValueError("DEM has no valid overlap with AV2 lanes for local vertical alignment")
        self.offset = float(np.median(points[valid, 2]-dem_z[valid]))
        self.alignment = {"method": "Median local AV2 lane z minus DEM z at unique source centerline vertices",
                          "navd88_to_local_z_offset_m": self.offset,
                          "navd88_to_city_z_offset_m": self.offset + self.frame.origin[2],
                          "lane_residual": residual_summary(points[valid, 2]-dem_z[valid]-self.offset)}
        self.av2_ground = ArgoverseStaticMap.from_map_dir(
            Path(data_root)/config["sources"]["av2_log"]/"map", build_raster=True).raster_ground_height_layer

    def _dem_at(self, xy):
        xy = np.asarray(xy, dtype=float)
        flat = xy.reshape(-1, 2)
        # Allow coordinate-conversion roundoff at the DEM boundary.
        tolerance = 1e-6
        inside = ((flat >= self.extent[:2]-tolerance) & (flat <= self.extent[2:]+tolerance)).all(axis=1)
        # Clamp pixel-center interpolation to the raster's half-pixel border.
        clipped = np.clip(flat, self.centers[:2], self.centers[2:])
        z = self.interpolate(clipped[:, ::-1])
        z[~inside] = np.nan
        return z.reshape(xy.shape[:-1])

    def height_at(self, xy):
        """Return local z with the same leading shape as an (..., 2) XY input."""
        return self._dem_at(xy) + self.offset

    def to_mesh(self):
        xmin, ymin, xmax, ymax = self.bounds
        x = np.linspace(xmin, xmax, int(np.ceil((xmax-xmin)/self.spacing))+1)
        y = np.linspace(ymin, ymax, int(np.ceil((ymax-ymin)/self.spacing))+1)
        xx, yy = np.meshgrid(x, y)
        xy = np.column_stack([xx.ravel(), yy.ravel()])
        z = self.height_at(xy)
        valid = np.isfinite(z)
        protected = ((xy >= self.protected_bounds[:2]) &
                     (xy <= self.protected_bounds[2:])).all(axis=1)
        if not valid.all() and (not self.allow_exterior_nodata or np.any(~valid & protected)):
            raise ValueError(f"DEM misses {np.count_nonzero(~valid)} terrain vertices; "
                             "extend source coverage within the declared map")
        # Omit triangles where DEM coverage is unavailable.
        idx = np.arange(len(xy)).reshape(xx.shape)
        a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
        faces = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
        faces = faces[valid[faces].all(axis=1)]
        vertex_indices = np.full(len(xy), -1, dtype=int)
        vertex_indices[valid] = np.arange(np.count_nonzero(valid))
        faces = vertex_indices[faces]
        ground = self.av2_ground.get_ground_height_at_xy(self.frame.local_to_city(xy))-self.frame.origin[2]
        good = np.isfinite(ground) & valid
        slope_y, slope_x = np.gradient(z.reshape(xx.shape), y, x)
        return {"vertices": np.column_stack([xy[valid], z[valid]]).tolist(), "faces": faces.tolist(),
                "source": self.source, "bounds_local_xy": self.bounds.tolist(),
                "spacing_m": [float(x[1]-x[0]), float(y[1]-y[0])],
                "alignment": self.alignment,
                "coverage": {"dem_fraction": float(np.isfinite(z[valid]).mean()),
                             "requested_grid_dem_fraction": float(valid.mean()), "missing_exterior_vertices": int(np.count_nonzero(~valid)),
                             "declared_map_missing_vertices": int(np.count_nonzero(~valid & protected)),
                             "av2_ground_fraction": float(good.mean()),
                             "av2_ground_residual": residual_summary(ground[good]-z[good]) if good.any() else None},
                "slope_percentiles": {"p50_p95_max": np.nanpercentile(np.hypot(slope_x, slope_y), [50, 95, 100]).tolist(),
                                      "units": "rise/run, sampled DEM mesh"},
                "assumptions": ["One constant vertical offset is fitted locally; this is not an exact NAVD88-to-AV2 datum transformation.",
                                "The coarse surface uses USGS bare-earth DEM throughout; AV2 fine lane geometry is retained separately.",
                                "Residuals measure agreement with existing AV2 samples, not accuracy outside sampled AV2 coverage.",
                                "Bilinear raster interpolation; no zero fill or extrapolation beyond the DEM extent.",
                                "Explicitly enabled missing exterior context is left unmeshed; declared map and actual road heights must remain source-supported."]}
