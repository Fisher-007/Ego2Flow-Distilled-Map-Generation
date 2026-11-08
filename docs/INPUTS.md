# Inputs

## Data layout

Store local AV2 Sensor, OSM, and terrain data in this layout:

```text
data/
├── av2/
│   ├── sensor/
│   │   ├── train/<log_id>/
│   │   │   ├── city_SE3_egovehicle.feather
│   │   │   ├── annotations.feather
│   │   │   ├── calibration/
│   │   │   ├── map/log_map_archive_*.json
│   │   │   └── sensors/
│   │   │       ├── cameras/ring_*/<timestamp_ns>.jpg
│   │   │       └── lidar/<timestamp_ns>.feather
│   │   └── val/<log_id>/...
│   └── metadata/
├── osm/region.osm
└── terrain/region.tif
```

Set `input.data_root` to the `data/` directory above. Set the region, source filters, and download options using the [configuration reference](../configs/README.md#build-configuration).

## Source catalog

If supplying `input.catalog`, use a JSON file with top-level `city` and `candidates`. Include `split`, `log_id`, `vector_map`, and `bbox_city_xy` in each candidate. Set `vector_map` relative to `input.data_root/av2`.

## Existing city configuration

If supplying `input.city_config`, check these fields in the JSON file:

| Field | Contents |
| --- | --- |
| `prototype_id` | Set to `map_name` from the build configuration |
| `city`, `log_id`, `split` | Primary source |
| `coordinates` | `origin_city_xyz`, units, and origin evidence |
| `bbox_city_xy` | Region bounds |
| `sources` | AV2 logs, OSM, and terrain paths relative to `input.data_root`, or absolute paths |
| `landmarks`, `coordinate_check_route` | Registration references |
| `coarse`, `fusion` | Reconstruction parameters |
| `areas` | Area IDs, centres, core and influence bounds, and source observations |

Include `log_id`, `split`, and `nearest_timestamp_ns` in each source observation. Preserve the measured vertical origin in `coordinates.origin_city_xyz`.

## Supporting sources

Provide OSM and DEM data covering the declared region and its road and visibility buffers.

Reference additional building footprints, point clouds, or assets in `instructions`, including their coordinate system and source information.
