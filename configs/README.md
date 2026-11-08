# Configuration reference

Copy `configs/build.example.json` to `configs/build.local.json` and replace the example values using the fields below.

## Build configuration

| Field | Description |
| --- | --- |
| `map_name` | Map name beginning with a letter, followed by letters, digits, or underscores |
| `output` | New run directory |
| `agent` | API root, model, key environment variable, response format, timeout, and inference parameters |
| `agent.roles` | Overrides for `decision`, `static_review`, and `dynamic_review`; other settings inherit from `agent` |
| `input.city` | AV2 city code: `ATX`, `DTW`, `MIA`, `PAO`, `PIT`, or `WDC` |
| `input.bbox_city_xy` | `[xmin, ymin, xmax, ymax]` in the AV2 city frame, in metres; optional when `input.city_config` is supplied |
| `input.area_count` | Positive number of source-supported review areas |
| `input.data_root` | Root directory of the [input data layout](../docs/INPUTS.md#data-layout) |
| `input.splits`, `input.log_ids` | Dataset splits and optional source-log subset |
| `input.city_config` | Path to an existing [city configuration](../docs/INPUTS.md#existing-city-configuration), if available |
| `input.catalog` | Path to an existing [source catalog](../docs/INPUTS.md#source-catalog), if available |
| `input.osm`, `input.terrain_dem` | Paths to existing OSM and DEM files; see [coverage requirements](../docs/INPUTS.md#supporting-sources) |
| `input.allow_download` | `true` to allow downloads of missing public data; `false` to use only local inputs |
| `software` | Installed interpreters, model weights, and native software paths |
| `resources` | Physical GPU index, CPU list, RPC port, and thread count |
| `environment` | Host-specific display, Vulkan, and other runtime variables |
| `dynamic_policy` | Configure the learned-policy adapter's `name`, `command`, `cwd`, and `config`; see [Adapter API](../docs/ADAPTERS.md#command-configuration) |
| `instructions` | Additional source requirements and the adapter's route/scenario format documentation |

Choose `resources.port` so that it and ports +1, +2, and +10 are available.

## Paths and credentials

Resolve relative build paths from the configuration file's directory. Set `software.map_python` to `null` to use the interpreter running `build_map.py`. Set `agent.api_key_env` to the name of the environment variable containing the API key.

For each adapter, write `command` as an argument array. Resolve relative `cwd`, `config`, and executable paths containing `/` from the configuration file's directory. Resolve relative script paths in the remaining arguments from `cwd`.

## Saved runs

Use `--run` to resume with the saved configuration. Changes to the original build configuration apply only to new runs.

To change API settings during a run, follow [Model API](../docs/ENVIRONMENT.md#model-api). Use a new output directory for a different source region or model setup.

## Downstream tests

Copy `examples/onevl/test.example.json` to `configs/onevl.local.json`. Set `output` to a new directory and `cases` to the source-case file. Configure `prepare` and `predictor` using the [adapter API](../docs/ADAPTERS.md#command-configuration).

Set `resources` to the test CPU/GPU allocation and an available simulator port. Use `environment` to override the construction run's environment settings when needed.
