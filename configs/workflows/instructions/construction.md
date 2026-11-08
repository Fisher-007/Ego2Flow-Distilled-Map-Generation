# Construction recipe

The agent receives input configuration, environment paths, allocated resources, the run directory, and stage artifacts. Inspect tool signatures before calling a command:

```bash
python -m scripts.workflow.inventory --stage fusion --signatures
python -m scripts.workflow.inventory --tool pack_lane_groups --signatures
python -m scripts.workflow.inventory --module scripts.scene.prepare_city_ground
```

Tools perform source retrieval, geometry, sampling, export, and measurement. The construction agent selects sources and conditions; reviewers interpret the resulting evidence.

## Sources and fusion

1. Read the configured inputs. Use an existing `city_config` and catalog, or run `index_city_network` for the requested splits. For local data, use `index_local_sources`. Both index tools take `input.data_root/av2`; `prepare_city_selection` and construction tools take `input.data_root`. Preserve each catalog entry's split and apply the configured city, region, and log selection.
2. Inspect `propose_city_areas` and `review_city_coverage`. Write a selection containing `city`, `selected_candidate_ids`, `bbox_city_xy`, and `selection_rationale`. `prepare_city_selection` creates `city_config.json` and `area_selection.json`. Set `prototype_id` to the map name and retain the measured XYZ origin.
3. Prepare source views with `fetch_city_evidence` and `plot_city_evidence`. In local mode, read existing images, calibration, poses, and LiDAR into the same view-index format.
4. Use configured OSM and DEM files or acquire them with `fetch_osm_region` and `prepare_terrain`. Terrain preparation uses a selected raster ID/name and `coarse.terrain_bounds_local_xy`.
5. Run `build_coarse` and `check_coarse`. Set the route check's `start_lane/end_lane` from passenger-permitted lanes in the converted SUMO network. `map_generation.coarse.convert_osm` can produce that network before the complete coarse build.
6. Inspect registration, overlaps, source epochs, and ownership. Write source and geometry reviews, then run `assemble_city_sources` and `fuse_city_sources`. Assembly paths are relative to `data_root`; vector-map files retain the `log_map_archive_` prefix.
7. Use `restore_city_access`, `review_city_access`, and `review_city_reachability` to classify interfaces by missing geometry, permissions, source disagreement, or native representation.

Stage artifacts:

| Stage | Artifacts |
| --- | --- |
| `source_selection` | `city_config`, `source_catalog`, `area_selection` |
| `coarse` | `coarse_scene` |
| `fusion` | `fused_scene`, `source_review` |

## Information graph

`prepare_city_controls` retrieves source control evidence. After reviewing images, use `prepare_city_graph` with the scene, city configuration, and a `visual_review` containing `agent` and `areas`. Each area supplies `area_id`, `log_id`, `views`, and `findings`; findings identify retained attributes and their evidence.

The `information_graph inspect/revise/compile` commands connect observations, roads, objects, tasks, decisions, and construction conditions. Use the construction role's identity for `--agent`. `bind_city_control_approaches` and `prepare_signal_controls` link controlled lanes, stopping positions, and observed signal orientation.

Submit `information_graph` and `generation_conditions`.

## Structure generation

For a single gap:

1. Run `prepare_access_corridor --scene ... --graph ... --seam ... --reason ... --output ...` to produce the corridor scene, graph, decision, and metadata.
2. Run `scripts.assets.generate_feedback_road sample` in the Scenario Dreamer environment with `--decision`, `--source-dir`, `--output`, `--information-graph`, and `--graph-task city_corridor_generation`. Checkpoints come from the configured environment.
3. Review the sample and `raw_review_input.json`. Set `road.revision` parameters in the decision file for `repair`, then call `assemble`, which reads `repair_review_input.json`.
4. Apply geometric constraints with `constrain_generated_corridor` and merge the accepted result with `apply_access_corridor`. Coupled interfaces use `group_access_repair_regions` and `generate_access_group`.

Record source coverage, commands, model inputs and outputs, constraints, acceptance decisions, and protected geometry in `structure_report`. Fully observed interfaces retain their source geometry. Submit `structure_report` and `road_scene`.

## Appearance generation

1. Select source image crops, material properties, and spatial scales for `prepare_city_materials`. Set regional `material_context` and any material-family overrides.
2. Run `scripts.appearance.cosmos3_transfer` in the Cosmos environment with `--model`, `--case`, `--guardrail-root`, `--information-conditions`, `--information-scope material_only`, and `--output`. Store each case's result in its `generated/` directory.
3. Review `generated_*.png`, then convert accepted images with `scripts.assets.materialize_frontage_materials --root ...`.
4. Prepare building support before `prepare_city_architecture`, which uses `final_support_range_m`. Export and import the resulting geometry and materials into the native map.
5. Use the canopy, landcover, actor-layer, and actor-appearance tools for the corresponding source observations. Keep moving actors separately controllable.

Submit `appearance_report` and `asset_manifest`, with references to the adopted native materials and entities.

## Native assembly

`build_roads --scene ... --output ... --workers ... --carla-root ... --native-planner` creates road geometry, OpenDRIVE, and source mappings. Assemble multi-lane geometry as follows:

1. Find source chains with `review_native_corridors`; use `prepare_coarse_lane_groups` for parallel coarse lanes.
2. Define accepted groups by common ports, successors, and lateral order. `groups.json` has the form `{"groups": [[["lane_a", "lane_b"], ["lane_c", "lane_d"]]]}`: each outer entry is a group, its columns are ordered left to right, and each column is a longitudinal lane chain.
3. Compile sections with `prepare_native_sections`, then run `pack_lane_groups --group groups.json ...` using the section cache and spacing. Review native topology for all retained roads.
4. Build physical road surfaces with `build_packed_surfaces` and terrain/support with `prepare_city_ground`. Height and shared-ground tools maintain layer separation and contacts.
5. Export assets with `build_assets`. Run `export_frontage_architecture.py` through Blender using `--background --python-exit-code 1 --python ... -- ...`.
6. Import with `scripts.build.import_map --package MAP<name> --map-name <name> ...`. Install navigation, materials, signs, buildings, vegetation, and replacement surfaces using the corresponding build tools.
7. Bind source signal displays with `install_observed_signal_heads` and `register_native_signal_groups`; `--opendrive-triggers` selects lane-validity triggers. Reopen the map to inspect saved bindings and use `prewarm_carla` to prepare its cache.

Tracked jobs set `MAP_CPUSET`, `MAP_GPU`, and `MAP_RPC_PORT`. CUDA sees the assigned GPU as logical device 0; CARLA uses its physical Vulkan adapter index. Start `bash scripts/env/run_carla.sh <level>` as a `service=true` job and run measurements on the same resource slot.

Submit `native_build` and `runtime_manifest`. The manifest lists the native map, scene, roads, physical surfaces, graph conditions, materials, assets, controls, source/camera registration, and route inputs.

## Static and dynamic review

Reviewers follow the [review procedures](review.md). Construction addresses their feedback before delivery.

## Delivery

`delivery_manifest` links the native map, loading and actor/control initialization instructions, review coverage, and remaining limitations. `reproduction_manifest` lists sources, models, generated and adopted artifacts, code/configuration files, and command order. Stop the run's remaining services before submitting delivery.

After delivery, `test_map.py` can run paired planning evaluation through its preparation and prediction adapters.
