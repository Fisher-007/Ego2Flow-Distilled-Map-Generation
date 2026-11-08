# Method

Map construction preserves traffic-relevant geometry, connectivity, controls, and visibility. An information graph links source observations to retention decisions, generation conditions, native entities, and review feedback.

## Pipeline

| Stage | Implementation | Outputs |
| --- | --- | --- |
| Source selection and coordinates | `scripts.data.prepare_city_selection`, `map_generation.coordinates`, `map_generation.scene` | Source logs and times, XYZ origin, bounds, calibration |
| Coarse reconstruction and fusion | `build_coarse`, `assemble_city_sources`, `fuse_city_sources`, access and ownership tools | Lane geometry, topology, permissions, source alignment |
| Information graph | `scripts.agent.prepare_city_graph`, `map_generation.information_graph` | Observations, traffic relations, retention decisions, conditions |
| Constrained road completion | `prepare_access_corridor`, `scripts.assets.generate_feedback_road`, `generate_access_group` | Samples, latent constraints, accepted road geometry |
| Conditional appearance | `prepare_city_materials`, `scripts.appearance.cosmos3_transfer`, `materialize_frontage_materials` | Source crops, generated textures, native material bindings |
| Native assembly | `build_roads`, `pack_lane_groups`, `build_packed_surfaces`, `build_assets`, `scripts.build.*` | OpenDRIVE, physical meshes, terrain, assets, controls |
| Static review | Static reviewer and native rendering/geometry tools | Source-aligned views, topology and contact measurements |
| Dynamic review | `drive_city_routes`, `run_dynamic_policy`, `replay_dynamic_history` | Physical driving, learned-policy control, traffic observations, source replay |
| Feedback and delivery | `map_generation.city_workflow` | Revised conditions, updated artifacts, delivery manifests |

## Structure and appearance

Scenario Dreamer supplies a pretrained local road prior. Known geometry and topology remain fixed while missing interfaces undergo local latent denoising. Feedback can add endpoint, tangent, smoothness, and displacement constraints. Final geometric boundary alignment is recorded separately from model sampling.

Cosmos generates appearance from source crops and graph-compiled material conditions. Accepted textures are converted to reusable materials and imported into the shared 3D scene. Traffic signs, road connectivity, and measured occlusion geometry remain tied to source conditions. Moving traffic participants are represented as separate actors.

When the source already covers a road interface, its measured geometry is retained. Generation records distinguish sampled candidates, accepted results, and geometric postprocessing.

## Review and refinement

Construction, static review, and dynamic review use separate agent contexts. Reviewers inspect the current map and return findings with source locations and affected conditions. Construction updates the graph and native artifacts, then repeats affected checks. Unchanged results can be reused with their scope recorded.

Static review compares native views and geometry with source evidence. Dynamic review combines BasicAgent route checks, sustained traffic with a configured learned driving policy, and source-time camera/actor replay. Delivery follows acceptance by both reviewers.

## Downstream evaluation

`test_map.py` evaluates paired real and simulated inputs after construction. Preparation and prediction use external adapters; trajectory metrics are computed by `map_testing.scoring`.
