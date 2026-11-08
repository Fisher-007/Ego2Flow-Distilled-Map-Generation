# Review procedures

Apply the procedures for the assigned static or dynamic review role.

Static review uses source-aligned views prepared by `prepare_city_view_snapshots`, `prepare_city_review_views`, and `render_city_views`. Geometry measurements include `probe_pavement`, `measure_native_road_support`, and `review_packed_roads`. Check roads together with terrain, markings, curbs, and grounded assets.

Dynamic review prepares source routes with `prepare_city_driving_routes` and `prepare_city_transfer_routes`, resolves them with `check_city_driving_plans --native-planner`, and drives the plans with `drive_city_routes`. `check_multilane_runtime` measures lane transitions. `drive_traffic` checks route interactions with the applicable signal bindings.

Run the configured learned policy under sustained traffic demand using `scripts.testing.run_dynamic --run ... --request ... --output ...`. The user-supplied runner documentation defines the routes and scenario input schema. Register `result.json` as `dynamic_policy_results` and inspect its control records, demand accounting, and driving outcomes. Use the [dynamic-policy interface](dynamic_policy.md).

For source-time replay, use `scripts.assets.prepare_replay_clip` with the source log, split, time window, camera, scene, and actor layers, followed by `scripts.scene.replay_dynamic_history`. Local images use `--local-images`. Replay frames, static view indexes, and driving plans have separate formats.

Review reports contain the current `revision`, evidence paths, coverage, and findings. An accepted report uses `status: "pass"`. Feedback identifies the source/location, observed issue, affected conditions, and requested change. It returns construction to the affected stage and invalidates dependent completion; affected artifacts are registered again after revision.

