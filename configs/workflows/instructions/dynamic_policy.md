# Dynamic-policy interface

The configured adapter exchanges JSON request and result files through an external process. It reads model settings from `request.adapter_config` and writes the result to `--output`. Each invocation records `request.json`, `adapter_result.json`, `output.log`, and `execution.json`.

Set `execution_status: "completed"` on completion and include a nonempty `evidence` list. Relative evidence paths resolve from the invocation directory. The framework sets CPU affinity, `CUDA_VISIBLE_DEVICES`, `MAP_CPUSET`, `MAP_GPU`, and `MAP_RPC_PORT`; the model uses logical `cuda:0`.

## Request

The dynamic reviewer prepares a request for the current compiled map:

```json
{
  "map_name": "ExampleMap",
  "revision": 1,
  "map_asset": "/Game/MAPExampleMap/Maps/ExampleMap/ExampleMap",
  "runtime_manifest": "/absolute/run/compile/runtime_manifest.json",
  "scene": "/absolute/run/scene.json",
  "routes": "/absolute/run/dynamic/routes.json",
  "scenario": "/absolute/run/dynamic/scenario.json",
  "observation_seconds": 180,
  "scope": "Approach, connecting road, and traffic interaction"
}
```

`runtime_manifest` references the current compilation artifact. `routes` describes source-supported native routes. `scenario` specifies initial states, arrival times, vehicles, and request IDs. Their schema belongs to the external traffic runner; provide its documentation path in the build config's `instructions` so the agent can prepare these inputs.

Route preparation can use `prepare_city_driving_routes`, `prepare_city_transfer_routes`, and `check_city_driving_plans`. The dynamic tool executes the adapter:

```bash
python -m scripts.testing.run_dynamic \
  --run /absolute/construction-run \
  --request /absolute/construction-run/dynamic/request.json \
  --output /absolute/construction-run/dynamic/policy-attempt-01
```

The tool adds `resources`, `carla_root`, `adapter_name`, and `adapter_config` to the request. The adapter connects to the framework's CARLA service at the requested port, checks the loaded map, and controls its vehicles and sensors. On exit, it destroys its objects and restores changed world settings while leaving the service running.

## Result

| Field | Contents |
| --- | --- |
| `kind` | `dynamic_policy` |
| `execution_status` | `completed` when the observation finishes |
| `map_name`, `revision` | Request identity |
| `policy` | `name` matching the configured adapter, `checkpoint`, and actual inference/control `settings` |
| `learned_control` | Controlled vehicle `request_ids` and positive `control_steps` |
| `traffic` | Demand accounting and observation duration |
| `evidence` | Trajectories, control logs, sensor alignment, collision records, or image/video indexes |
| `model_failures` | Observed driving failures, or an empty array |
| `limitations` | Coverage and model limitations, or an empty array |

The `traffic` object contains:

| Field | Meaning |
| --- | --- |
| `requested` | Total demand generated during the observation |
| `admitted` | Requests admitted to the map |
| `completed` | Requests that completed their route |
| `unadmitted` | Requests still waiting for admission |
| `in_system` | Admitted requests still active at the end |
| `terminated_unfinished` | Admitted requests removed before completing |
| `observation_seconds` | Simulated duration, covering the requested interval |
| `waiting_vehicle_seconds` | Cumulative time from request to admission or observation end |
| `in_system_vehicle_seconds` | Cumulative time from admission to completion, removal, or observation end |

Counts are nonnegative integers and satisfy:

```text
requested = admitted + unadmitted
admitted = completed + in_system + terminated_unfinished
```

Generate traffic demand throughout the observation window. Record learned-policy outcomes separately from background traffic. A completed observation may contain driving failures, which the reviewer evaluates alongside the map evidence.

The tool writes `result.json` after validating the response. Register it as `dynamic_policy_results`, together with the driving, traffic, replay, and review artifacts.

## Reusing an unaffected result

After a local map change, the reviewer can reference an unaffected policy run with:

- `kind: "dynamic_policy_reuse"` and the current `map_name` and `revision`.
- `previous_result`: the earlier execution's `result.json`.
- `unchanged_scope`: the routes, sensor views, controls, and model settings that remain applicable.
- `evidence`: files supporting that scope assessment.

Affected coverage is rerun on the current map.

