# Adapter API

Two interfaces connect external models to the pipeline:

- **Dynamic policy:** run a learned driving policy during construction review.
- **Paired testing:** prepare corresponding real/simulated inputs and run a planning model after construction.

Both interfaces exchange JSON files with an external process.

## Command configuration

```json
{
  "name": "model-name",
  "command": ["/absolute/model-env/bin/python", "/absolute/adapters/adapter.py"],
  "cwd": "/absolute/adapters",
  "config": "/absolute/model-settings.json"
}
```

The framework invokes:

```text
taskset -c <cpus> <command...> --request <request.json> --output <adapter_result.json>
```

`command` is an argument array executed in `cwd`. The adapter reads its settings from `request.adapter_config` and writes the result to `--output`. Paths in `cwd`, `config`, and a first executable containing `/` are relative to the declaring configuration file. Remaining command arguments are passed unchanged.

Each invocation uses a new output directory containing `request.json`, `adapter_result.json`, `output.log`, and `execution.json`. Set `execution_status: "completed"` on completion and include a nonempty `evidence` list of output files. Relative evidence paths are resolved from the invocation directory. A failed invocation returns a nonzero exit code or a failed execution status; its logs remain available.

The framework sets CPU affinity, `CUDA_VISIBLE_DEVICES`, `MAP_CPUSET`, `MAP_GPU`, and `MAP_RPC_PORT`. A model on the selected GPU uses logical `cuda:0`. Each adapter manages its own model initialization and simulator objects.

## Dynamic policy

Set `dynamic_policy` in the build configuration and implement `run_policy` in [learned_policy.py](../examples/adapters/learned_policy.py). Follow the [LEAD example](LEAD_EXAMPLE.md) for model integration.

The [dynamic-policy interface](../configs/workflows/instructions/dynamic_policy.md) defines the request, result, traffic accounting, and result reuse. Provide the traffic runner's route and scenario formats through the build configuration's `instructions`. The dynamic reviewer prepares the request and runs the adapter through `scripts.testing.run_dynamic`.

## Paired testing

`test_map.py` reads a completed construction run and writes to a separate test directory:

```text
prepare_pair → real prediction → simulated prediction → scoring
```

Configure `prepare` and `predictor` with the command interface above. They may use separate Python environments. The same predictor configuration is used for both image sources.

The process inherits the build's engine and graphics environment, uses the delivered `CARLA_ROOT`, adds this package to `PYTHONPATH`, and sets `MAP_RUNTIME_DIR` to the test output directory. The test config's `environment` overrides these values, while `resources` determines CPU/GPU allocation and ports.

### Pair preparation

The request contains `kind: "prepare_pair"`, `map_name`, `revision`, `artifacts`, `carla_root`, `resources`, and a `cases` file. The artifact index provides the runtime and delivery manifests.

Return the map identity, evidence, limitations, and three JSON paths: `real_inputs`, `sim_inputs`, and `reference`. The preparation adapter starts its own simulator on the test port, captures the paired views, and closes it when finished.

Real and simulated input bundles share this structure:

```json
{
  "coordinate_frame": "anchor_ego_forward_left_m",
  "future_times_s": [0.5, 1.0],
  "cases": [{
    "case_id": "example_anchor",
    "log_id": "source-log-id",
    "timestamp_ns": 1000000000,
    "context": {"prompt": "causal task prompt", "states": []},
    "views": [{
      "camera": "front",
      "timestamp_ns": 1000000000,
      "image": "/absolute/image.png",
      "intrinsics": [[1,0,0],[0,1,0],[0,0,1]],
      "camera_to_ego": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    }]
  }]
}
```

Replace the example calibration with source values. Views may include several cameras and historical frames; their timestamps are at or before the anchor. Paired bundles share every field except image paths, including camera order, calibration, timestamps, and causal context. Record simulator ticks separately in capture evidence.

`reference` contains the same `coordinate_frame` and `future_times_s`, with:

```json
{"cases": [{"case_id": "example_anchor", "trajectory_xy_m": [[0.5, 0.0], [1.0, 0.0]]}]}
```

The example coordinates illustrate the format. Trajectories are in anchor-ego coordinates, in meters, with X forward and Y left. The time grid starts after t=0. Future reference trajectories are provided to scoring; prediction inputs contain only causal history and task context.

### Prediction

The request contains `kind: "predict"`, map identity, `side: "real"` or `"sim"`, `inputs`, `resources`, and adapter configuration. Return:

```json
{
  "execution_status": "completed",
  "map_name": "ExampleMap",
  "revision": 1,
  "side": "real",
  "model": {"name": "model-name", "checkpoint": "/absolute/checkpoint", "settings": {"seed": 0}},
  "coordinate_frame": "anchor_ego_forward_left_m",
  "future_times_s": [0.5, 1.0],
  "cases": [{
    "case_id": "example_anchor",
    "valid": true,
    "trajectory_xy_m": [[0.5, 0.0], [1.0, 0.0]]
  }],
  "evidence": ["raw_model_outputs.json"]
}
```

The `model` object is identical across both sides. For an invalid prediction, set `valid: false`, `trajectory_xy_m: null`, and an `error` message. Missing predictions are counted as invalid cases.

### Scores

Scoring reports `real_vs_reference`, `sim_vs_reference`, and `sim_vs_real`. Per-case outputs include `error_by_time_m` and cumulative `ade_by_horizon_m`. Aggregate `by_horizon` entries contain `time_s`, `ade_m`, and `fde_m`.

`all_valid_summary` uses each metric's available cases; `paired_cases_summary` uses the common valid set. Predictions must cover the complete time grid. Empty valid sets yield `null` metrics.

At each endpoint, ADE averages Euclidean distances over all samples up to that point; FDE is the distance at the endpoint. Both are measured in meters.

| Metric | Comparison |
| --- | --- |
| `real_vs_reference` | Real-image prediction against recorded future |
| `sim_vs_reference` | Simulated-image prediction against recorded future |
| `sim_vs_real` | Simulated-image prediction against real-image prediction |
