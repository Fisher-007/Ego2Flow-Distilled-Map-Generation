# Distilled Map Generation

## Overview

Distilled Map Generation constructs CARLA maps from Argoverse 2 road geometry, images, LiDAR, and poses, together with OpenStreetMap and terrain data. An information graph connects source observations to road structure, traffic controls, and visual assets. Separate construction, static review, and dynamic review agents refine the map through feedback.

The pipeline uses Scenario Dreamer for local road completion, Cosmos for appearance generation, and CARLA/Unreal for native map assembly. Dynamic review uses BasicAgent and a configurable learned driving policy. Optional downstream testing evaluates paired real and simulated inputs with the same planning model.

## Installation

Requirements:

- Linux x86-64, Python 3.10, an NVIDIA GPU, and a working Vulkan environment.
- A CARLA 0.9.16 source build with UE4.26 and a matching Python client.
- Blender 4.5, SUMO 1.27, FFmpeg, `taskset`, and `patch`.
- Separate Scenario Dreamer and Cosmos environments with their pretrained models.
- A vision-language API supporting Chat Completions and structured responses.

Create the map environment:

```bash
python3.10 -m venv /path/to/map-env
source /path/to/map-env/bin/activate
python -m pip install -r requirements/map.txt
```

Install the generation models and native dependencies following [Environment setup](docs/ENVIRONMENT.md), and install `requirements/blender.txt` with Blender's Python. Sources are listed in [Third-party components](THIRD_PARTY.md).

## Configuration

Prepare the input data following [Inputs](docs/INPUTS.md).

```bash
cp configs/build.example.json configs/build.local.json
export MAP_AGENT_API_KEY='your-api-key'
```

Edit `configs/build.local.json` following the [configuration reference](configs/README.md).

Configure the model adapters in `examples/adapters/`:

- **Dynamic review:** implement `learned_policy.py:run_policy` and configure `dynamic_policy` in the build configuration. See the [LEAD example](docs/LEAD_EXAMPLE.md).
- **Downstream testing:** implement `paired_test.py:prepare_pair` and `predict`, then configure `prepare` and `predictor` in `configs/onevl.local.json`. See the [OneVL example](docs/ONEVL_EXAMPLE.md).

The examples are templates. Install the selected models separately and implement their calls using the [adapter API](docs/ADAPTERS.md). Configure the learned-policy adapter before running the full construction pipeline.

## Run

### Build a map

Prepare the native plugin and patches once for the configured CARLA installation, check the environment, and start construction:

```bash
python build_map.py prepare-native --config configs/build.local.json
python build_map.py doctor --config configs/build.local.json
python build_map.py run --config configs/build.local.json
```

The map identity and artifact paths are written to `<output>/result.json`.

### Manage a run

Check progress or resume an interrupted run:

```bash
python build_map.py status --run runs/ExampleMap
python build_map.py run --run runs/ExampleMap
```

Use `--max-actions 20` to limit a runner invocation to 20 agent actions; the default `0` continues until completion or a request for input.

For a `blocked` run, read the finding in `events.jsonl`, provide the requested information in a text file, and resume:

```bash
python build_map.py respond --run runs/ExampleMap --instruction answer.txt
python build_map.py run --run runs/ExampleMap
```

To stop a run, press Ctrl+C to exit the foreground runner, then stop its remaining jobs:

```bash
python build_map.py stop --run runs/ExampleMap
```

To revise a completed map, write the requested changes to `feedback.txt`, reopen the run, and resume:

```bash
python build_map.py reopen --run runs/ExampleMap --instruction feedback.txt
python build_map.py run --run runs/ExampleMap
```

### Downstream testing

After completing construction and configuring the downstream adapters:

```bash
python test_map.py doctor --run runs/ExampleMap --config configs/onevl.local.json
python test_map.py run --run runs/ExampleMap --config configs/onevl.local.json
```

Trajectory errors and coverage are written to `scores.json`.

### Individual tools

```bash
python run_tool.py list --stage structure
python run_tool.py run --run runs/ExampleMap --tool generate_corridor -- --help
```

`run_tool.py` selects the configured interpreter, software paths, and CPU/GPU allocation. Arguments after `--` are passed to the selected tool.

Load a completed map on the configured simulator port:

```bash
python run_tool.py run --run runs/ExampleMap --tool run_carla -- \
  /Game/MAPExampleMap/Maps/ExampleMap/ExampleMap
```

### Offline tests

```bash
python -m pip install -r requirements/test.txt
make test PYTHON=python
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and tests.
