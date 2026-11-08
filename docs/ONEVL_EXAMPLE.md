# OneVL: paired planning evaluation

The downstream pipeline compares OneVL predictions from real and simulated images of the same source windows. Implement `prepare_pair` and `predict` in [paired_test.py](../examples/adapters/paired_test.py) using the installed OneVL model and its input format.

## Configuration

| Example | Settings |
| --- | --- |
| [test.example.json](../examples/onevl/test.example.json) | Output directory, cases, resources, and adapter commands |
| [prepare.example.json](../examples/onevl/prepare.example.json) | Source data, registration, camera history, and actor mapping |
| [model.example.json](../examples/onevl/model.example.json) | Model source, weights, prompt, inference settings, and prediction times |
| [cases.example.json](../examples/onevl/cases.example.json) | Source log, split, anchor timestamp, and cameras |

Install OneVL in its own environment. Copy the examples to local configuration files and fill in the paths and source timestamps. The example time grid samples every 0.5 seconds through 5 seconds. Both predictions and the reference use the same grid.

## Implement `prepare_pair`

1. Read the delivered map, source registration, and requested windows. Select windows with the required images, history, and future reference.
2. Prepare causal image history, vehicle state, task context, and navigation in the model's expected order.
3. Start CARLA on the configured test port. Reproduce source camera poses, calibration, timestamps, and actor context, then capture simulated images. `prepare_replay_clip`, `replay_dynamic_history`, and `render_av2_view` provide source and rendering utilities for the adapter.
4. Write `real_inputs.json` and `sim_inputs.json` with matching fields except image paths. Save camera alignment and native capture evidence.
5. Transform recorded future trajectories into anchor-ego coordinates and write `reference.json` separately. Return the three paths, map identity, evidence, and coverage limitations; close the simulator and objects created for capture.

Use the input format in [ADAPTERS.md](ADAPTERS.md#pair-preparation). The adapter assembles multi-camera history and model-specific context from these utilities.

## Implement `predict`

1. Read `request.inputs` and `request.adapter_config`. Load OneVL, its processor, and the model's prompt/token format in inference mode.
2. Encode each case's images and causal context. Use the same weights, preprocessing, seed, decoding, and prediction horizon for real and simulated inputs.
3. Save raw model output and parse trajectories. Convert the model's coordinates to anchor-ego X-forward/Y-left coordinates in meters. Return an invalid result with an error message when parsing fails or the trajectory is incomplete.
4. Return the [prediction schema](ADAPTERS.md#prediction), with identical `model` metadata on both sides and the corresponding `side` value.

The predictor receives causal inputs. Future reference trajectories are consumed by the scoring stage.

## Run and inspect results

Copy the test configuration to `configs/onevl.local.json`, then run:

```bash
python test_map.py doctor --run runs/ExampleMap --config configs/onevl.local.json
python test_map.py run --run runs/ExampleMap --config configs/onevl.local.json
```

Construction must be complete. Choose an available test port and a new output directory. The test inherits the construction environment; override display or Vulkan settings in the test config as needed.

The output contains `prepare/`, `real/`, and `sim/` execution records, normalized inputs, future references, `scores.json`, and `result.json`. Scores include prediction-to-reference errors, simulation-to-real prediction differences, and valid-case coverage.

For the example time grid, `paired_cases_summary.sim_vs_real.by_horizon` includes cumulative ADE and endpoint FDE at 1, 2, 3, 4, and 5 seconds. Full-horizon values also appear as `ade_m` and `fde_m`.
