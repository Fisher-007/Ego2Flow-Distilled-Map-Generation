# Environment setup

Install map processing, Scenario Dreamer, and Cosmos in separate Python environments. Their paths are configured in the build file's `software` section.

## Map tools

Use Python 3.10 and install `requirements/map.txt`. Use the Python client built for the configured CARLA server. Install `requirements/blender.txt` with Blender's bundled Python. SUMO supplies `netconvert` and `SUMO_HOME`; its release should match `sumolib` in the map environment.

## Scenario Dreamer

Install the [Scenario Dreamer source and pretrained models](https://github.com/princeton-computational-imaging/scenario-dreamer). The road generator uses the Waymo large LDM, its autoencoder, and the associated latent statistics, with `num_l2l_blocks=3`.

Use the upstream environment or install `requirements/scenario-dreamer.txt` in a separate Python 3.10 environment with matching CUDA and graph-operator dependencies. Keep the source `cfgs/`, `models/`, `utils/`, and inference metadata available.

| Configuration field | Path |
| --- | --- |
| `scenario_dreamer_python` | Environment's Python executable |
| `scenario_dreamer_source` | Source root containing `cfgs/config.yaml` |
| `scenario_dreamer_data` | Inference data and latent-statistics root |
| `scenario_dreamer_checkpoint` | Large LDM checkpoint |
| `scenario_dreamer_autoencoder` | Corresponding autoencoder checkpoint |

The launcher sets `PROJECT_ROOT`, `CONFIG_PATH`, `SCRATCH_ROOT`, `DATASET_ROOT`, `MAP_SD_CHECKPOINT`, and `MAP_SD_AUTOENCODER` from these fields.

## Cosmos

Follow the [Cosmos 3 Diffusers transfer example](https://github.com/NVIDIA/cosmos-framework/blob/main/.agents/skills/cookbooks/cosmos3/generator/transfer/run_video_transfer_with_diffusers.ipynb) to install the environment and models. The adapter imports `Cosmos3OmniModularPipeline`, `UniPCMultistepScheduler`, and `cosmos_guardrail`.

Set `cosmos_model` to the local model directory and arrange the checker models as follows:

```text
cosmos_guardrail_root/
├── nvidia--Cosmos-Guardrail1/   # Local nvidia/Cosmos-1.0-Guardrail snapshot
└── Qwen--Qwen3Guard-Gen-0.6B/   # Local Qwen/Qwen3Guard-Gen-0.6B snapshot
```

Each generation case contains source references, material prompts, spatial scale, conditioning images, and sampling parameters. Use `--group-offload` when the model requires CPU offloading.

## CARLA and Unreal

Follow the [CARLA 0.9.16 Linux source-build instructions](https://carla.readthedocs.io/en/0.9.16/build_linux/) with the matching UE4.26 installation. Install the resulting Python 3.10 client wheel. Required paths include:

```text
CARLA_ROOT/
├── Unreal/CarlaUE4/CarlaUE4.uproject
├── Unreal/CarlaUE4/{Content,Config,Plugins,Binaries}/
├── PythonAPI/carla/
└── Util/BuildTools/Import.py
UE4_ROOT/
├── Engine/Binaries/Linux/UE4Editor
└── Engine/Build/BatchFiles/Linux/Build.sh
```

Use a dedicated CARLA source installation. After building CARLA, run:

```bash
python build_map.py prepare-native --config configs/build.local.json
```

This installs the editor plugin and three patches in `native/`, then compiles the editor. Each construction run creates a separate asset project using the installed engine and CARLA assets.

Offscreen rendering requires Vulkan and the host's graphics environment. Set `DISPLAY`, `VK_ICD_FILENAMES`, or `MAP_VULKAN_LOADER` in `environment` as required. `MAP_VULKAN_LOADER` is a directory added to `LD_LIBRARY_PATH`.

## Model API

Set `base_url` to the API root, such as a provider's `/v1` endpoint; the client appends `/chat/completions`. The model must support images and structured actions. Select `json_schema` or `tool_call` for `response_mode`, and place supported inference options in `parameters`.

All three roles can share an endpoint or use per-role overrides.

To change API settings, stop the foreground runner and run:

```bash
python -m scripts.workflow.run configure-agent --run /absolute/run \
  --config /absolute/agent-settings.json
```

The JSON contains the fields of the `agent` object, including `provider: "http"`. New settings apply to the next new request. A pending request retains its recorded model, parameters, and images.
