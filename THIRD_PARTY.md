# Third-party components

Distilled Map Generation uses the following software, datasets, and pretrained models. Install these components separately and follow their respective licenses and data terms.

| Component | Use | Source |
| --- | --- | --- |
| Argoverse 2 | Lane geometry, poses, cameras, LiDAR, and annotations | [av2-api](https://github.com/argoverse/av2-api) |
| OpenStreetMap | Supporting roads and building footprints | [OpenStreetMap](https://www.openstreetmap.org/copyright) |
| USGS 3DEP | Terrain elevation | [3DEP](https://www.usgs.gov/3d-elevation-program) |
| SUMO | OSM conversion and road-network inspection | [SUMO](https://github.com/eclipse-sumo/sumo) |
| Scenario Dreamer | Vectorised road diffusion and autoencoder | [Scenario Dreamer](https://github.com/princeton-computational-imaging/scenario-dreamer) |
| Cosmos / Diffusers | Conditional appearance generation | [Cosmos](https://github.com/NVIDIA/cosmos), [Diffusers](https://github.com/huggingface/diffusers) |
| Cosmos Guardrail / Qwen3Guard | Components of the image-generation interface | [Cosmos Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail), [Qwen3Guard](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B) |
| Blender | Asset creation and FBX export | [Blender](https://www.blender.org/) |
| scenariogeneration / pyclothoids | OpenDRIVE and road curves | [scenariogeneration](https://github.com/pyoscx/scenariogeneration), [pyclothoids](https://github.com/phillipd94/pyclothoids) |
| CARLA | Native maps, sensors, and route-following controllers | [CARLA](https://github.com/carla-simulator/carla) |
| Unreal Engine | Rendering and native asset editing | [Unreal Engine](https://www.unrealengine.com/) |

The patches in `native/patches/` apply to the CARLA source installation. The editor plugin compiles against its Unreal environment. Additional Python dependencies are listed in `requirements/`.

LEAD and OneVL are model-integration examples. Obtain their implementations and weights from the authors, then follow the [LEAD](docs/LEAD_EXAMPLE.md) or [OneVL](docs/ONEVL_EXAMPLE.md) adapter guide.
