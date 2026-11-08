#!/usr/bin/env bash
# Launch an isolated CARLA instance using the explicitly assigned CPU/GPU resources.
set -euo pipefail

: "${MAP_CPUSET:?Set MAP_CPUSET to the CPUs assigned to this run}"
: "${MAP_GPU:?Set MAP_GPU to the CARLA graphics adapter assigned to this run}"
: "${CARLA_ROOT:?Set CARLA_ROOT to the installed CARLA source project}"
MAP_RPC_PORT="${MAP_RPC_PORT:-23000}"
MAP_RUNTIME_DIR="${MAP_RUNTIME_DIR:-/tmp/mapbuild-carla-${MAP_RPC_PORT}}"
mkdir -p "$MAP_RUNTIME_DIR/config" "$MAP_RUNTIME_DIR/cache" "$MAP_RUNTIME_DIR/data"

export XDG_CONFIG_HOME="$MAP_RUNTIME_DIR/config"
export XDG_CACHE_HOME="$MAP_RUNTIME_DIR/cache"
export XDG_DATA_HOME="$MAP_RUNTIME_DIR/data"
export CUDA_VISIBLE_DEVICES="$MAP_GPU"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2

# Use the configured Vulkan loader directory.
if [[ -n "${MAP_VULKAN_LOADER:-}" ]]; then
    export LD_LIBRARY_PATH="$MAP_VULKAN_LOADER${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

cd "$CARLA_ROOT"
if [[ -f "$CARLA_ROOT/Unreal/CarlaUE4/CarlaUE4.uproject" ]]; then
    : "${UE4_ROOT:?Set UE4_ROOT to the compiled Unreal Engine source directory}"
    server=("$UE4_ROOT/Engine/Binaries/Linux/UE4Editor"
            "$CARLA_ROOT/Unreal/CarlaUE4/CarlaUE4.uproject")
    render_options=(-game)
    # Complete shader compilation before rendering.
    render_options+=("-ini:Engine:[DevOptions.Shaders]:bAllowAsynchronousShaderCompiling=False")
    # Optional isolated cache produced by scripts/build/prewarm_carla.py.
    if [[ -n "${MAP_DDC_DIR:-}" && -f "$MAP_DDC_DIR/DefaultEngine.ini" ]]; then
        render_options+=("-DefEngineIni=$MAP_DDC_DIR/DefaultEngine.ini" -DDC=MAPPrewarm)
    fi
else
    server=(./CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4)
    render_options=(-onethread)
fi
# Unreal expects the map URL before switches; keep caller arguments first.
exec taskset -c "$MAP_CPUSET" \
    "${server[@]}" "$@" \
    -RenderOffScreen -vulkan -graphicsadapter="$MAP_GPU" \
    -carla-rpc-port="$MAP_RPC_PORT" -nosound -nocrashreports "${render_options[@]}"
