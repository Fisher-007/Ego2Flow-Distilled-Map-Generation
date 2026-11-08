"""Launch CARLA, wait for the map, run environment checks, and stop the server."""

import argparse
import os
from pathlib import Path, PurePosixPath
import signal
import socket
import subprocess
import sys
import time


def wait_for_world(server, port, timeout, server_log, load_map=True):
    import carla

    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(
                f"CARLA exited with code {server.returncode}; see {server_log}"
            )
        # An early connection can remain unusable; each probe gets a fresh client.
        client = carla.Client("127.0.0.1", port)
        client.set_timeout(5.0)
        try:
            world = client.get_world()
            return world.get_map().name if load_map else world
        except RuntimeError as error:
            last_error = error
        time.sleep(1)
    raise TimeoutError(
        f"CARLA world/map was not ready within {timeout:g}s; see {server_log}"
    ) from last_error


def stop_server(server):
    try:
        os.killpg(server.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        server.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        server.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Result directory")
    parser.add_argument(
        "--map", default="/Game/Carla/Maps/Town10HD_Opt",
        help="Unreal map asset path (default: %(default)s)",
    )
    parser.add_argument(
        "--startup-timeout", type=float, default=3600,
        help="Seconds allowed for world/map readiness (default: %(default)s)",
    )
    parser.add_argument("--quality-level", choices=("Low", "Epic"), default="Epic")
    args = parser.parse_args()
    for name in ("MAP_CPUSET", "MAP_GPU"):
        if not os.environ.get(name):
            parser.error(f"Set {name} to this run's allocated resources")
    if args.startup_timeout <= 0:
        parser.error("--startup-timeout must be positive")

    port = int(os.environ.get("MAP_RPC_PORT", "23000"))
    # Require the launch ports to be available.
    for checked_port in (port, port + 1, port + 2):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", checked_port))

    script_dir = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    server_log = output / "server.log"
    with server_log.open("w") as log:
        server = subprocess.Popen(
            ["bash", str(script_dir / "run_carla.sh"), args.map,
             f"-quality-level={args.quality_level}"],
            cwd=script_dir.parents[1],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"Started CARLA PID {server.pid} on port {port}", flush=True)
        try:
            started = time.monotonic()
            map_name = wait_for_world(server, port, args.startup_timeout, server_log)
            # CARLA may return a Content-relative name without the /Game/ prefix.
            actual = PurePosixPath(
                map_name.removesuffix(".umap").removeprefix("/Game/")
            )
            requested = PurePosixPath(
                args.map.removesuffix(".umap").removeprefix("/Game/")
            )
            matches = (
                actual == requested if args.map.startswith("/Game/")
                else actual.name == requested.name
            )
            if not matches:
                raise RuntimeError(
                    f"CARLA loaded {map_name!r}, expected {args.map!r}; see {server_log}"
                )
            print(
                f"World ready: {map_name} after {time.monotonic() - started:.1f}s",
                flush=True,
            )
            with (output / "check.log").open("w") as check_log:
                subprocess.run(
                    [
                        "taskset", "-c", os.environ["MAP_CPUSET"],
                        sys.executable, str(script_dir / "check_carla.py"),
                        "--port", str(port), "--output", str(output),
                    ],
                    stdout=check_log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=360,
                )
            print(f"CARLA check passed; results in {output}", flush=True)
        finally:
            stop_server(server)
            print("Stopped this CARLA process group", flush=True)


if __name__ == "__main__":
    main()
