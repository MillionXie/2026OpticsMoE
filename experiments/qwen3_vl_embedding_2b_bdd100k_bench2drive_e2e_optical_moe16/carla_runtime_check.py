from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the CARLA 0.9.15 runtime")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=24515)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    try:
        import carla
    except ImportError as exc:
        raise RuntimeError(
            "CARLA Python API is unavailable. Activate conda environment RFL "
            "and source SERVER_RFL_ENV.sh."
        ) from exc
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    maps = client.get_available_maps()
    print("CARLA client module:", carla.__file__)
    print("Connected world:", world.get_map().name)
    print("Available maps:", len(maps))
    print("Runtime check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
