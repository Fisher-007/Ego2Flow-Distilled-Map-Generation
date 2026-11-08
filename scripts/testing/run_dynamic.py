"""Run the installed learned-policy adapter during construction dynamic review."""
import argparse
from pathlib import Path

from map_testing.dynamic import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory for this attempt')
    args = parser.parse_args()
    print(run(args.run, args.request, args.output))


if __name__ == '__main__':
    main()
