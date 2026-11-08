#!/usr/bin/env python3
"""Check interfaces or run downstream paired tests on a delivered map."""
import argparse
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

from map_generation.city_workflow import read_json
from map_testing.paired import check, load_config, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['doctor', 'run'])
    parser.add_argument('--run', type=Path, required=True, help='Completed construction directory')
    parser.add_argument('--config', type=Path, required=True, help='Downstream adapters and resources')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == 'doctor':
        check(config, read_json(args.run / 'state.json'))
        print('Adapter paths are configured. This does not load models or test simulator compatibility.')
    else:
        print(run(config, args.run))


if __name__ == '__main__':
    main()
