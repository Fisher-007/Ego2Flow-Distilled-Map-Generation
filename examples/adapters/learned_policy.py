"""Command-line adapter template for a learned driving policy."""
import argparse
import json
from pathlib import Path


def run_policy(request, model_config, output_directory):
    """Run the configured policy and return the dynamic-policy result.

    Implement model loading, sensors, sustained demand, and control logging.
    Release created actors and restore world settings on exit. See
    docs/ADAPTERS.md and docs/LEAD_EXAMPLE.md for the interface.
    """
    raise NotImplementedError('Implement run_policy with the configured learned driving policy.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if request['kind'] != 'dynamic_policy':
        raise ValueError('Expected a dynamic_policy request')
    config = json.loads(Path(request['adapter_config']).read_text())
    result = run_policy(request, config, args.output.parent)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
