"""Command-line adapter templates for paired capture and planning inference."""
import argparse
import json
from pathlib import Path


def prepare_pair(request, preparation_config, output_directory):
    """Capture matched real/simulated inputs and a separate future reference.

    Preserve source calibration, timing, and actor context. Manage the simulator
    on request.resources and save capture evidence. See docs/ONEVL_EXAMPLE.md.
    """
    raise NotImplementedError('Implement source preparation and matching native capture here.')


def predict(request, model_config, output_directory):
    """Return normalized trajectories and raw outputs for request.inputs.

    Use the configured model and settings for both image sources. Record
    invalid predictions with their error messages.
    """
    raise NotImplementedError('Implement predict with the configured planning model.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    config = json.loads(Path(request['adapter_config']).read_text())
    function = {'prepare_pair': prepare_pair, 'predict': predict}[request['kind']]
    result = function(request, config, args.output.parent)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
