# Contributing

For an issue, include the command, configuration, relevant log excerpt, and software environment needed to reproduce it. Keep API keys in environment variables.

## Code layout

- `map_generation/`: scene representation, geometry, information graph, and workflow.
- `scripts/`: command-line tools, grouped by pipeline stage.
- `map_testing/`: external adapter interfaces and trajectory metrics.
- `configs/workflows/city_tools.json`: tool IDs, entrypoints, and environments.
- `configs/workflows/instructions/`: method, construction, review, and dynamic-policy instructions loaded by the workflow.
- `configs/workflows/prompts/`: common and role-specific agent prompts.

Add tools to the registry and document their inputs and outputs. Model-specific integrations use the interfaces described in [ADAPTERS.md](docs/ADAPTERS.md).

## Tests

```bash
python -m pip install -r requirements/test.txt
make test PYTHON=python
```

Include a focused test for changes to workflow transitions, adapter execution, or scoring. Use synthetic fixtures for offline tests and keep generated runs, datasets, and weights outside the source tree.
