PYTHON ?= python3

.PHONY: help test tools
help:
	@echo "make test                 Run offline integration tests"
	@echo "make tools                List registered construction tools"
	@echo "Use build_map.py / run_tool.py / test_map.py for configured execution."

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover -s tests -v

tools:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) run_tool.py list
