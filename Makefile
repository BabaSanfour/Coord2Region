PYTHON ?= python3

.PHONY: schema
schema:
	PYTHONPATH=. $(PYTHON) scripts/generate_schema.py
