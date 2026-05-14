# EXACT 2026 — common workflows
# Run `make help` to see all targets.

.DEFAULT_GOAL := help
SHELL := /bin/bash

PY := uv run python
PYTEST := uv run pytest

.PHONY: help install install-all lint format typecheck test test-unit test-int test-e2e \
        api eval smoke train-sft train-grpo docker-api clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime + dev deps via uv
	uv sync --extra dev

install-all:  ## Install with physics + logic + llm + train extras
	uv sync --all-extras

lint:  ## Run ruff lint
	uv run ruff check src tests scripts

format:  ## Auto-format with ruff
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

typecheck:  ## Run mypy
	uv run mypy src

test:  ## Run all tests
	$(PYTEST) -x -v

test-unit:  ## Run unit tests only
	$(PYTEST) -x -v tests/unit

test-int:  ## Run integration tests
	$(PYTEST) -x -v tests/integration

test-e2e:  ## Run end-to-end API tests
	$(PYTEST) -x -v tests/e2e

api:  ## Run FastAPI dev server (reload on change)
	uv run uvicorn exact_agent.api.app:app --reload --host 0.0.0.0 --port 8000

smoke:  ## Hit the API with sample payloads
	$(PY) scripts/smoke_api.py

eval:  ## Run local eval on the holdout split
	$(PY) scripts/run_eval.py --split eval

train-sft:  ## Fine-tune Qwen via Unsloth (LoRA)
	$(PY) -m exact_agent.train.run_sft_qwen --config configs/training/sft_qwen3_8b.yaml

train-grpo:  ## GRPO fine-tune (TRL)
	$(PY) -m exact_agent.train.run_grpo --config configs/training/grpo_qwen3_8b.yaml

docker-api:  ## Build API container image
	docker build -f docker/Dockerfile.api -t exact-api:latest .

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
