.PHONY: help install install-dev eda train evaluate serve test eval-llm eval-providers lint format docker-build docker-run clean
.DEFAULT_GOAL := help

PY      ?= py -3.12
VENV    := .venv
BIN     := $(VENV)/Scripts
IMAGE   := claim-approval-agent
PORT    ?= 8000

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Create venv and install runtime dependencies
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

install-dev:  ## Install runtime + test/eval/notebook tooling
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/python -m ipykernel install --user --name claim-agent --display-name "claim-agent (3.12)"

eda:  ## Run the EDA notebook and write plots to artifacts/plots
	$(BIN)/jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --inplace

train:  ## Train, compare, tune, and register the model
	$(BIN)/python -m src.models.train

evaluate:  ## Cross-validate and write metrics and plots to artifacts/
	$(BIN)/python -m src.models.evaluate

serve:  ## Run the API locally with reload
	$(BIN)/uvicorn src.api.app:app --host 0.0.0.0 --port $(PORT) --reload

test:  ## Run the pytest suite
	$(BIN)/pytest tests/ -v

eval-llm:  ## DeepEval: faithfulness, relevancy, hallucination
	$(BIN)/pytest tests/evaluation/ -v -m llm_eval

eval-providers:  ## Promptfoo: verify the fallback provider matches the primary
	npx promptfoo@latest eval -c evaluation/promptfoo.yaml

lint:  ## Lint and check formatting
	$(BIN)/ruff check src/ tests/
	$(BIN)/ruff format --check src/ tests/

typecheck:  ## Static type check
	$(BIN)/mypy src/ --ignore-missing-imports

format:  ## Apply formatting
	$(BIN)/ruff format src/ tests/

docker-build:  ## Build the container image
	docker build -t $(IMAGE):latest .

docker-run:  ## Run the container on $(PORT)
	docker run --rm -p $(PORT):8000 --env-file .env $(IMAGE):latest

clean:  ## Remove caches and generated plots
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	rm -f artifacts/plots/*.png artifacts/plots/*.html
