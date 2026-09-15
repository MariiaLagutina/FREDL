UV_CACHE_DIR := $(HOME)/goinfre/uv
HF_HOME := $(HOME)/goinfre/huggingface
TRANSFORMERS_CACHE := $(HOME)/goinfre/huggingface

export UV_CACHE_DIR
export HF_HOME
export TRANSFORMERS_CACHE

.PHONY: install run debug clean lint lint-strict test prepare-cache

prepare-cache:
	mkdir -p $(HOME)/goinfre/uv
	mkdir -p $(HOME)/goinfre/huggingface
	mkdir -p $(HOME)/goinfre/rag_venv
	mkdir -p $(HOME)/.cache
	rm -rf $(HOME)/.cache/uv
	rm -rf $(HOME)/.cache/huggingface
	ln -sfn $(HOME)/goinfre/uv $(HOME)/.cache/uv
	ln -sfn $(HOME)/goinfre/huggingface $(HOME)/.cache/huggingface
	ln -sfn $(HOME)/goinfre/rag_venv .venv

install: prepare-cache
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache
	rm -rf .pytest_cache

lint:
	uv run flake8 . --exclude=.venv,data,.local
	uv run mypy . \
		--exclude '^(\.venv|data|\.local)/' \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 . --exclude=.venv,data,.local
	uv run mypy . --strict --exclude '^(\.venv|data|\.local)/'

test:
	uv run pytest
