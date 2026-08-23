.PHONY: install run debug clean lint lint-strict test

install:
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
