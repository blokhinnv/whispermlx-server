.PHONY: lint format typecheck test run

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy src/whispermlx_server

test:
	uv run pytest

run:
	uv run whispermlx-server
