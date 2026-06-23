#!/usr/bin/env bash
set -euo pipefail

uv lock --check
uv run ruff check .
uv run pytest -q
