# Justfile for common tasks. `just` with no args lists recipes.
# Args pass through as "$@" so just's {{…}} never fights Copier's.
set positional-arguments

[private]
default:
	@just --list

# Install/refresh .venv from uv.lock (direnv does this on cd, too)
setup:
	uv sync

# Run the console script
run *args:
	uv run feelings "$@"

# Ruff lint + format check
lint:
	uv run ruff check .
	uv run ruff format --check .

# Ruff autofix + format
fmt:
	uv run ruff check --fix .
	uv run ruff format .

# ty, via uv (ty version comes from uv.lock)
typecheck:
	uv check

# pytest; extra args pass through (just test -k greet)
test *args:
	uv run pytest "$@"

# pytest with coverage
cov:
	uv run pytest --cov --cov-report=term-missing

# Everything CI runs
all: lint typecheck test
