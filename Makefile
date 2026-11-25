UV ?= uv
# Keep the project environment isolated from site-packages exported globally (e.g. ROS).
unexport PYTHONPATH

.PHONY: install lint format typecheck test cov check serve clean

install:
	$(UV) sync

lint:
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format:
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

typecheck:
	$(UV) run mypy

test:
	$(UV) run pytest

cov:
	$(UV) run pytest --cov --cov-report=term-missing

check: lint typecheck test

serve:
	$(UV) run ctgov-extract serve --host 127.0.0.1 --port 8000

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
