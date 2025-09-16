.PHONY: czc

czc:
	cz c && pre-commit run --all-files

.PHONY: deps

deps:
	python -m pip install poetry
	python -m pip install commitizen
	python -m pip install pre-commit
	poetry install
	poetry run pre-commit install
	poetry env info
