.DEFAULT_GOAL := install

UV_SYNC_INSTALL_ARGS := --all-extras --group dev

.PHONY: all install install-all check
all: install check

install:
	uv lock --upgrade && uv sync $(UV_SYNC_INSTALL_ARGS)

install-all:
	uv lock --upgrade && uv sync --all-extras --all-groups

check:
ifeq ($(OS),Windows_NT)
	uv run ruff check --no-cache --fix .
	uv run ruff format --no-cache .
	uv run pyright
else
	uv run ruff check --no-cache --fix .
	uv run ruff format --no-cache .
	uv run pyright
endif