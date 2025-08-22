.DEFAULT_GOAL := install

UV_SYNC_INSTALL_ARGS := --all-extras --all-groups

.PHONY: all install check
all: install check

install:
	uv lock --upgrade && uv sync $(UV_SYNC_INSTALL_ARGS)

check:
ifeq ($(OS),Windows_NT)
	.venv\Scripts\activate && ruff check --no-cache --fix . && ruff format --no-cache .
else
	. .venv/bin/activate && ruff check --no-cache --fix . && ruff format --no-cache .
endif