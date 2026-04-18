SHELL := /bin/bash
ROOT  := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

REMOTE_HOST        ?= pp-sam
REMOTE_PATH        ?= projects/sam

RSYNC               = rsync --archive --verbose --compress --rsh='ssh -o ClearAllForwardings=yes'

CONDA_ENV_NAME      = sam

# -----------------------------------------------------------------------------
# notebook
# -----------------------------------------------------------------------------

.DEFAULT_GOAL = env-shell

# -----------------------------------------------------------------------------
# conda environment
# -----------------------------------------------------------------------------

.PHONY: env-init-conda
env-init-conda:
	@conda create --yes --copy --name "$(CONDA_ENV_NAME)" \
		conda-forge::python=3.12.12 \
		conda-forge::poetry=2.2.1 \
		conda-forge::ffmpeg==7.1.1 \
		nvidia::cuda=13.2.1

.PHONY: env-init-poetry
env-init-poetry:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		poetry install --no-root --no-directory

.PHONY: env-update
env-update:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		poetry update

.PHONY: env-list
env-list:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		poetry show

.PHONY: env-remove
env-remove:
	@conda env remove --yes --name "$(CONDA_ENV_NAME)"

.PHONY: env-shell
env-shell:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" --cwd "$(ROOT)"\
		bash

.PHONY: env-info
env-info:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		conda info

# -----------------------------------------------------------------------------
# rsync push
# -----------------------------------------------------------------------------

.PHONY: rsync-push
rsync-push:
	@$(RSYNC) \
		--exclude='/.git' \
		--exclude='/.vscode' \
		--exclude='/.idea' \
		--exclude='/work/*' \
		--exclude='*.log' \
		--exclude='__pycache__' \
		--exclude='.pytest_cache' \
		--exclude='.ipynb_checkpoints' \
		'$(ROOT)/' \
		'$(REMOTE_HOST):$(REMOTE_PATH)'

# -----------------------------------------------------------------------------
# rsync pull
# -----------------------------------------------------------------------------

.PHONY: rsync-pull
rsync-pull:
	@$(RSYNC) \
		--exclude='/.git' \
		--exclude='/.idea' \
		--exclude='/work/*' \
		--exclude='*.log' \
		--exclude='__pycache__' \
		--exclude='.pytest_cache' \
		--exclude='.ipynb_checkpoints' \
		'$(REMOTE_HOST):$(REMOTE_PATH)' \
		'$(ROOT)/'
