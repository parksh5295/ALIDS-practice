#!/usr/bin/env bash
# Enter this project's Pipenv shell even if the parent terminal has stale
# virtualenv environment variables.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
export PATH="${HOME}/.pyenv/bin:${PATH}"

if command -v pyenv >/dev/null 2>&1; then
  eval "$(pyenv init -)"
  eval "$(pyenv virtualenv-init - 2>/dev/null)" || true
fi

unset PIPENV_ACTIVE
export PIPENV_IGNORE_VIRTUALENVS=1

exec pyenv exec pipenv shell
