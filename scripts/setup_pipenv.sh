#!/usr/bin/env bash
# ALIDS-practice: pyenv + pipenv bootstrap (Bash: Linux, macOS, Git Bash, WSL).
# Usage: bash scripts/setup_pipenv.sh
#   or:  chmod +x scripts/setup_pipenv.sh && ./scripts/setup_pipenv.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REQ_FILE="${PROJECT_ROOT}/requirements.txt"

PYTHON_VERSION="${PYTHON_VERSION:-3.11.1}"
PIPENV_VERSION="${PIPENV_VERSION:-2024.1.0}"

export PIPENV_TIMEOUT="${PIPENV_TIMEOUT:-100}"
export PIPENV_MAX_RETRIES="${PIPENV_MAX_RETRIES:-5}"
export PIPENV_SKIP_LOCK="${PIPENV_SKIP_LOCK:-1}"

# Set AGGRESSIVE_CLEAN=1 to also wipe ~/.local/share/virtualenvs and global pip/pipenv caches (dangerous on shared machines).
AGGRESSIVE_CLEAN="${AGGRESSIVE_CLEAN:-0}"

echo "[0] Cleaning existing project pipenv / Pipfile..."
cd "${PROJECT_ROOT}"
export PATH="${HOME}/.pyenv/bin:${PATH}"
if command -v pyenv >/dev/null 2>&1; then
  eval "$(pyenv init -)" 2>/dev/null || true
  pyenv exec pipenv --rm 2>/dev/null || true
fi
rm -f "${PROJECT_ROOT}/Pipfile" "${PROJECT_ROOT}/Pipfile.lock"

if [[ "${AGGRESSIVE_CLEAN}" == "1" ]]; then
  echo "    (AGGRESSIVE_CLEAN) removing global virtualenvs and pip caches..."
  rm -rf "${HOME}/.local/share/virtualenvs" || true
  rm -rf "${HOME}/.cache/pipenv" "${HOME}/.cache/pip" || true
else
  echo "    (skip global venv wipe; set AGGRESSIVE_CLEAN=1 to mimic full reset)"
fi
echo "    Done."
echo ""

# 1. pyenv on PATH
export PATH="${HOME}/.pyenv/bin:${PATH}"

if ! command -v pyenv >/dev/null 2>&1; then
  echo "[1] Installing pyenv..."
  curl -sS https://pyenv.run | bash
fi

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init - 2>/dev/null)" || true

# 2. Install Python if missing
if ! pyenv versions --bare | grep -qx "${PYTHON_VERSION}"; then
  echo "[2] Installing Python ${PYTHON_VERSION}..."
  pyenv install "${PYTHON_VERSION}"
fi

# 3. Project-local Python
cd "${PROJECT_ROOT}"
pyenv local "${PYTHON_VERSION}"

# 4. Remove pipenv venv again after pyenv local (same as reference)
if pyenv exec pipenv --venv >/dev/null 2>&1; then
  echo "[3] Removing pipenv env after pyenv local..."
  pyenv exec pipenv --rm || true
fi

# 5. pip + pipenv inside this pyenv
echo "[4] Installing pipenv ${PIPENV_VERSION} into pyenv Python..."
export PIP_CONSTRAINT=
pyenv exec python -m pip install --upgrade pip
pyenv exec python -m pip install "pipenv==${PIPENV_VERSION}"

# 6. Create pipenv environment (no lock yet)
echo "[5] Creating pipenv environment..."
pyenv exec pipenv --python "${PYTHON_VERSION}" install --skip-lock

# 7. Install from requirements.txt
if [[ -f "${REQ_FILE}" ]]; then
  echo "[6] Installing from ${REQ_FILE}..."
  pyenv exec pipenv install -r "${REQ_FILE}"
else
  echo "ERROR: requirements file not found: ${REQ_FILE}"
  exit 1
fi

# 8. Optional dependency check
echo ""
echo "[7] pipdeptree (optional conflict hints)..."
if pyenv exec pipenv run python -m pip install --quiet pipdeptree 2>/dev/null; then
  TMP_OUT="$(mktemp)"
  pyenv exec pipenv run pipdeptree --warn silence > "${TMP_OUT}" 2>/dev/null || true
  if grep -iE 'numpy|urllib3' "${TMP_OUT}" > /tmp/alids_pipdeptree_hints.txt 2>/dev/null; then
    echo "    Hints (numpy/urllib3 lines from pipdeptree):"
    cat /tmp/alids_pipdeptree_hints.txt || true
  else
    echo "    No numpy/urllib3 lines in pipdeptree output (or pipdeptree skipped)."
  fi
  rm -f "${TMP_OUT}"
  pyenv exec pipenv run pipdeptree > "${PROJECT_ROOT}/pip_dependency_tree.txt" 2>/dev/null || true
  echo "    Full tree saved to: ${PROJECT_ROOT}/pip_dependency_tree.txt"
else
  echo "    pipdeptree install skipped."
fi

echo ""
echo "Done. Pipenv environment is ready."
echo "  Shell:  cd ${PROJECT_ROOT} && pyenv exec pipenv shell"
echo "  Run:    pyenv exec pipenv run python train_ids.py --config configs/multi_layer_perceptron.yaml"
