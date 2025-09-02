#!/usr/bin/env bash
set -euo pipefail
. .venv/bin/activate
python run.py "$@"
