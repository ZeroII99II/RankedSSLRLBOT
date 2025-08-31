set -e
python3.10 -m venv .venv
. .venv/bin/activate
pip install -U pip wheel
pip install -r env/requirements.txt
python - << 'PY'
import torch, rlgym, rlbot
print("OK:", torch.__version__)
PY