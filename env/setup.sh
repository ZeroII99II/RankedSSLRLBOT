set -e
python3.9 -m venv .venv
. .venv/bin/activate
pip install -U pip wheel
pip install -r env/requirements.txt
python - << 'PY'
import torch, gymnasium, rlbot
print("OK", torch.__version__, gymnasium.__version__, getattr(rlbot, "__version__", "dev"))
PY