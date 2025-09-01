# Master task list for Codex

0) Repo sanity & Python 3.9

Codex, do this:

Ensure .venv uses Python 3.9.

Confirm env/requirements.txt includes:

- torch==2.2.2 (+ CUDA cu121 install in setup.ps1 where applicable)
- rlbot v5 (tag when available; else temporary develop)
- gymnasium==0.29.1, shimmy==1.3.0, protobuf==3.20.3

Run env/setup.ps1 and verify:

- prints torch 2.2.x, rlbot import OK, gymnasium import OK.

Acceptance test:

```
.\.venv\Scripts\python - << 'PY'
import torch, gymnasium, rlbot, sys
print("OK", torch.__version__, gymnasium.__version__, getattr(rlbot,"__version__","dev"), sys.version)
PY
```

1) Gym→Gymnasium shim

Files:

- src/utils/gym_compat.py (already outlined earlier).

Replace training imports:
`from src.utils.gym_compat import gym, reset_env, step_env`

Acceptance:

`src/training/train.py --dry_run 1` starts, steps env, no banner/warnings, no crashes.

2) RLBot v5 + RLBotPack opponents

Codex:

Add submodule:

```
git submodule add https://github.com/RLBot/RLBotPack vendor/RLBotPack
git submodule update --init --recursive
```

Add configs:

- `configs/rlbot_2v2_nexto.toml`
- `configs/rlbot_2v2_necto.toml`
  (Both: SSLBot×2 vs opponent×2; match_length 5 min; rendering on; instant_start)

Add scripts:

- `scripts/run_rlbot_scrim.bat/.sh` (reads RLBOT_MATCH_CONFIG, defaults to Nexto)
- `scripts/scrim_nexto.bat/.sh`, `scripts/scrim_necto.bat/.sh`

Acceptance:

`scripts\scrim_nexto.bat` launches, your two bots load, match runs.

3) Teamplay rewards (win more in 2v2)

Files:

- `src/training/rewards_teamplay.py` (double-commit penalty, second-man patience, third-man depth, pass attempts, clear-to-safe).

Codex:

Integrate this with your reward function only in 2v2 phases.

Implement ETT (estimated time-to-touch) in your env step or obs builder to detect challenges:

- `my_is_challenging` if ETT < 0.6s
- `both_challenging` if both teammates ETT < 0.8s
- `target_depth_uu` 1200–1800 when last man

Acceptance:

During eval logs, `double_commit_pct` decreases over time when teamplay shaping enabled.

4) Snapshot self-play league (stability + improvement)

Files:

- `src/training/selfplay_league.py` (provided earlier).

Codex:

Save snapshot every `eval_every_updates`.

Sample snapshot opponents in 40–60% of env instances.

Update snapshot Elo from eval results.

Acceptance:

- `models/checkpoints/snapshots/` fills with snapshots.
- `league_meta.json` updates Elo & win counts.

5) KPI logging & eval loop

Files:

- `src/training/eval_kpis.py` (aggregate KPIs).
- `src/eval/scrim_logger.py` for JSONL logs during RLBot scrims (optional).

Codex:

After each eval block (20 games): compute KPIs:
- `on_target_pct`, `aerial_intercept_rate`, `double_commit_pct`, `rotation_violations_pct`, `boost_per_touch`, `fifty_win_rate`, `goal_diff`.

Print concise table and save `eval_logs/eval_<step>.json`.

Promotion gates (2v2):

Promote to next phase when all:
- `double_commit_pct ≤ 0.05` (then ≤ 0.03 at SSL)
- `on_target_pct ≥ 0.45`
- `aerial_intercept_rate ≥ 0.35`
- `goal_diff ≥ +5` over the eval set

6) Export to TorchScript

Codex:

Ensure `src/training/policy.py` defines `build_policy(107, 5, 3)` and TorchScripts cleanly.

Ensure `src/inference/export.py` saves to `models/exported/ssl_policy.ts`.

Acceptance:

```
.\.venv\Scripts\python -m src.inference.export --ckpt models\checkpoints\best.pt --out models\exported\ssl_policy.ts --obs_dim 107
```

7) Proof scrims (show it can win)

Codex:

Ensure `configs/rlbot_proof.toml` exists: SSLBot×2 vs Psyonix Pro×2, unlimited match length (script ends run).

Add `scripts/run_proof_scrim.py` (30-minute run, then terminate RLBot).

Save replay to `proof/` (document how to copy from RL’s replay directory if needed).

Acceptance:

Run proof script, produce replay + a KPI JSON with positive goal diff.

8) CI (3.9) & quality gates

Codex:

`.github/workflows/ci.yml` using `actions/setup-python@v5` with 3.9.

CI steps:

- Install env (CPU).
- Smoke: import torch/rlgym/rlbot/gymnasium; run `train.py --dry_run 1`; export TorchScript.
- Optional: unit tests for obs size parity (107), policy I/O shapes.

Acceptance:

- Green badge; CI duration < 10 min.

9) Submission pack

Codex:

- `apply/README_APPLY.md` (how to run demo).
- `apply/Benchmarks.md` (render from latest eval JSONs).
- `apply/Fair Play & Safety.md`.
- `bot.toml` with `[details]` block (name, author, description).
- `bob.toml` placeholder (we’ll adjust when official template lands).

Acceptance:

One command to bundle:

```
git archive -o submission.zip --format=zip --prefix=RankedSSLRLBOT/ HEAD
```
