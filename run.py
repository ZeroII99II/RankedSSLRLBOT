#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path


def main():
    match = sys.argv[1] if len(sys.argv) > 1 else "rlbot/match_nexto.toml"
    exe_name = "RLBotServer.exe" if os.name == "nt" else "RLBotServer"
    exe_path = Path(__file__).resolve().parent / exe_name
    if not exe_path.exists():
        raise FileNotFoundError(f"{exe_name} not found. Run scripts/fetch_rlbotserver.py first.")
    subprocess.run([str(exe_path), "-m", match], check=True)


if __name__ == "__main__":
    main()
