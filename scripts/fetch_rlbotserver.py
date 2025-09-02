#!/usr/bin/env python3
"""Download the RLBotServer binary from the RLBot/core releases."""
import argparse
import os
import platform
import stat
from pathlib import Path

import requests


REPO_API = "https://api.github.com/repos/RLBot/core/releases"


def fetch(tag: str | None) -> Path:
    url = f"{REPO_API}/latest" if tag is None else f"{REPO_API}/tags/{tag}"
    release = requests.get(url, timeout=10).json()
    exe_name = "RLBotServer.exe" if os.name == "nt" else "RLBotServer"
    asset = next((a for a in release.get("assets", []) if a.get("name") == exe_name), None)
    if asset is None:
        raise RuntimeError(f"Could not find {exe_name} in release {release.get('tag_name')}")
    data = requests.get(asset["browser_download_url"], timeout=60).content
    out_path = Path(__file__).resolve().parent.parent / exe_name
    out_path.write_bytes(data)
    out_path.chmod(out_path.stat().st_mode | stat.S_IEXEC)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch RLBotServer binary")
    parser.add_argument("--tag", help="Release tag to download (default: latest)")
    args = parser.parse_args()
    path = fetch(args.tag)
    print(f"Downloaded RLBotServer to {path}")


if __name__ == "__main__":
    main()
