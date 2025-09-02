#!/usr/bin/env python3
"""Fetch the latest RLBotServer release.

Downloads the latest RLBotServer binary from the RLBot/core GitHub
repository and places it in the repository root (or a user-specified
location). The binary is required for running RLBot locally.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
import urllib.request

API_URL = "https://api.github.com/repos/RLBot/core/releases/latest"


def _get_asset_url() -> str:
    """Return the download URL for the RLBotServer asset."""
    with urllib.request.urlopen(API_URL) as resp:  # nosec: B310 - trusted host
        data = json.load(resp)
    asset_name = "RLBotServer.exe" if os.name == "nt" else "RLBotServer"
    for asset in data.get("assets", []):
        if asset.get("name") == asset_name:
            return asset.get("browser_download_url", "")
    raise RuntimeError(f"Asset {asset_name} not found in latest release")


def _download(url: str, dest: Path) -> None:
    """Download file at ``url`` to ``dest`` and make it executable."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:  # nosec B310
        fh.write(resp.read())
    if os.name != "nt":
        mode = dest.stat().st_mode
        dest.chmod(mode | stat.S_IEXEC)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch RLBotServer binary")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Destination directory for the binary (default: repo root)",
    )
    args = parser.parse_args(argv)
    filename = "RLBotServer.exe" if os.name == "nt" else "RLBotServer"
    dest_path = args.dest / filename

    try:
        url = _get_asset_url()
    except Exception as exc:  # pragma: no cover - network error handling
        print(f"Failed to determine download URL: {exc}", file=sys.stderr)
        return 1

    print(f"Downloading {url}\n -> {dest_path}")
    try:
        _download(url, dest_path)
    except Exception as exc:  # pragma: no cover - network error handling
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

