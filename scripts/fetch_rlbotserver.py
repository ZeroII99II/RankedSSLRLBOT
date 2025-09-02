#!/usr/bin/env python3
"""Download the RLBotServer binary from the RLBot/core releases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
import urllib.request


REPO_API = "https://api.github.com/repos/RLBot/core/releases"


def _get_asset_url(tag: str | None) -> str:
    """Return the download URL for the RLBotServer asset."""

    url = f"{REPO_API}/latest" if tag is None else f"{REPO_API}/tags/{tag}"
    with urllib.request.urlopen(url) as resp:  # nosec: B310 - trusted host
        data = json.load(resp)

    asset_name = "RLBotServer.exe" if os.name == "nt" else "RLBotServer"
    for asset in data.get("assets", []):
        if asset.get("name") == asset_name:
            return asset.get("browser_download_url", "")
    raise RuntimeError(f"Asset {asset_name} not found in release")


def _download(url: str, dest: Path) -> None:
    """Download file at ``url`` to ``dest`` and make it executable."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:  # nosec: B310
        fh.write(resp.read())
    if os.name != "nt":
        dest.chmod(dest.stat().st_mode | stat.S_IEXEC)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch RLBotServer binary")
    parser.add_argument("--tag", help="Release tag to download (default: latest)")
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
        url = _get_asset_url(args.tag)
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


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

