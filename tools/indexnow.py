"""Tell IndexNow search engines (Bing, Yandex, Seznam, Naver) that the page changed.

The key is the name of the 32-hex-character ``.txt`` file in ``web/`` (its
content is the key too), as the IndexNow protocol requires. Run after a
deploy that changed the page:

    python tools/indexnow.py

Prints the HTTP status from api.indexnow.org: 200 or 202 means accepted.
Google does not take IndexNow; the owner submits there through Search Console.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

SITE = "https://jetzhu.github.io/MorseCodeAudioCoder/"
WEB = Path(__file__).resolve().parents[1] / "web"


def find_key() -> str:
    keys = [p.stem for p in WEB.glob("*.txt") if re.fullmatch(r"[0-9a-f]{32}", p.stem)]
    if len(keys) != 1:
        raise SystemExit(f"expected exactly one IndexNow key file in {WEB}, found {keys}")
    return keys[0]


def main(urls: list[str] | None = None) -> int:
    key = find_key()
    body = {
        "host": "jetzhu.github.io",
        "key": key,
        "keyLocation": f"{SITE}{key}.txt",
        "urlList": urls or [SITE],
    }
    req = urllib.request.Request(
        "https://api.indexnow.org/indexnow",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.status
    except urllib.error.HTTPError as exc:  # 4xx: key not live yet, bad URL
        status = exc.code
    print(f"IndexNow: HTTP {status} for {len(body['urlList'])} URL(s)")
    return 0 if status in (200, 202) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
