#!/usr/bin/env python3
"""
JL Pulse — BioDigital Content API fetcher.

Usage:
    export BIODIGITAL_CLIENT_ID=...
    export BIODIGITAL_CLIENT_SECRET=...
    python fetch_library.py            # writes library.json
    python fetch_library.py --search "squat"
    python fetch_library.py --filter module     # only full models, not bookmarks

Output: library.json with a flattened list ready to drop into MOVEMENTS config.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

TOKEN_URL = "https://apis.biodigital.com/oauth2/v2/token"
MYHUMAN_URL = "https://apis.biodigital.com/services/v2/content/collections/myhuman"
COLLECTIONS_URL = "https://apis.biodigital.com/services/v2/content/collections/mycollections"


def get_access_token(client_id: str, client_secret: str) -> str:
    """OAuth2 Client Credentials flow — returns a short-lived bearer token."""
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    body = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "scope": "contentapi",
    }).encode()
    req = urllib.request.Request(
        TOKEN_URL,
        data=body,
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
    if "access_token" not in data:
        raise RuntimeError(f"Token request failed: {data}")
    return data["access_token"]


def get_json(url: str, token: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def fetch_library(token: str) -> list[dict]:
    data = get_json(MYHUMAN_URL, token)
    return data.get("myhuman", [])


def fetch_collections(token: str) -> list[dict]:
    data = get_json(COLLECTIONS_URL, token)
    return data.get("mycollections", [])


def flatten(item: dict, dev_key: str) -> dict:
    """Convert raw content object to a JL-Pulse-ready entry."""
    cid = item.get("content_id", "")
    ctype = item.get("content_type", "")
    # Build a clean embed URL regardless of the verbose server URL.
    if ctype == "bookmark":
        embed = f"https://human.biodigital.com/viewer/?be={cid}&dk={dev_key}"
    else:
        embed = f"https://human.biodigital.com/viewer/?id={cid}&dk={dev_key}"
    return {
        "id": cid,
        "title": item.get("content_title", cid),
        "type": ctype,
        "gender": item.get("content_gender"),
        "thumbnail": item.get("content_thumbnail_url"),
        "embed_url": embed,
        "is_animated": (item.get("content_flags") or {}).get("is_animated", False),
        "is_tour": (item.get("content_flags") or {}).get("is_tour", False),
        "accessibility": item.get("content_accessibility", []),
        "authored_date": item.get("content_authored_date"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--search", help="case-insensitive title filter")
    p.add_argument("--filter", choices=["module", "bookmark"], help="content_type filter")
    p.add_argument("--gender", choices=["male", "female"], help="gender filter")
    p.add_argument("--dev-key", default=os.environ.get("BIODIGITAL_DEV_KEY", "ab649fdf7795a30296204cd650e842d386cb32c5"))
    p.add_argument("--out", default="library.json")
    p.add_argument("--client-id", default=os.environ.get("BIODIGITAL_CLIENT_ID"))
    p.add_argument("--client-secret", default=os.environ.get("BIODIGITAL_CLIENT_SECRET"))
    args = p.parse_args()

    if not args.client_id or not args.client_secret:
        print("ERROR: set BIODIGITAL_CLIENT_ID and BIODIGITAL_CLIENT_SECRET", file=sys.stderr)
        print("  Find them at https://developer.biodigital.com under your app credentials.", file=sys.stderr)
        return 2

    print("→ Requesting OAuth2 token…", file=sys.stderr)
    token = get_access_token(args.client_id, args.client_secret)
    print(f"  Token acquired ({len(token)} chars)", file=sys.stderr)

    print("→ Fetching /myhuman…", file=sys.stderr)
    raw = fetch_library(token)
    print(f"  {len(raw)} items returned", file=sys.stderr)

    items = [flatten(it, args.dev_key) for it in raw]

    if args.search:
        q = args.search.lower()
        items = [it for it in items if q in it["title"].lower() or q in it["id"].lower()]
    if args.filter:
        items = [it for it in items if it["type"] == args.filter]
    if args.gender:
        items = [it for it in items if it["gender"] == args.gender]

    # Try collections too (optional — ignores failures)
    try:
        cols = fetch_collections(token)
        print(f"→ Also fetched {len(cols)} collections", file=sys.stderr)
    except Exception as e:
        cols = []
        print(f"  (collections fetch skipped: {e})", file=sys.stderr)

    out = {
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "count": len(items),
        "items": items,
        "collections": cols,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"✓ Wrote {args.out} ({len(items)} items)", file=sys.stderr)

    # Pretty preview
    for it in items[:10]:
        thumb = "🎬" if it["is_animated"] else "🧍"
        print(f"  {thumb} {it['type']:8} {it['id']:55}  {it['title']}")
    if len(items) > 10:
        print(f"  … and {len(items) - 10} more", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
