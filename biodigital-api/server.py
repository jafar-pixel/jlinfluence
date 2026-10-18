"""
JL Pulse — BioDigital Content API proxy.

A thin FastAPI server that:
  1. Holds your BioDigital client_id + client_secret server-side (never sent to the browser).
  2. Mints and caches OAuth2 tokens.
  3. Exposes a /api/library endpoint the JL Pulse front-end can call.

Deploy to Vercel as a Python serverless function or run locally:
    export BIODIGITAL_CLIENT_ID=...
    export BIODIGITAL_CLIENT_SECRET=...
    uvicorn server:app --reload --port 8080

Front-end usage:
    fetch('/api/library?search=squat').then(r => r.json())
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

TOKEN_URL = "https://apis.biodigital.com/oauth2/v2/token"
MYHUMAN_URL = "https://apis.biodigital.com/services/v2/content/collections/myhuman"
COLLECTIONS_URL = "https://apis.biodigital.com/services/v2/content/collections/mycollections"

CLIENT_ID = os.environ.get("BIODIGITAL_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("BIODIGITAL_CLIENT_SECRET", "")
DEV_KEY = os.environ.get("BIODIGITAL_DEV_KEY", "ab649fdf7795a30296204cd650e842d386cb32c5")

app = FastAPI(title="JL Pulse — BioDigital API Proxy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://jl-pulse.vercel.app", "http://localhost:5174", "http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---- simple in-memory token cache ----
_token_cache: dict[str, Any] = {"token": None, "expires_at": 0}


def _http_json(req: urllib.request.Request) -> dict:
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def get_token() -> str:
    now = time.time()
    if _token_cache["token"] and _token_cache["expires_at"] - 60 > now:
        return _token_cache["token"]
    if not CLIENT_ID or not CLIENT_SECRET:
        raise HTTPException(500, "BIODIGITAL_CLIENT_ID / BIODIGITAL_CLIENT_SECRET not configured")

    basic = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    body = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "scope": "contentapi",
    }).encode()
    req = urllib.request.Request(
        TOKEN_URL, data=body, method="POST",
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )
    try:
        data = _http_json(req)
    except Exception as e:
        raise HTTPException(502, f"BioDigital token request failed: {e}")
    if "access_token" not in data:
        raise HTTPException(502, f"BioDigital did not return a token: {data}")
    _token_cache["token"] = data["access_token"]
    _token_cache["expires_at"] = now + int(data.get("expires_in", 3600))
    return _token_cache["token"]


def bearer_get(url: str) -> dict:
    token = get_token()
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    })
    try:
        return _http_json(req)
    except urllib.error.HTTPError as e:
        raise HTTPException(e.code, f"BioDigital API error: {e.read().decode()}")
    except Exception as e:
        raise HTTPException(502, f"BioDigital API unreachable: {e}")


def flatten(item: dict) -> dict:
    cid = item.get("content_id", "")
    ctype = item.get("content_type", "")
    if ctype == "bookmark":
        embed = f"https://human.biodigital.com/viewer/?be={cid}&dk={DEV_KEY}"
    else:
        embed = f"https://human.biodigital.com/viewer/?id={cid}&dk={DEV_KEY}"
    flags = item.get("content_flags") or {}
    return {
        "id": cid,
        "title": item.get("content_title", cid),
        "type": ctype,
        "gender": item.get("content_gender"),
        "thumbnail": item.get("content_thumbnail_url"),
        "embed_url": embed,
        "is_animated": flags.get("is_animated", False),
        "is_tour": flags.get("is_tour", False),
        "is_quiz": flags.get("is_quiz", False),
        "accessibility": item.get("content_accessibility", []),
        "authored_date": item.get("content_authored_date"),
    }


@app.get("/api/health")
def health():
    return {"ok": True, "has_credentials": bool(CLIENT_ID and CLIENT_SECRET)}


@app.get("/api/library")
def library(
    search: str | None = Query(None, description="title / id substring"),
    type: str | None = Query(None, pattern="^(module|bookmark)$"),
    gender: str | None = Query(None, pattern="^(male|female)$"),
    animated: bool | None = None,
):
    raw = bearer_get(MYHUMAN_URL).get("myhuman", [])
    items = [flatten(it) for it in raw]
    if search:
        q = search.lower()
        items = [x for x in items if q in x["title"].lower() or q in x["id"].lower()]
    if type:
        items = [x for x in items if x["type"] == type]
    if gender:
        items = [x for x in items if x["gender"] == gender]
    if animated is not None:
        items = [x for x in items if x["is_animated"] == animated]
    return {"count": len(items), "items": items}


@app.get("/api/collections")
def collections():
    data = bearer_get(COLLECTIONS_URL)
    return data


@app.get("/api/collections/{col_id}/content")
def collection_content(col_id: str):
    url = f"https://apis.biodigital.com/services/v2/content/collections/mycollections/{col_id}/content_list"
    return bearer_get(url)
