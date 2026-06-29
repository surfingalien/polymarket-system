"""
Cloud persistence for bot state — Supabase backend.

Provides a simple get/put interface over a single `bot_state` key-value table.
Falls back silently if Supabase credentials are not configured.

Supabase setup (run once in the Supabase SQL editor):
------------------------------------------------------
    CREATE TABLE bot_state (
        key        TEXT PRIMARY KEY,
        value      JSONB NOT NULL,
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );
    -- Allow service-role key full access (no extra RLS policy needed when
    -- you use the service_role key and RLS is disabled, OR enable RLS and
    -- add a permissive policy if you prefer the anon key):
    ALTER TABLE bot_state DISABLE ROW LEVEL SECURITY;

Streamlit secrets (Settings → Secrets):
----------------------------------------
    SUPABASE_URL = "https://xxxxxxxxxxxx.supabase.co"
    SUPABASE_KEY = "your-service-role-or-anon-key"
"""
from __future__ import annotations

import os

import structlog

log = structlog.get_logger(__name__)

# Once a connection clearly can't work (bad URL / DNS), stop retrying every cycle
# so we don't spam the logs. Reset only on process restart.
_DISABLED_REASON: str | None = None


def _looks_like_supabase_url(url: str) -> bool:
    return url.startswith("https://") and ".supabase.co" in url and " " not in url.strip()


def _client():
    global _DISABLED_REASON
    if _DISABLED_REASON is not None:
        return None
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        return None
    if not _looks_like_supabase_url(url):
        _DISABLED_REASON = (
            f"SUPABASE_URL doesn't look valid ({url!r}). It must be exactly "
            "https://<project-ref>.supabase.co (no path, no trailing slash). "
            "Cloud persistence disabled for this session."
        )
        log.warning("supabase_url_invalid", detail=_DISABLED_REASON)
        return None
    try:
        from supabase import create_client  # type: ignore
        return create_client(url, key)
    except Exception as exc:
        log.warning("supabase_client_init_failed", error=str(exc))
        return None


def _disable(reason: str) -> None:
    """Turn off cloud persistence for the session after an unrecoverable error
    (e.g. DNS 'Name or service not known' = wrong SUPABASE_URL host)."""
    global _DISABLED_REASON
    if _DISABLED_REASON is None:
        _DISABLED_REASON = reason
        log.warning("cloud_disabled", detail=reason)


def is_available() -> bool:
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY"))


def cloud_load(key: str) -> dict | None:
    """Fetch a JSON blob by key. Returns None if not found or unavailable."""
    c = _client()
    if c is None:
        return None
    try:
        res = c.table("bot_state").select("value").eq("key", key).limit(1).execute()
        if res and res.data:
            return res.data[0]["value"]
    except Exception as exc:
        _es = str(exc)
        log.warning("cloud_load_failed", key=key, error=_es)
        if "Name or service not known" in _es or "getaddrinfo" in _es or "Errno -2" in _es:
            _disable(f"Supabase host unreachable (DNS): {_es}. Check SUPABASE_URL.")
    return None


def cloud_save(key: str, value: dict) -> bool:
    """Upsert a JSON blob. Returns True on success."""
    c = _client()
    if c is None:
        return False
    try:
        import time as _time
        c.table("bot_state").upsert({
            "key": key,
            "value": value,
            "updated_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        }).execute()
        log.info("cloud_save_ok", key=key)
        return True
    except Exception as exc:
        _es = str(exc)
        log.warning("cloud_save_failed", key=key, error=_es)
        if "Name or service not known" in _es or "getaddrinfo" in _es or "Errno -2" in _es:
            _disable(f"Supabase host unreachable (DNS): {_es}. Check SUPABASE_URL.")
        return False
