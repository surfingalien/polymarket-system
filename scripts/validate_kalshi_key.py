#!/usr/bin/env python3
"""
Quick validator for your Kalshi RSA private key.

Usage:
    export KALSHI_API_KEY="your-key-id-here"
    export KALSHI_PRIVATE_KEY_PEM="$(cat keys/kalshi_private.pem)"
    python scripts/validate_kalshi_key.py

Or point directly at a PEM file:
    python scripts/validate_kalshi_key.py keys/kalshi_private.pem
"""
from __future__ import annotations

import asyncio
import base64
import os
import sys
import time


def _check_parse(pem: str):
    from cryptography.hazmat.primitives import serialization

    try:
        key = serialization.load_pem_private_key(pem.encode(), password=None)
        key_type = type(key).__name__
        key_bits = key.key_size if hasattr(key, "key_size") else "?"
        print(f"  ✅ PEM parsed OK — {key_type}, {key_bits}-bit")
        return key
    except Exception as e:
        print(f"  ❌ PEM parse failed: {e}")
        sys.exit(1)


def _check_sign(key, key_id: str) -> dict:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    ts_ms = str(int(time.time() * 1000))
    msg = ts_ms + "GET" + "/trade-api/v2/markets"
    sig = key.sign(
        msg.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    sig_b64 = base64.b64encode(sig).decode()
    headers = {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": sig_b64,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "Content-Type": "application/json",
    }
    print(f"  ✅ RSA-PSS signing works — sig length {len(sig_b64)} chars")
    return headers


async def _check_api(headers: dict):
    import httpx

    url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params={"limit": 5, "status": "open"}, headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        n = len(data.get("markets", []))
        print(f"  ✅ API call succeeded — {n} markets returned (200 OK)")
    elif resp.status_code == 401:
        print(f"  ❌ 401 Unauthorized — key ID or signature mismatch")
        print(f"     Response: {resp.text[:300]}")
        sys.exit(1)
    elif resp.status_code == 403:
        print(f"  ❌ 403 Forbidden — key exists but lacks market-read permission")
        sys.exit(1)
    else:
        print(f"  ⚠️  Unexpected status {resp.status_code}: {resp.text[:200]}")


def main():
    print("=== Kalshi key validator ===\n")

    # -- get PEM --
    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path) as f:
            pem = f.read().strip()
        print(f"PEM source: {path}")
    else:
        pem = os.environ.get("KALSHI_PRIVATE_KEY_PEM", "").strip()
        if not pem:
            print("ERROR: set KALSHI_PRIVATE_KEY_PEM env var or pass path as argument")
            sys.exit(1)
        print("PEM source: $KALSHI_PRIVATE_KEY_PEM")

    key_id = os.environ.get("KALSHI_API_KEY", "").strip()
    if not key_id:
        print("WARNING: KALSHI_API_KEY not set — will test parse+sign only, not live API")

    print()
    print("Step 1: parse PEM")
    key = _check_parse(pem)

    print("Step 2: test signing")
    headers = _check_sign(key, key_id or "test-key-id")

    if key_id:
        print("Step 3: live API call to trading-api.kalshi.com")
        asyncio.run(_check_api(headers))
    else:
        print("Step 3: skipped (no KALSHI_API_KEY)")

    print("\n✅ Validation complete")


if __name__ == "__main__":
    main()
