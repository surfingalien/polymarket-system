#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Generate RSA-2048 key pair for Kalshi API authentication.

Run once:
    python scripts/setup_kalshi_keys.py

Then upload keys/kalshi_public.pem to:
    https://kalshi.com/profile/api (or demo.kalshi.co equivalent)

Copy the key ID shown in the dashboard into your .env as KALSHI_API_KEY_ID.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def generate_kalshi_keys(output_dir: str = "keys") -> None:
    keys_dir = Path(output_dir)
    keys_dir.mkdir(exist_ok=True)

    priv_path = keys_dir / "kalshi_private.pem"
    pub_path = keys_dir / "kalshi_public.pem"

    if priv_path.exists():
        overwrite = input(f"{priv_path} already exists. Overwrite? [y/N] ").strip().lower()
        if overwrite != "y":
            print("Aborted.")
            sys.exit(0)

    # Generate RSA-2048 private key (Kalshi requirement: RSA ≥ 2048 bit)
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Write private key (PEM, unencrypted — protect with OS file permissions)
    with open(priv_path, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    os.chmod(priv_path, 0o600)

    # Write public key
    public_key = private_key.public_key()
    with open(pub_path, "wb") as f:
        f.write(
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )

    print(f"\n✓ Private key saved to: {priv_path}  (chmod 600 applied)")
    print(f"✓ Public key saved to:  {pub_path}")
    print("\nNext steps:")
    print(f"  1. Go to https://kalshi.com/profile/api")
    print(f"  2. Add a new API key, paste the contents of {pub_path}")
    print(f"  3. Copy the returned Key ID into your .env:")
    print(f"     KALSHI_API_KEY_ID=<key-id-from-dashboard>")
    print(f"     KALSHI_PRIVATE_KEY_PATH={priv_path}")
    print()

    # Print public key for copy-paste
    print("=== PUBLIC KEY (paste this into Kalshi dashboard) ===")
    print(pub_path.read_text())


if __name__ == "__main__":
    generate_kalshi_keys()
