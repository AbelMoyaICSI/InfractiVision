#!/usr/bin/env python3
"""upload_release_asset.py — (LEGACY) Sube zips de release a GCS público (releases/latest).
Solo se usa si zip >2GB (límite GitHub Releases). Con la optimización actual (<2GB)
no es necesario; se mantiene como fallback manual.

Los assets >2GB no caben en GitHub Releases; se sirven desde el bucket
Firebase `infractivision-e8c03.firebasestorage.app`. Los zips se suben con
ACL `publicRead` y quedan disponibles en:
    https://storage.googleapis.com/infractivision-e8c03.firebasestorage.app/releases/latest/<name>

Uso:
    FIREBASE_SA_JSON='{...}' python scripts/upload_release_asset.py dist/*.zip
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BUCKET = "infractivision-e8c03.firebasestorage.app"
PROJECT = "infractivision-e8c03"


def main() -> int:
    sa = os.getenv("FIREBASE_SA_JSON")
    if not sa:
        print("ERROR: falta env FIREBASE_SA_JSON", file=sys.stderr)
        return 1
    paths = [Path(a) for a in sys.argv[1:] if Path(a).exists()]
    if not paths:
        print("ERROR: no hay archivos para subir (dist/*.zip)", file=sys.stderr)
        return 1

    from google.cloud import storage
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_info(json.loads(sa))
    client = storage.Client(project=PROJECT, credentials=creds)
    bucket = client.bucket(BUCKET)

    ok = True
    for path in paths:
        dest = f"releases/latest/{path.name}"
        blob = bucket.blob(dest)
        blob.upload_from_filename(str(path), predefined_acl="publicRead")
        print(f"GCS OK: {path.name} -> {blob.public_url}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())