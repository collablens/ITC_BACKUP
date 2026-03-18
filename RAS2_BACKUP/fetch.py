#!/usr/bin/env python3
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import firebase_admin
from firebase_admin import credentials, db

# ---- CONFIG ----
SERVICE_ACCOUNT_PATH = "/home/pi/Code/ITCNaanCapture/service_account.json"
DATABASE_URL = "https://itc-kt-default-rtdb.asia-southeast1.firebasedatabase.app"
FIREBASE_PATH = "IP_ADDRESS/pc1"

PORT = 2000

# Per-file endpoint mapping (✅ what you asked)
FILE_TO_ENDPOINT = {
    "upload_16mp_cam_url": "/upload",
    "upload_gs_cam_url": "/uploadLow",
}
# ----------------


def normalize_host(value: str) -> str:
    """
    Accepts:
      - "192.168.1.64"
      - "192.168.1.64:2000"
      - "http://192.168.1.64:2000/something"
    Returns: "192.168.1.64"
    """
    s = str(value).strip()
    if not s:
        return ""
    if "://" in s:
        p = urlparse(s)
        host = p.hostname or ""
    else:
        # strip path if any and strip port if present
        s = s.split("/", 1)[0]
        host = s.split(":", 1)[0]
    return host.strip()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)  # atomic on same filesystem
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def main():
    # Init Firebase Admin
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})

    # Read IP from Firebase
    ref = db.reference(FIREBASE_PATH)
    raw = ref.get()

    host = normalize_host(raw)
    if not host:
        print(f"❌ No valid IP/host found at '{FIREBASE_PATH}'. Got: {raw!r}")
        return

    print(f"📥 Fetched host from Firebase: {host}")

    for file_name, endpoint in FILE_TO_ENDPOINT.items():
        url = f"http://{host}:{PORT}{endpoint}"
        atomic_write_text(Path(file_name), url)
        print(f"✅ Updated '{file_name}' → {url}")


if __name__ == "__main__":
    main()
