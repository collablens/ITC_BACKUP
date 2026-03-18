#!/usr/bin/env python3
import socket
import firebase_admin
from firebase_admin import credentials, db

# ---- CONFIG ----
SERVICE_ACCOUNT_PATH = "service_account.json"
DATABASE_URL = "https://itc-kt-default-rtdb.asia-southeast1.firebasedatabase.app"
FIREBASE_PATH = "IP_ADDRESS/raspberry1"   # writes here
# ----------------

def get_local_ip() -> str:
    """
    Gets the primary LAN IP of this machine by opening a UDP socket.
    This does NOT send packets; it's used to determine the outbound interface.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Any reachable public IP works; no traffic is actually sent for UDP connect
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()

def main():
    ip = get_local_ip()
    print(f"Detected IP: {ip}")

    # Init Firebase Admin
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})

    # Write IP to /IP_ADDRESS/raspberry1
    ref = db.reference(FIREBASE_PATH)
    ref.set(ip)

    print(f"✅ Wrote '{ip}' to '{FIREBASE_PATH}'")

if __name__ == "__main__":
    main()