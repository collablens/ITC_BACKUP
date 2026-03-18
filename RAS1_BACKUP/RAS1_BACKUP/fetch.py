#!/usr/bin/env python3
import firebase_admin
from firebase_admin import credentials, db

# ---- CONFIG ----
SERVICE_ACCOUNT_PATH = "/home/pi/Code/ITCNaanCapture/service_account.json" 
DATABASE_URL = "https://itc-kt-default-rtdb.asia-southeast1.firebasedatabase.app"
FIREBASE_PATH = "IP_ADDRESS/pc1"

OUTPUT_FILES = [
    "upload_gs_cam_url", 
    "upload_16_mp_cam_url"
]

PORT = 2001
ENDPOINT = "/upload"
# ----------------

def main():
    # Init Firebase Admin
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})

    # Read IP from Firebase
    ref = db.reference(FIREBASE_PATH)
    ip_address = ref.get()

    if ip_address:
        # Format as URL
        url = f"http://{ip_address}:{PORT}{ENDPOINT}"

        print(f"📥 Fetched IP from Firebase: {ip_address}")
        print(f"🔗 Formatted URL: {url}")

        # Write URL to both files
        for file_name in OUTPUT_FILES:
            with open(file_name, "w") as f:
                f.write(url)
            print(f"✅ Updated '{file_name}' with URL")

    else:
        print(f"❌ No IP address found at '{FIREBASE_PATH}'")

if __name__ == "__main__":
    main()
