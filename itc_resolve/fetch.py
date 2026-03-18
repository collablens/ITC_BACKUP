#!/usr/bin/env python3
import firebase_admin
from firebase_admin import credentials, db

# ---- CONFIG ----
SERVICE_ACCOUNT_PATH = "service_account.json"
DATABASE_URL = "https://itc-kt-default-rtdb.asia-southeast1.firebasedatabase.app"
FIREBASE_PATH = "IP_ADDRESS/raspberry1"
OUTPUT_FILE = "raspberry_ip.txt"
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
        
        # Write URL to text file
        with open(OUTPUT_FILE, "w") as f:
            f.write(url)
        
        print(f"✅ Updated '{OUTPUT_FILE}' with URL: {url}")
    else:
        print(f"❌ No IP address found at '{FIREBASE_PATH}'")

if __name__ == "__main__":
    main()