import http.cookiejar
import json
import ssl
import sys
import urllib.request

TOKEN = "2eb9977615ff4fa0"
BASE_HOST = "ufo-43751a60-b805-4610-8c62-e348a58586df-r2.ufospace.cloud"
BASE_URL = f"https://{BASE_HOST}"

def upload_file(remote_filename, content_str):
    cj = http.cookiejar.CookieJar()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cj),
        urllib.request.HTTPSHandler(context=ctx)
    )

    # 1. GET /lab to get session & _xsrf cookies
    req1 = urllib.request.Request(
        f"{BASE_URL}/lab?token={TOKEN}",
        headers={"User-Agent": "Mozilla/5.0"}
    )
    opener.open(req1, timeout=10)
    
    cookies = {c.name: c.value for c in cj}
    xsrf = cookies.get("_xsrf", "")
    
    # 2. PUT to /api/contents/{remote_filename}
    headers = {
        "Authorization": f"token {TOKEN}",
        "X-XSRFToken": xsrf,
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0"
    }
    payload = {
        "type": "file",
        "format": "text",
        "content": content_str
    }
    
    req2 = urllib.request.Request(
        f"{BASE_URL}/api/contents/{remote_filename}?token={TOKEN}",
        headers=headers,
        data=json.dumps(payload).encode("utf-8"),
        method="PUT"
    )
    with opener.open(req2, timeout=15) as res:
        return res.getcode(), res.read().decode("utf-8")

if __name__ == "__main__":
    local_file = sys.argv[1] if len(sys.argv) > 1 else "scripts/full_pipeline.py"
    remote_name = sys.argv[2] if len(sys.argv) > 2 else "full_pipeline.py"
    
    with open(local_file, "r", encoding="utf-8") as f:
        content = f.read()
        
    print(f"Uploading {local_file} -> {remote_name}...")
    code, resp = upload_file(remote_name, content)
    print(f"Result code: {code}")
    print("Success! File uploaded directly via JupyterLab API.")
