import asyncio
import base64
import json
import ssl
import sys
import websockets

TOKEN = "2eb9977615ff4fa0"
BASE_HOST = "ufo-43751a60-b805-4610-8c62-e348a58586df-r2.ufospace.cloud"
TERM_ID = "1"

def get_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

async def run_in_terminal(cmd_str, timeout=30):
    uri = f"wss://{BASE_HOST}/terminals/websocket/{TERM_ID}?token={TOKEN}"
    headers = {"Origin": f"https://{BASE_HOST}"}
    ssl_ctx = get_ssl_context()
    
    async with websockets.connect(uri, ssl=ssl_ctx, additional_headers=headers) as ws:
        await ws.send(json.dumps(["set_size", 40, 120]))
        # Drain buffer
        while True:
            try:
                await asyncio.wait_for(ws.recv(), timeout=0.2)
            except asyncio.TimeoutError:
                break
                
        # Send Ctrl+C first in case there is a dangling prompt
        await ws.send(json.dumps(["stdin", "\x03\n"]))
        await asyncio.sleep(0.3)
        while True:
            try:
                await asyncio.wait_for(ws.recv(), timeout=0.2)
            except asyncio.TimeoutError:
                break

        tag = "===TAG_DONE_999==="
        clean_cmd = cmd_str.strip().replace("\r\n", "\n")
        sent_cmd = f"{clean_cmd}\necho '{tag}'\n"
        await ws.send(json.dumps(["stdin", sent_cmd]))
        
        output = ""
        loop = asyncio.get_event_loop()
        start = loop.time()
        
        while loop.time() - start < timeout:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                data = json.loads(msg)
                if data[0] == "stdout":
                    output += data[1]
                    if f"\n{tag}" in output or f"\r\n{tag}" in output:
                        break
            except asyncio.TimeoutError:
                continue
                
        return output

async def create_remote_file(file_path, content):
    b64_str = base64.b64encode(content.encode("utf-8")).decode("ascii")
    chunk_size = 500
    chunks = [b64_str[i:i+chunk_size] for i in range(0, len(b64_str), chunk_size)]
    
    # Write to a temp .b64 file in chunks
    b64_temp = f"/tmp/{abs(hash(file_path))}.b64"
    cmds = [f"rm -f {b64_temp}"]
    for chunk in chunks:
        cmds.append(f"printf '%s' '{chunk}' >> {b64_temp}")
    cmds.append(f"base64 -d {b64_temp} > {file_path}")
    cmds.append(f"rm -f {b64_temp}")
    
    full_cmd = " && ".join(cmds)
    return await run_in_terminal(full_cmd, timeout=25)

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    with open("scripts/full_pipeline.py", "r", encoding="utf-8") as f:
        code = f.read()
    print("Uploading full_pipeline.py chunked...", flush=True)
    asyncio.run(create_remote_file("full_pipeline.py", code))
    print("Upload completed! Checking remote file...", flush=True)
    res = asyncio.run(run_in_terminal("ls -l full_pipeline.py; head -n 5 full_pipeline.py", timeout=10))
    print(res, flush=True)
