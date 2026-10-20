import asyncio
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

async def run_clean_cmd(cmd_str, timeout=10):
    uri = f"wss://{BASE_HOST}/terminals/websocket/{TERM_ID}?token={TOKEN}"
    headers = {"Origin": f"https://{BASE_HOST}"}
    ssl_ctx = get_ssl_context()
    
    async with websockets.connect(uri, ssl=ssl_ctx, additional_headers=headers) as ws:
        await ws.send(json.dumps(["set_size", 40, 120]))
        # Clear buffer
        while True:
            try:
                await asyncio.wait_for(ws.recv(), timeout=0.2)
            except asyncio.TimeoutError:
                break
                
        tag = "===TAG_STATUS_OK==="
        sent = f"{cmd_str}\necho '{tag}'\n"
        await ws.send(json.dumps(["stdin", sent]))
        
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
                
        # Split off before tag
        if tag in output:
            output = output.split(tag)[0]
        return output

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ps aux | grep -E 'python|yolo'; echo '---LOG---'; tail -n 25 train_pipeline.log 2>&1"
    res = asyncio.run(run_clean_cmd(cmd, timeout=12))
    print(res)
