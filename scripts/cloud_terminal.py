import asyncio
import json
import ssl
import sys
import websockets

TOKEN = "2eb9977615ff4fa0"
BASE_HOST = "ufo-43751a60-b805-4610-8c62-e348a58586df-r2.ufospace.cloud"
TERM_ID = "1"

async def exec_command(cmd, timeout=15):
    uri = f"wss://{BASE_HOST}/terminals/websocket/{TERM_ID}?token={TOKEN}"
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    
    headers = {
        "Origin": f"https://{BASE_HOST}"
    }
    
    async with websockets.connect(uri, ssl=ssl_ctx, additional_headers=headers) as ws:
        await ws.send(json.dumps(["set_size", 40, 120]))
        # Drain existing buffer
        while True:
            try:
                await asyncio.wait_for(ws.recv(), timeout=0.3)
            except asyncio.TimeoutError:
                break
                
        sent_cmd = f"{cmd.strip()}; echo '<<<DONE>>>'\n"
        await ws.send(json.dumps(["stdin", sent_cmd]))
        
        output = ""
        loop = asyncio.get_event_loop()
        start_time = loop.time()
        
        while loop.time() - start_time < timeout:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                data = json.loads(msg)
                if data[0] == "stdout":
                    output += data[1]
                    # Check if '<<<DONE>>>' appears as output (after newline)
                    if "\n<<<DONE>>>" in output or "\r\n<<<DONE>>>" in output:
                        break
            except asyncio.TimeoutError:
                continue
                
        return output

if __name__ == "__main__":
    test_cmd = sys.argv[1] if len(sys.argv) > 1 else "pwd"
    res = asyncio.run(exec_command(test_cmd, timeout=20))
    print("OUTPUT:\n" + res)
