import os
import time
import socket
import threading
import wave
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from .settings import (
    AUDIO_TCP_HOST, AUDIO_TCP_PORT,
    API_HOST, API_PORT,
    RECORDINGS_DIR,
    SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH_BYTES,
    CHUNK_SECONDS
)

os.makedirs(RECORDINGS_DIR, exist_ok=True)

app = FastAPI(title="Sleep Recorder API")

def now_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def write_wav(path: str, pcm_bytes: bytes):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH_BYTES)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)

def list_wavs():
    out = []
    for name in os.listdir(RECORDINGS_DIR):
        if not name.lower().endswith(".wav"):
            continue
        p = os.path.join(RECORDINGS_DIR, name)
        try:
            st = os.stat(p)
            out.append({
                "name": name,
                "size": st.st_size,
                "mtime": int(st.st_mtime),
                "url": f"/recordings/{name}",
            })
        except FileNotFoundError:
            pass
    out.sort(key=lambda x: x["name"], reverse=True)
    return out

@app.get("/api/recordings")
def api_recordings():
    return JSONResponse(list_wavs())

# backend sert aussi /recordings (utile pour debug); nginx le servira en prod
app.mount("/recordings", StaticFiles(directory=RECORDINGS_DIR), name="recordings")

def audio_tcp_server():
    print(f"[AUDIO] listening on {AUDIO_TCP_HOST}:{AUDIO_TCP_PORT}")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((AUDIO_TCP_HOST, AUDIO_TCP_PORT))
    srv.listen(5)

    bytes_per_second = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH_BYTES
    target_chunk_bytes = CHUNK_SECONDS * bytes_per_second

    while True:
        conn, addr = srv.accept()
        print(f"[AUDIO] client {addr} connected")
        conn.settimeout(5.0)

        buf = b""
        carry = b""
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                data = carry + data
                if len(data) % 2 == 1:
                    carry = data[-1:]
                    data = data[:-1]
                else:
                    carry = b""

                buf += data
                while len(buf) >= target_chunk_bytes:
                    chunk = buf[:target_chunk_bytes]
                    buf = buf[target_chunk_bytes:]
                    fname = f"{now_id()}_{CHUNK_SECONDS}s.wav"
                    write_wav(os.path.join(RECORDINGS_DIR, fname), chunk)
                    print(f"[AUDIO] wrote {fname}")
        except socket.timeout:
            print("[AUDIO] timeout")
        finally:
            conn.close()
            if buf:
                fname = f"{now_id()}_tail.wav"
                write_wav(os.path.join(RECORDINGS_DIR, fname), buf)
                print(f"[AUDIO] wrote {fname}")
            print("[AUDIO] disconnected")

def main():
    t = threading.Thread(target=audio_tcp_server, daemon=True)
    t.start()
    uvicorn.run(app, host=API_HOST, port=API_PORT)

if __name__ == "__main__":
    main()
