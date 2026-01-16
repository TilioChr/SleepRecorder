import os
import re
import json
import socket
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from .settings import (
    AUDIO_TCP_HOST, AUDIO_TCP_PORT,
    API_HOST, API_PORT,
    RECORDINGS_DIR,
    SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH_BYTES,
    CHUNK_SECONDS
)

# -------------------------
# Paths / persistence
# -------------------------

REC_DIR = Path(RECORDINGS_DIR)
REC_DIR.mkdir(parents=True, exist_ok=True)

META_PATH = REC_DIR / "meta.json"
NAME_SAFE = re.compile(r"^[a-zA-Z0-9._-]+$")  # avoid path traversal + weird chars


def _load_meta() -> dict:
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        # if corrupted, don't crash the whole app
        return {}


def _save_meta(meta: dict) -> None:
    tmp = META_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(META_PATH)


def _check_name(name: str) -> None:
    # disallow any path separators and enforce a strict charset
    if "/" in name or "\\" in name or not NAME_SAFE.match(name):
        raise HTTPException(status_code=400, detail="Nom de fichier invalide")


def _parse_dt_from_name(name: str):
    """
    Expected filename prefix: YYYYMMDD-HHMMSS...
    Returns: ("YYYY-MM-DD", "HH:MM:SS") or (None, None)
    """
    m = re.match(r"^(\d{8})-(\d{6})", name)
    if not m:
        return None, None
    ymd, hms = m.group(1), m.group(2)
    date_str = f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"
    time_str = f"{hms[0:2]}:{hms[2:4]}:{hms[4:6]}"
    return date_str, time_str


def _ts_from_date_time(date_str: str, time_str: str) -> int | None:
    """
    Convertit ("YYYY-MM-DD", "HH:MM:SS") en timestamp Unix (secondes).
    Interprété en heure locale (naïf), cohérent avec datetime.now() utilisé pour nommer.
    """
    try:
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        return int(dt.timestamp())
    except Exception:
        return None


def _date_time_from_ts(ts: int):
    dt = datetime.fromtimestamp(int(ts))
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")


# -------------------------
# FastAPI
# -------------------------

app = FastAPI(title="Sleep Recorder API")


def now_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def write_wav(path: Path, pcm_bytes: bytes, recorded_at: int | None = None):
    # ensure parent exists
    path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH_BYTES)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)

    # Persist une date/heure stable dans meta.json, indépendante du nom de fichier
    meta = _load_meta()
    meta.setdefault(path.name, {})
    meta[path.name].setdefault("tag", "Non tagué")
    meta[path.name].setdefault("recorded_at", int(recorded_at if recorded_at is not None else time.time()))
    _save_meta(meta)


def list_wavs():
    meta = _load_meta()
    out = []
    meta_dirty = False

    try:
        entries = list(REC_DIR.iterdir())
    except FileNotFoundError:
        entries = []

    for p in entries:
        if not p.is_file():
            continue
        if p.suffix.lower() != ".wav":
            continue

        try:
            st = p.stat()
        except FileNotFoundError:
            continue

        entry_meta = meta.get(p.name) or {}
        tag = entry_meta.get("tag", "Non tagué")
        recorded_at = entry_meta.get("recorded_at")

        # 1) Si on a un nom avec date/heure et pas recorded_at, on migre une fois
        name_date, name_time = _parse_dt_from_name(p.name)
        if recorded_at is None and name_date and name_time:
            ts = _ts_from_date_time(name_date, name_time)
            if ts is not None:
                meta.setdefault(p.name, {})
                meta[p.name]["recorded_at"] = ts
                recorded_at = ts
                meta_dirty = True

        # 2) Si toujours pas de recorded_at, fallback sur mtime (meilleur possible)
        if recorded_at is None:
            recorded_at = int(st.st_mtime)
            meta.setdefault(p.name, {})
            meta[p.name].setdefault("recorded_at", recorded_at)
            meta_dirty = True

        # 3) date/time affichés:
        #    - si le nom est parseable, on garde l'affichage depuis le nom
        #    - sinon, on affiche recorded_at (stable)
        if name_date and name_time:
            date_str, time_str = name_date, name_time
        else:
            date_str, time_str = _date_time_from_ts(recorded_at)

        out.append({
            "name": p.name,
            "size": st.st_size,
            "mtime": int(st.st_mtime),
            "date": date_str,
            "time": time_str,
            "tag": tag,
            "url": f"/recordings/{p.name}",
        })

    if meta_dirty:
        _save_meta(meta)

    # most recent first
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out


@app.get("/api/recordings")
def api_recordings():
    return JSONResponse(list_wavs())


class RenameBody(BaseModel):
    new_name: str


class TagBody(BaseModel):
    tag: str


@app.post("/api/recordings/{name}/rename")
def api_rename(name: str, body: RenameBody):
    _check_name(name)
    _check_name(body.new_name)

    src = REC_DIR / name
    dst = REC_DIR / body.new_name

    if not src.exists():
        raise HTTPException(status_code=404, detail="Fichier introuvable")
    if dst.exists():
        raise HTTPException(status_code=409, detail="Le nouveau nom existe déjà")

    src.rename(dst)

    meta = _load_meta()

    # Déplacer la meta si elle existe
    if name in meta:
        meta[body.new_name] = meta.pop(name)

    # Assurer recorded_at même pour d'anciens fichiers (ou si pas de meta)
    meta.setdefault(body.new_name, {})
    meta[body.new_name].setdefault("tag", "Non tagué")
    meta[body.new_name].setdefault("recorded_at", int(dst.stat().st_mtime))

    _save_meta(meta)

    return {"ok": True, "name": body.new_name}


@app.delete("/api/recordings/{name}")
def api_delete(name: str):
    _check_name(name)

    p = REC_DIR / name
    if not p.exists():
        raise HTTPException(status_code=404, detail="Fichier introuvable")

    p.unlink()

    meta = _load_meta()
    if name in meta:
        meta.pop(name, None)
        _save_meta(meta)

    return {"ok": True}


@app.post("/api/recordings/{name}/tag")
def api_tag(name: str, body: TagBody):
    _check_name(name)

    p = REC_DIR / name
    if not p.exists():
        raise HTTPException(status_code=404, detail="Fichier introuvable")

    meta = _load_meta()
    meta.setdefault(name, {})
    meta[name].setdefault("recorded_at", int(p.stat().st_mtime))
    meta[name]["tag"] = body.tag
    _save_meta(meta)

    return {"ok": True, "tag": body.tag}


# Backend also serves /recordings (useful in local). nginx can serve it in prod.
app.mount("/recordings", StaticFiles(directory=str(REC_DIR)), name="recordings")


# -------------------------
# Audio TCP server
# -------------------------

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

                # keep PCM aligned on 2 bytes (s16le)
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
                    write_wav(REC_DIR / fname, chunk, recorded_at=int(time.time()))
                    print(f"[AUDIO] wrote {fname}")

        except socket.timeout:
            print("[AUDIO] timeout")
        finally:
            conn.close()

            if buf:
                fname = f"{now_id()}_tail.wav"
                write_wav(REC_DIR / fname, buf, recorded_at=int(time.time()))
                print(f"[AUDIO] wrote {fname}")

            print("[AUDIO] disconnected")


def main():
    t = threading.Thread(target=audio_tcp_server, daemon=True)
    t.start()
    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    main()
