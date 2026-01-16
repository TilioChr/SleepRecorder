import json
import os
import re
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

from .audio_analysis import analyze_wav_file
from .settings import (
    AUDIO_TCP_HOST,
    AUDIO_TCP_PORT,
    API_HOST,
    API_PORT,
    RECORDINGS_DIR,
    SAMPLE_RATE,
    CHANNELS,
    SAMPLE_WIDTH_BYTES,
    CHUNK_SECONDS,
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
    """Expected filename prefix: YYYYMMDD-HHMMSS..."""
    m = re.match(r"^(\d{8})-(\d{6})", name)
    if not m:
        return None, None
    ymd, hms = m.group(1), m.group(2)
    date_str = f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"
    time_str = f"{hms[0:2]}:{hms[2:4]}:{hms[4:6]}"
    return date_str, time_str


def _ts_from_date_time(date_str: str, time_str: str) -> int | None:
    try:
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        return int(dt.timestamp())
    except Exception:
        return None


def _date_time_from_ts(ts: int):
    dt = datetime.fromtimestamp(int(ts))
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")


def _ensure_defaults(meta: dict, filename: str, file_mtime: int | None = None) -> dict:
    """Garantit que les champs existent, sans écraser l'existant."""
    meta.setdefault(filename, {})
    meta[filename].setdefault("tag", "Non tagué")
    if "recorded_at" not in meta[filename]:
        # on essaye d'inférer recorded_at depuis le nom, sinon fallback sur mtime
        d, t = _parse_dt_from_name(filename)
        ts = _ts_from_date_time(d, t) if (d and t) else None
        if ts is None:
            ts = int(file_mtime if file_mtime is not None else time.time())
        meta[filename]["recorded_at"] = int(ts)
    # état du traitement VAD/classif
    meta[filename].setdefault("analysis_status", "pending")  # pending | processing | done
    meta[filename].setdefault("analysis_kind", None)  # silence | voice | snore | noise
    meta[filename].setdefault("analysis_confidence", None)
    meta[filename].setdefault("analysis_has_activity", None)
    return meta[filename]


# -------------------------
# FastAPI
# -------------------------

app = FastAPI(title="Sleep Recorder API")


def now_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def write_wav(path: Path, pcm_bytes: bytes, recorded_at: int | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH_BYTES)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)

    # initialiser meta + marquer comme à traiter
    meta = _load_meta()
    _ensure_defaults(meta, path.name, file_mtime=int(path.stat().st_mtime))
    if recorded_at is not None:
        meta[path.name]["recorded_at"] = int(recorded_at)
    meta[path.name]["analysis_status"] = "pending"
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
        if not p.is_file() or p.suffix.lower() != ".wav":
            continue

        try:
            st = p.stat()
        except FileNotFoundError:
            continue

        m = meta.get(p.name)
        if m is None:
            _ensure_defaults(meta, p.name, file_mtime=int(st.st_mtime))
            meta_dirty = True
            m = meta[p.name]
        else:
            before = dict(m)
            _ensure_defaults(meta, p.name, file_mtime=int(st.st_mtime))
            if meta[p.name] != before:
                meta_dirty = True
            m = meta[p.name]

        recorded_at = int(m.get("recorded_at") or st.st_mtime)
        name_date, name_time = _parse_dt_from_name(p.name)
        if name_date and name_time:
            date_str, time_str = name_date, name_time
        else:
            date_str, time_str = _date_time_from_ts(recorded_at)

        out.append(
            {
                "name": p.name,
                "size": st.st_size,
                "mtime": int(st.st_mtime),
                "date": date_str,
                "time": time_str,
                "tag": m.get("tag", "Non tagué"),
                "url": f"/recordings/{p.name}",
                "analysis_status": m.get("analysis_status", "pending"),
                "analysis_kind": m.get("analysis_kind"),
                "analysis_confidence": m.get("analysis_confidence"),
                "analysis_has_activity": m.get("analysis_has_activity"),
            }
        )

    if meta_dirty:
        _save_meta(meta)

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

    if name in meta:
        meta[body.new_name] = meta.pop(name)

    # garantir la présence des champs pour le nouveau nom
    _ensure_defaults(meta, body.new_name, file_mtime=int(dst.stat().st_mtime))
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
    _ensure_defaults(meta, name, file_mtime=int(p.stat().st_mtime))
    meta[name]["tag"] = body.tag
    _save_meta(meta)

    return {"ok": True, "tag": body.tag}


# Backend also serves /recordings (useful in local). nginx can serve it in prod.
app.mount("/recordings", StaticFiles(directory=str(REC_DIR)), name="recordings")


# -------------------------
# Background analysis worker
# -------------------------

_analysis_lock = threading.Lock()


def _auto_tag_from_kind(kind: str | None) -> str | None:
    if kind == "voice":
        return "Parole"
    if kind == "snore":
        return "Ronflement"
    if kind == "noise":
        return "Bruit"
    return None


def analysis_worker():
    """Traite chaque .wav une seule fois (analysis_status: pending->processing->done)."""
    print("[ANALYSIS] worker started")
    while True:
        try:
            to_process = None

            with _analysis_lock:
                meta = _load_meta()

                try:
                    wavs = [p for p in REC_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".wav"]
                except FileNotFoundError:
                    wavs = []

                for p in sorted(wavs, key=lambda x: x.stat().st_mtime):
                    st = p.stat()
                    _ensure_defaults(meta, p.name, file_mtime=int(st.st_mtime))
                    if meta[p.name].get("analysis_status") == "pending":
                        meta[p.name]["analysis_status"] = "processing"
                        _save_meta(meta)
                        to_process = p
                        break

            if to_process is None:
                time.sleep(1.0)
                continue

            try:
                res = analyze_wav_file(str(to_process))
            except Exception as e:
                with _analysis_lock:
                    meta = _load_meta()
                    if to_process.name in meta:
                        meta[to_process.name]["analysis_status"] = "pending"
                        meta[to_process.name]["analysis_kind"] = None
                        meta[to_process.name]["analysis_confidence"] = None
                        meta[to_process.name]["analysis_has_activity"] = None
                        _save_meta(meta)
                print(f"[ANALYSIS] error on {to_process.name}: {e}")
                time.sleep(0.5)
                continue

            with _analysis_lock:
                meta = _load_meta()
                if to_process.name not in meta:
                    continue

                meta[to_process.name]["analysis_status"] = "done"
                meta[to_process.name]["analysis_kind"] = res.kind
                meta[to_process.name]["analysis_confidence"] = float(res.confidence)
                meta[to_process.name]["analysis_has_activity"] = bool(res.has_activity)

                current_tag = meta[to_process.name].get("tag") or "Non tagué"
                auto = _auto_tag_from_kind(res.kind)
                if auto and current_tag == "Non tagué":
                    meta[to_process.name]["tag"] = auto

                _save_meta(meta)

        except Exception as e:
            print(f"[ANALYSIS] worker loop error: {e}")
            time.sleep(1.0)


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
    t_audio = threading.Thread(target=audio_tcp_server, daemon=True)
    t_audio.start()

    t_analysis = threading.Thread(target=analysis_worker, daemon=True)
    t_analysis.start()

    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    main()
