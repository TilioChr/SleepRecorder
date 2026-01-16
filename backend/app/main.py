import os
import re
import json
import socket
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import webrtcvad

import torch

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

# --- important: where panns downloads its files ---
os.environ.setdefault("PANNS_DATA_DIR", "/app/panns_data")

# -------------------------
# Paths / persistence
# -------------------------

REC_DIR = Path(RECORDINGS_DIR)
REC_DIR.mkdir(parents=True, exist_ok=True)

META_PATH = REC_DIR / "meta.json"
NAME_SAFE = re.compile(r"^[a-zA-Z0-9._-]+$")
META_LOCK = threading.Lock()

STATUS_UNPROCESSED = "unprocessed"
STATUS_PROCESSING = "processing"
STATUS_DONE = "done"
STATUS_ERROR = "error"


def _load_meta() -> dict:
    with META_LOCK:
        try:
            return json.loads(META_PATH.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except Exception:
            return {}


def _save_meta(meta: dict) -> None:
    with META_LOCK:
        tmp = META_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(META_PATH)


def _check_name(name: str) -> None:
    if "/" in name or "\\" in name or not NAME_SAFE.match(name):
        raise HTTPException(status_code=400, detail="Nom de fichier invalide")


def _parse_dt_from_name(name: str):
    m = re.match(r"^(\d{8})-(\d{6})", name)
    if not m:
        return None, None
    ymd, hms = m.group(1), m.group(2)
    date_str = f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"
    time_str = f"{hms[0:2]}:{hms[2:4]}:{hms[4:6]}"
    return date_str, time_str


def _ts_from_date_time(date_str: str, time_str: str) -> Optional[int]:
    try:
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        return int(dt.timestamp())
    except Exception:
        return None


def _date_time_from_ts(ts: int):
    dt = datetime.fromtimestamp(int(ts))
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")


def _ensure_meta_entry(meta: dict, filename: str, recorded_at: Optional[int] = None) -> dict:
    entry = meta.get(filename)
    if not isinstance(entry, dict):
        entry = {}
        meta[filename] = entry

    entry.setdefault("tag", "Non tagué")
    entry.setdefault("tag_source", "auto")
    entry.setdefault("processing_status", STATUS_UNPROCESSED)
    entry.setdefault("activity", None)
    entry.setdefault("sound_type", None)
    entry.setdefault("processed_at", None)
    entry.setdefault("error", None)
    entry.setdefault("scores", {})

    if entry.get("recorded_at") is None:
        if recorded_at is not None:
            entry["recorded_at"] = int(recorded_at)
        else:
            d, t = _parse_dt_from_name(filename)
            if d and t:
                ts = _ts_from_date_time(d, t)
                entry["recorded_at"] = int(ts if ts is not None else time.time())
            else:
                entry["recorded_at"] = int(time.time())

    return entry


# -------------------------
# Audio helpers
# -------------------------

def _read_wav_pcm16(path: Path) -> Tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)

    if sw != 2:
        raise ValueError(f"Unsupported sample width: {sw} bytes (expected 2)")

    x = np.frombuffer(raw, dtype=np.int16)
    if ch == 2:
        x = x.reshape(-1, 2).mean(axis=1).astype(np.int16)
    elif ch != 1:
        raise ValueError(f"Unsupported channels: {ch} (expected 1)")

    return sr, x


def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    if len(x) == 0:
        return x
    duration = len(x) / float(sr_in)
    new_len = int(duration * sr_out)
    if new_len <= 0:
        return np.zeros((0,), dtype=x.dtype)

    xf = x.astype(np.float32)
    xp = np.linspace(0.0, 1.0, num=len(xf), endpoint=False)
    xq = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
    y = np.interp(xq, xp, xf)
    return y.astype(np.int16)


# -------------------------
# VAD gate
# -------------------------

def vad_activity(sr: int, x_i16: np.ndarray) -> float:
    if len(x_i16) < sr // 10:
        return 0.0

    if sr != 16000:
        x_i16 = _resample_linear(x_i16, sr, 16000)
        sr = 16000

    frame_ms = 30
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(x_i16) // frame_len
    if n_frames <= 0:
        return 0.0

    vad = webrtcvad.Vad(2)
    x_frames = x_i16[: n_frames * frame_len].reshape(n_frames, frame_len)

    voiced = 0
    for i in range(n_frames):
        if vad.is_speech(x_frames[i].tobytes(), sr):
            voiced += 1

    return voiced / float(n_frames)


# -------------------------
# PANNs pretrained (lazy import)
# -------------------------

_AT = None
_PANNS_LABELS = None

def _ensure_panns_loaded():
    global _AT, _PANNS_LABELS
    if _AT is not None and _PANNS_LABELS is not None:
        return

    # import ici pour éviter crash au démarrage si le CSV n'est pas encore là
    from panns_inference import AudioTagging, labels as PANNS_LABELS

    _PANNS_LABELS = PANNS_LABELS
    _AT = AudioTagging(checkpoint_path=None, device="cpu")


def _max_prob_for_keywords(probs: np.ndarray, keywords: list[str]) -> float:
    best = 0.0
    for i, lbl in enumerate(_PANNS_LABELS):
        ll = lbl.lower()
        if any(k in ll for k in keywords):
            p = float(probs[i])
            if p > best:
                best = p
    return best


def classify_with_panns(sr: int, x_i16: np.ndarray) -> dict:
    _ensure_panns_loaded()

    if sr != 32000:
        x_i16 = _resample_linear(x_i16, sr, 32000)
        sr = 32000

    audio = (x_i16.astype(np.float32) / 32768.0)[None, :]  # (1, N)

    with torch.inference_mode():
        clipwise_output, _ = _AT.inference(audio)

    probs = clipwise_output[0]

    p_voice = _max_prob_for_keywords(probs, ["speech", "conversation", "narration", "talk", "voice"])
    p_snore = _max_prob_for_keywords(probs, ["snor"])

    # seuils à ajuster ensuite
    if p_snore >= 0.25 and p_snore >= p_voice:
        return {"sound_type": "snore", "scores": {"voice": p_voice, "snore": p_snore}}
    if p_voice >= 0.30 and p_voice >= p_snore:
        return {"sound_type": "voice", "scores": {"voice": p_voice, "snore": p_snore}}
    return {"sound_type": "noise", "scores": {"voice": p_voice, "snore": p_snore}}


def _auto_tag_from_sound(sound_type: str) -> str:
    if sound_type == "snore":
        return "Ronflement"
    if sound_type == "voice":
        return "Parole"
    if sound_type == "noise":
        return "Bruit"
    return "Non tagué"


def analyze_audio(path: Path) -> dict:
    sr, x = _read_wav_pcm16(path)
    voiced_ratio = vad_activity(sr, x)
    activity = voiced_ratio >= 0.08

    if not activity:
        return {"activity": False, "sound_type": "silence", "voiced_ratio": float(voiced_ratio), "scores": {}}

    cls = classify_with_panns(sr, x)
    return {
        "activity": True,
        "sound_type": cls["sound_type"],
        "voiced_ratio": float(voiced_ratio),
        "scores": cls.get("scores", {}),
    }


# -------------------------
# Worker thread
# -------------------------

def processing_worker():
    print("[VAD] worker started")
    while True:
        try:
            wavs = [p for p in REC_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".wav"]
        except FileNotFoundError:
            wavs = []

        meta = _load_meta()
        dirty = False

        for p in wavs:
            if p.name not in meta:
                st = p.stat()
                _ensure_meta_entry(meta, p.name, recorded_at=int(st.st_mtime))
                dirty = True
        if dirty:
            _save_meta(meta)

        job = None
        for p in wavs:
            stt = (meta.get(p.name) or {}).get("processing_status", STATUS_UNPROCESSED)
            if stt == STATUS_UNPROCESSED:
                job = p
                break

        if job is None:
            time.sleep(1.5)
            continue

        meta = _load_meta()
        entry = _ensure_meta_entry(meta, job.name, recorded_at=int(job.stat().st_mtime))
        if entry.get("processing_status") != STATUS_UNPROCESSED:
            time.sleep(0.2)
            continue

        entry["processing_status"] = STATUS_PROCESSING
        entry["error"] = None
        _save_meta(meta)

        try:
            res = analyze_audio(job)

            meta = _load_meta()
            entry = _ensure_meta_entry(meta, job.name, recorded_at=entry.get("recorded_at"))

            entry["activity"] = bool(res.get("activity"))
            entry["sound_type"] = res.get("sound_type")
            entry["scores"] = res.get("scores", {})
            entry["processed_at"] = int(time.time())
            entry["processing_status"] = STATUS_DONE
            entry["error"] = None

            if entry.get("tag_source", "auto") != "manual":
                entry["tag"] = _auto_tag_from_sound(entry.get("sound_type") or "")
                entry["tag_source"] = "auto"

            _save_meta(meta)
            print(f"[VAD] done {job.name}: {entry.get('sound_type')} tag={entry.get('tag')} scores={entry.get('scores')}")

        except Exception as e:
            meta = _load_meta()
            entry = _ensure_meta_entry(meta, job.name, recorded_at=int(job.stat().st_mtime))
            entry["processing_status"] = STATUS_ERROR
            entry["error"] = str(e)
            entry["processed_at"] = int(time.time())
            _save_meta(meta)
            print(f"[VAD] error {job.name}: {e}")

        time.sleep(0.1)


# -------------------------
# FastAPI
# -------------------------

app = FastAPI(title="Sleep Recorder API")


def now_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def write_wav(path: Path, pcm_bytes: bytes, recorded_at: Optional[int] = None):
    path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH_BYTES)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)

    meta = _load_meta()
    entry = _ensure_meta_entry(meta, path.name, recorded_at=int(recorded_at if recorded_at is not None else time.time()))
    entry["processing_status"] = STATUS_UNPROCESSED
    entry["error"] = None
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

        entry = _ensure_meta_entry(meta, p.name, recorded_at=int(st.st_mtime))

        name_date, name_time = _parse_dt_from_name(p.name)
        recorded_at = entry.get("recorded_at")
        if name_date and name_time:
            date_str, time_str = name_date, name_time
            if recorded_at is None:
                ts = _ts_from_date_time(name_date, name_time)
                if ts is not None:
                    entry["recorded_at"] = ts
                    meta_dirty = True
        else:
            if recorded_at is None:
                entry["recorded_at"] = int(st.st_mtime)
                meta_dirty = True
            date_str, time_str = _date_time_from_ts(int(entry["recorded_at"]))

        out.append({
            "name": p.name,
            "size": st.st_size,
            "mtime": int(st.st_mtime),
            "date": date_str,
            "time": time_str,
            "tag": entry.get("tag", "Non tagué"),
            "url": f"/recordings/{p.name}",
            "processing_status": entry.get("processing_status", STATUS_UNPROCESSED),
        })

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

    st = dst.stat()
    entry = _ensure_meta_entry(meta, body.new_name, recorded_at=int(st.st_mtime))
    meta[body.new_name] = entry
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
    entry = _ensure_meta_entry(meta, name, recorded_at=int(p.stat().st_mtime))
    entry["tag"] = body.tag
    entry["tag_source"] = "manual"
    _save_meta(meta)

    return {"ok": True, "tag": body.tag}


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

    t_worker = threading.Thread(target=processing_worker, daemon=True)
    t_worker.start()

    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    main()
