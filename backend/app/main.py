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
META_LOCK = threading.Lock()

# processing_status values
STATUS_UNPROCESSED = "unprocessed"   # rouge
STATUS_PROCESSING = "processing"     # orange
STATUS_DONE = "done"                 # bleu
STATUS_ERROR = "error"               # rouge (fallback)


def _load_meta() -> dict:
    with META_LOCK:
        try:
            return json.loads(META_PATH.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except Exception:
            # if corrupted, don't crash the whole app
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
    """
    Garantit que meta[filename] existe avec les champs minimums.
    recorded_at est la date stable de "création"/enregistrement (pas dépendante du rename).
    """
    entry = meta.get(filename)
    if not isinstance(entry, dict):
        entry = {}
        meta[filename] = entry

    entry.setdefault("tag", "Non tagué")
    entry.setdefault("tag_source", "auto")  # auto par défaut (manual si user change)
    entry.setdefault("processing_status", STATUS_UNPROCESSED)
    entry.setdefault("activity", None)
    entry.setdefault("sound_type", None)
    entry.setdefault("processed_at", None)
    entry.setdefault("error", None)

    if entry.get("recorded_at") is None:
        if recorded_at is not None:
            entry["recorded_at"] = int(recorded_at)
        else:
            # tentative migration depuis le nom
            d, t = _parse_dt_from_name(filename)
            if d and t:
                ts = _ts_from_date_time(d, t)
                if ts is not None:
                    entry["recorded_at"] = ts
                else:
                    entry["recorded_at"] = int(time.time())
            else:
                entry["recorded_at"] = int(time.time())

    return entry


# -------------------------
# VAD + classification (simple heuristics)
# -------------------------

def _read_wav_pcm16(path: Path) -> Tuple[int, np.ndarray]:
    """
    Retourne (sample_rate, samples int16 mono).
    Hypothèse projet: wav mono, 16k, s16le. On protège quand même.
    """
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)

    if sw != 2:
        raise ValueError(f"Unsupported sample width: {sw} bytes (expected 2)")

    samples = np.frombuffer(raw, dtype=np.int16)
    if ch == 2:
        samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)
    elif ch != 1:
        raise ValueError(f"Unsupported channels: {ch} (expected 1)")

    if sr != 16000:
        # resample simple (linéaire) pour éviter dépendances lourdes
        x = samples.astype(np.float32)
        duration = len(x) / float(sr)
        new_len = int(duration * 16000)
        if new_len <= 0:
            return 16000, np.zeros((0,), dtype=np.int16)
        xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
        xq = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        y = np.interp(xq, xp, x).astype(np.int16)
        return 16000, y

    return sr, samples


def _frame_bytes(samples_i16: np.ndarray, frame_size: int) -> bytes:
    # frame_size in samples
    return samples_i16[:frame_size].tobytes()


def analyze_audio(path: Path) -> dict:
    """
    1) VAD (webrtcvad) -> activity + voiced_ratio
    2) Si activity: heuristiques -> snore / voice / noise
    """
    sr, x = _read_wav_pcm16(path)
    if len(x) < sr // 10:  # <100ms
        return {"activity": False, "sound_type": "silence", "voiced_ratio": 0.0}

    vad = webrtcvad.Vad(2)  # 0..3 (2 = bon compromis)
    frame_ms = 30
    frame_len = int(sr * frame_ms / 1000)  # 480 samples @ 16k
    if frame_len <= 0:
        return {"activity": False, "sound_type": "silence", "voiced_ratio": 0.0}

    # tronquer en frames complètes
    n_frames = len(x) // frame_len
    if n_frames == 0:
        return {"activity": False, "sound_type": "silence", "voiced_ratio": 0.0}

    x_frames = x[: n_frames * frame_len].reshape(n_frames, frame_len)

    voiced = 0
    voiced_idx = []
    for i in range(n_frames):
        fb = x_frames[i].tobytes()
        is_speech = vad.is_speech(fb, sr)
        if is_speech:
            voiced += 1
            voiced_idx.append(i)

    voiced_ratio = voiced / float(n_frames)
    activity = voiced_ratio >= 0.08  # seuil "il se passe quelque chose"
    if not activity:
        return {"activity": False, "sound_type": "silence", "voiced_ratio": voiced_ratio}

    # Features sur les frames "voiced" (ou fallback sur tout)
    use = x_frames[voiced_idx] if voiced_idx else x_frames
    y = use.flatten().astype(np.float32)

    # RMS
    rms = float(np.sqrt(np.mean(y * y)) / 32768.0 + 1e-12)

    # ZCR
    zc = np.mean(np.abs(np.diff(np.sign(y)))) / 2.0  # approx
    zcr = float(zc)

    # Spectre moyen (FFT sur fenêtres 1s si possible)
    win = sr  # 1 sec
    if len(y) < win:
        seg = y
    else:
        seg = y[:win]
    seg = seg * np.hanning(len(seg)).astype(np.float32)

    spec = np.fft.rfft(seg)
    mag = np.abs(spec) + 1e-12
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)

    total = float(np.sum(mag))
    centroid = float(np.sum(freqs * mag) / total)

    # ratio énergie basse fréquence (<400 Hz)
    low_mask = freqs < 400.0
    low_ratio = float(np.sum(mag[low_mask]) / total)

    # Heuristiques:
    # - ronflement: très basse fréquence + peu de zcr
    # - parole: VAD élevé + centroid plus haut + zcr plus haut
    # - sinon: bruit
    if low_ratio > 0.70 and centroid < 450.0 and zcr < 0.10:
        sound_type = "snore"
    elif voiced_ratio > 0.35 and centroid > 600.0 and zcr >= 0.10:
        sound_type = "voice"
    else:
        sound_type = "noise"

    return {
        "activity": True,
        "sound_type": sound_type,
        "voiced_ratio": float(voiced_ratio),
        "rms": rms,
        "zcr": zcr,
        "centroid": centroid,
        "low_ratio": low_ratio,
    }


def _auto_tag_from_sound(sound_type: str) -> str:
    if sound_type == "snore":
        return "Ronflement"
    if sound_type == "voice":
        return "Parole"
    if sound_type == "noise":
        return "Bruit"
    return "Non tagué"


# -------------------------
# Worker thread
# -------------------------

def processing_worker():
    """
    Passe chaque fichier 1 seule fois:
    - si processing_status == unprocessed -> processing -> done/error
    - ne touche pas aux tags manuels
    """
    print("[VAD] worker started")
    while True:
        try:
            entries = [p for p in REC_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".wav"]
        except FileNotFoundError:
            entries = []

        meta = _load_meta()
        dirty = False

        # 1) s'assurer que tous les fichiers ont une entrée meta
        for p in entries:
            if p.name not in meta:
                st = p.stat()
                _ensure_meta_entry(meta, p.name, recorded_at=int(st.st_mtime))
                dirty = True

        if dirty:
            _save_meta(meta)

        # 2) trouver un job unprocessed
        job = None
        for p in entries:
            entry = meta.get(p.name) or {}
            if entry.get("processing_status") == STATUS_UNPROCESSED:
                job = p
                break

        if job is None:
            time.sleep(1.5)
            continue

        # 3) marquer processing (atomique via meta)
        meta = _load_meta()
        entry = _ensure_meta_entry(meta, job.name, recorded_at=int(job.stat().st_mtime))
        # Si entre temps quelqu’un l’a déjà pris
        if entry.get("processing_status") != STATUS_UNPROCESSED:
            time.sleep(0.2)
            continue

        entry["processing_status"] = STATUS_PROCESSING
        entry["error"] = None
        _save_meta(meta)

        # 4) analyser sans lock
        try:
            res = analyze_audio(job)

            meta = _load_meta()
            entry = _ensure_meta_entry(meta, job.name, recorded_at=entry.get("recorded_at"))

            entry["activity"] = bool(res.get("activity"))
            entry["sound_type"] = res.get("sound_type")
            entry["processed_at"] = int(time.time())
            entry["processing_status"] = STATUS_DONE
            entry["error"] = None

            # tag auto uniquement si pas manual
            tag_source = entry.get("tag_source", "auto")
            if tag_source != "manual":
                new_tag = _auto_tag_from_sound(entry.get("sound_type") or "")
                entry["tag"] = new_tag
                entry["tag_source"] = "auto"

            _save_meta(meta)
            print(f"[VAD] done {job.name}: {entry.get('sound_type')} tag={entry.get('tag')}")

        except Exception as e:
            meta = _load_meta()
            entry = _ensure_meta_entry(meta, job.name, recorded_at=int(job.stat().st_mtime))
            entry["processing_status"] = STATUS_ERROR
            entry["error"] = str(e)
            entry["processed_at"] = int(time.time())
            _save_meta(meta)
            print(f"[VAD] error {job.name}: {e}")

        # petit yield
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

    # meta: recorded_at + status unprocessed
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

        # date/time affichés
        name_date, name_time = _parse_dt_from_name(p.name)
        recorded_at = entry.get("recorded_at")
        if name_date and name_time:
            date_str, time_str = name_date, name_time
            # migration douce: si recorded_at absent, on le remplit
            if recorded_at is None:
                ts = _ts_from_date_time(name_date, name_time)
                if ts is not None:
                    entry["recorded_at"] = ts
                    meta_dirty = True
        else:
            if recorded_at is None:
                entry["recorded_at"] = int(st.st_mtime)
                recorded_at = entry["recorded_at"]
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

    # déplacer la meta si elle existe
    if name in meta:
        meta[body.new_name] = meta.pop(name)

    # garantir cohérence
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
    entry["tag_source"] = "manual"  # IMPORTANT: le worker n'écrase plus
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

    t_vad = threading.Thread(target=processing_worker, daemon=True)
    t_vad.start()

    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    main()
