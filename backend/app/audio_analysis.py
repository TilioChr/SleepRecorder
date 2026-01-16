import wave
from dataclasses import dataclass

import numpy as np

try:
    import webrtcvad
except Exception:  # pragma: no cover
    webrtcvad = None


@dataclass
class AnalysisResult:
    has_activity: bool
    kind: str  # "silence" | "voice" | "snore" | "noise"
    confidence: float


def _read_wav_mono_s16(path: str):
    """Retourne (sr, pcm_int16_mono)."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        if sw != 2:
            raise ValueError("Only 16-bit PCM WAV supported")
        n = wf.getnframes()
        raw = wf.readframes(n)

    x = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return sr, x


def _frame_rms(x: np.ndarray, frame_len: int, hop: int):
    if len(x) < frame_len:
        return np.array([], dtype=np.float32)
    n = 1 + (len(x) - frame_len) // hop
    rms = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = i * hop
        frame = x[s : s + frame_len].astype(np.float32)
        rms[i] = np.sqrt(np.mean(frame * frame) + 1e-12)
    return rms


def _simple_activity_mask(x: np.ndarray, sr: int, frame_ms: int = 30):
    frame_len = int(sr * frame_ms / 1000)
    hop = frame_len
    rms = _frame_rms(x, frame_len, hop)
    if rms.size == 0:
        return frame_len, hop, np.array([], dtype=bool)

    # bruit de fond approximé via médiane
    noise = float(np.median(rms))
    thr = max(noise * 3.0, 250.0)  # seuil minimum empirique
    return frame_len, hop, rms > thr


def _vad_voice_ratio(x: np.ndarray, sr: int, frame_ms: int = 30):
    """Retourne un ratio [0..1] de frames classées "speech" par WebRTC VAD."""
    if webrtcvad is None:
        return 0.0
    if sr not in (8000, 16000, 32000, 48000):
        return 0.0
    vad = webrtcvad.Vad(2)
    frame_len = int(sr * frame_ms / 1000)
    n = len(x) // frame_len
    if n <= 0:
        return 0.0

    speech = 0
    for i in range(n):
        frame = x[i * frame_len : (i + 1) * frame_len]
        if len(frame) != frame_len:
            break
        if vad.is_speech(frame.tobytes(), sr):
            speech += 1
    return speech / max(n, 1)


def _snore_vs_noise_features(
    x: np.ndarray, sr: int, active_mask: np.ndarray, frame_len: int, hop: int
):
    """Retourne (low_ratio, centroid_hz) agrégés sur frames actives."""
    if active_mask.size == 0 or not bool(active_mask.any()):
        return 0.0, 0.0

    # FFT sur frames actives
    nfft = 1
    while nfft < frame_len:
        nfft *= 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
    low_band = freqs <= 300.0

    low_ratios = []
    centroids = []

    for i, is_active in enumerate(active_mask):
        if not is_active:
            continue
        s = i * hop
        frame = x[s : s + frame_len].astype(np.float32)
        if len(frame) != frame_len:
            continue

        # fenêtre légère
        frame *= np.hanning(frame_len).astype(np.float32)
        spec = np.abs(np.fft.rfft(frame, n=nfft))
        pwr = spec * spec
        total = float(pwr.sum()) + 1e-12

        low = float(pwr[low_band].sum())
        low_ratios.append(low / total)

        centroid = float((freqs * pwr).sum() / total)
        centroids.append(centroid)

    if not low_ratios:
        return 0.0, 0.0

    return float(np.median(low_ratios)), float(np.median(centroids))


def analyze_wav_file(path: str) -> AnalysisResult:
    """Analyse simple: activité -> (voix | ronflement | bruit)."""
    sr, x = _read_wav_mono_s16(path)

    frame_len, hop, active = _simple_activity_mask(x, sr, frame_ms=30)
    has_activity = bool(active.any())
    if not has_activity:
        return AnalysisResult(False, "silence", 1.0)

    voice_ratio = _vad_voice_ratio(x, sr, frame_ms=30)
    # Si beaucoup de frames VAD positives: on classe voix
    if voice_ratio >= 0.20:
        conf = min(1.0, 0.5 + voice_ratio)
        return AnalysisResult(True, "voice", float(conf))

    low_ratio, centroid = _snore_vs_noise_features(x, sr, active, frame_len, hop)

    # Heuristique ronflement: énergie majoritairement <300Hz + centroid bas
    if low_ratio >= 0.60 and centroid <= 600.0:
        conf = float(min(1.0, (low_ratio - 0.45) * 1.8))
        return AnalysisResult(True, "snore", conf)

    # sinon bruit d'environnement
    conf = float(min(1.0, 0.55 + (centroid / 4000.0)))
    return AnalysisResult(True, "noise", conf)
