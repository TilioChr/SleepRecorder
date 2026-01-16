import os

AUDIO_TCP_HOST = "0.0.0.0"
AUDIO_TCP_PORT = int(os.getenv("AUDIO_TCP_PORT", "5000"))

API_HOST = "0.0.0.0"
API_PORT = int(os.getenv("API_PORT", "8000"))

RECORDINGS_DIR = os.getenv("RECORDINGS_DIR", "/data/recordings")

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHANNELS = int(os.getenv("CHANNELS", "1"))
SAMPLE_WIDTH_BYTES = int(os.getenv("SAMPLE_WIDTH_BYTES", "2"))  # s16le
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "5"))
