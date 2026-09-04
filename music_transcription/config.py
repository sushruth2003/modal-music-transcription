"""Pinned model identity, durable resource names, and safety limits."""

from pathlib import Path

APP_NAME = "music-transcription"
PYTHON_VERSION = "3.12"

MODEL_REPO_ID = "MuScriptor/muscriptor-large"
MODEL_REVISION = "8809fdfbed2affa7ade94a7059e746e3880720e7"
MODEL_PACKAGE = "muscriptor==0.3.0"
MODEL_VOLUME_NAME = "music-transcription-models"
MODEL_MOUNT_PATH = Path("/models")
MODEL_SNAPSHOT_PATH = MODEL_MOUNT_PATH / "muscriptor-large" / MODEL_REVISION
MODEL_CHECKPOINT_PATH = MODEL_SNAPSHOT_PATH / "model.safetensors"
MODEL_CONFIG_PATH = MODEL_SNAPSHOT_PATH / "config.json"
MODEL_READY_PATH = MODEL_SNAPSHOT_PATH / "READY.json"

ARTIFACT_VOLUME_NAME = "music-transcription-artifacts"
ARTIFACT_MOUNT_PATH = Path("/artifacts")
JOB_DICT_NAME = "music-transcription-jobs"
RATE_LIMIT_DICT_NAME = "music-transcription-rate-limits"

HUGGINGFACE_SECRET_NAME = "huggingface-secret"
HUGGINGFACE_HUB_PACKAGE = "huggingface-hub==1.29.0"

GPU_TYPE = "L4"
GPU_MAX_CONTAINERS = 4
GPU_SCALEDOWN_WINDOW_SECONDS = 120

AUDIO_SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 10 * 60
MAX_M1_BATCH_FILES = 4
SUPPORTED_AUDIO_SUFFIXES = frozenset({".flac", ".m4a", ".mp3", ".ogg", ".wav"})
URL_SOURCE_SUFFIX = ".flac"
SUPPORTED_MEDIA_HOSTS = frozenset(
    {
        "instagram.com",
        "www.instagram.com",
        "m.instagram.com",
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
    }
)
URL_DOWNLOAD_PACKAGE = "yt-dlp[default]==2026.8.19"
DENO_VERSION = "2.9.6"
URL_DOWNLOAD_TIMEOUT_SECONDS = 10 * 60

FASTAPI_PACKAGE = "fastapi==0.141.1"
MULTIPART_PACKAGE = "python-multipart==0.0.32"
FRONTEND_MOUNT_PATH = Path("/frontend")
WEB_MAX_CONTAINERS = 1
WEB_MAX_CONCURRENT_INPUTS = 25
WEB_TARGET_CONCURRENT_INPUTS = 10
WEB_MAX_UPLOAD_BYTES = 100 * 1024 * 1024
WEB_UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_INSTRUMENT_HINTS = 16
WEB_SUBMISSIONS_PER_IP_HOUR = 3
WEB_SUBMISSIONS_GLOBAL_DAY = 10
WEB_RATE_LIMIT_WINDOW_SECONDS = 60 * 60
SCORE_RENDER_TIMEOUT_SECONDS = 5 * 60

# Modal's published L4 rate when this milestone was implemented. This is only
# used for a clearly labelled inference-time estimate, not as a billing record.
L4_PRICE_PER_SECOND_USD = 0.000222

MIN_EXPECTED_CHECKPOINT_BYTES = 5_000_000_000
