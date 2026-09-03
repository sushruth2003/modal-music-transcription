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

HUGGINGFACE_SECRET_NAME = "huggingface-secret"
HUGGINGFACE_HUB_PACKAGE = "huggingface-hub==1.29.0"

GPU_TYPE = "L4"
GPU_MAX_CONTAINERS = 4
GPU_SCALEDOWN_WINDOW_SECONDS = 120

AUDIO_SAMPLE_RATE = 16_000
MAX_M1_BATCH_FILES = 4
SUPPORTED_AUDIO_SUFFIXES = frozenset({".flac", ".m4a", ".mp3", ".ogg", ".wav"})

MIN_EXPECTED_CHECKPOINT_BYTES = 5_000_000_000
