"""Pinned model identity, durable resource names, and safety limits."""

from pathlib import Path

APP_NAME = "music-transcription"
PYTHON_VERSION = "3.12"

MODEL_REPO_ID = "MuScriptor/muscriptor-large"
MODEL_REVISION = "8809fdfbed2affa7ade94a7059e746e3880720e7"
MODEL_PACKAGE = "muscriptor==0.3.0"
BEAT_PACKAGE = "beat-this==1.1.0"
MODEL_VOLUME_NAME = "music-transcription-models"
MODEL_MOUNT_PATH = Path("/models")
MODEL_SNAPSHOT_PATH = MODEL_MOUNT_PATH / "muscriptor-large" / MODEL_REVISION
MODEL_CHECKPOINT_PATH = MODEL_SNAPSHOT_PATH / "model.safetensors"
MODEL_CONFIG_PATH = MODEL_SNAPSHOT_PATH / "config.json"
MODEL_READY_PATH = MODEL_SNAPSHOT_PATH / "READY.json"

# Beat This! final0 is the checkpoint used by muscriptor==0.3.0 for beat-grid
# detection. Keep it beside MuScriptor's weights so inference never downloads
# model data at runtime.
BEAT_CHECKPOINT_NAME = "final0"
BEAT_CHECKPOINT_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt"
BEAT_CHECKPOINT_SHA256 = "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331"
BEAT_CHECKPOINT_BYTES = 81_058_141
BEAT_CHECKPOINT_PATH = MODEL_MOUNT_PATH / "beat-this" / "beat_this-final0.ckpt"

ARTIFACT_VOLUME_NAME = "music-transcription-artifacts"
ARTIFACT_MOUNT_PATH = Path("/artifacts")
JOB_DICT_NAME = "music-transcription-jobs"
RATE_LIMIT_DICT_NAME = "music-transcription-rate-limits"

HUGGINGFACE_SECRET_NAME = "huggingface-secret"
HUGGINGFACE_HUB_PACKAGE = "huggingface-hub==1.29.0"

GPU_TYPE = "L4"
GPU_MAX_CONTAINERS = 1
GPU_SCALEDOWN_WINDOW_SECONDS = 30

BEAT_MAX_CONTAINERS = GPU_MAX_CONTAINERS
BEAT_SCALEDOWN_WINDOW_SECONDS = GPU_SCALEDOWN_WINDOW_SECONDS

AUDIO_SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 10 * 60
MAX_M1_BATCH_FILES = 4
SUPPORTED_AUDIO_SUFFIXES = frozenset({".flac", ".m4a", ".mp3", ".ogg", ".wav"})
SUPPORTED_VIDEO_SUFFIXES = frozenset({".mkv", ".mov", ".mp4", ".webm"})
SUPPORTED_SOURCE_SUFFIXES = SUPPORTED_AUDIO_SUFFIXES | SUPPORTED_VIDEO_SUFFIXES

# Exact MT3_FULL_PLUS names accepted by the pinned muscriptor==0.3.0 package.
# An empty selection means unconstrained instrument detection.
MUSCRIPTOR_INSTRUMENT_GROUPS = (
    ("Keyboards", ("acoustic_piano", "electric_piano", "organ")),
    (
        "Guitars & bass",
        (
            "acoustic_guitar",
            "clean_electric_guitar",
            "distorted_electric_guitar",
            "acoustic_bass",
            "electric_bass",
        ),
    ),
    (
        "Strings & voice",
        (
            "violin",
            "viola",
            "cello",
            "contrabass",
            "orchestral_harp",
            "string_ensemble",
            "synth_strings",
            "voice",
        ),
    ),
    ("Percussion", ("chromatic_percussion", "timpani", "drums")),
    ("Brass", ("trumpet", "trombone", "tuba", "french_horn", "brass_section")),
    (
        "Woodwinds",
        (
            "soprano_and_alto_sax",
            "tenor_sax",
            "baritone_sax",
            "oboe",
            "english_horn",
            "bassoon",
            "clarinet",
            "flutes",
        ),
    ),
    ("Synths & effects", ("orchestra_hit", "synth_lead", "synth_pad")),
)
MUSCRIPTOR_INSTRUMENT_NAMES = tuple(
    name for _group, names in MUSCRIPTOR_INSTRUMENT_GROUPS for name in names
)

FASTAPI_PACKAGE = "fastapi==0.141.1"
MULTIPART_PACKAGE = "python-multipart==0.0.32"
FRONTEND_MOUNT_PATH = Path("/frontend")
WEB_MAX_CONTAINERS = 1
WEB_MAX_CONCURRENT_INPUTS = 25
WEB_TARGET_CONCURRENT_INPUTS = 10
WEB_MAX_UPLOAD_BYTES = 100 * 1024 * 1024
WEB_UPLOAD_CHUNK_BYTES = 1024 * 1024
WEB_SUBMISSIONS_PER_IP_HOUR = 2
WEB_SUBMISSIONS_GLOBAL_DAY = 5
WEB_RATE_LIMIT_WINDOW_SECONDS = 60 * 60
PUBLIC_BETA_MONTHLY_BUDGET_USD = 10.0
# Reserve a deliberately conservative fixed amount before each job. Modal's
# workspace budget remains the authoritative cap because this estimate cannot
# include every startup, CPU, storage, or retry charge.
PUBLIC_BETA_JOB_RESERVATION_USD = 0.25
SCORE_RENDER_TIMEOUT_SECONDS = 5 * 60

# Modal's published L4 rate when this milestone was implemented. This is only
# used for a clearly labelled inference-time estimate, not as a billing record.
L4_PRICE_PER_SECOND_USD = 0.000222

MIN_EXPECTED_CHECKPOINT_BYTES = 5_000_000_000
