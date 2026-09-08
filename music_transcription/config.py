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

# Production YourMT3+ MoE identity; legacy MuScriptor pins remain for benchmarks/rollback.
YOURMT3_REPO_ID = "mimbres/YourMT3"
YOURMT3_REVISION = "5e66c1ea173a8186e0d20432b841d3180cc015b5"
YOURMT3_EXPERIMENT = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops"
YOURMT3_CHECKPOINT_RELATIVE = f"amt/logs/2024/{YOURMT3_EXPERIMENT}/checkpoints/last.ckpt"
YOURMT3_SNAPSHOT_PATH = MODEL_MOUNT_PATH / "yourmt3-moe" / YOURMT3_REVISION
YOURMT3_CHECKPOINT_PATH = YOURMT3_SNAPSHOT_PATH / YOURMT3_CHECKPOINT_RELATIVE
YOURMT3_READY_PATH = YOURMT3_SNAPSHOT_PATH / "READY.json"
YOURMT3_CHECKPOINT_BYTES = 561_544_628
YOURMT3_CHECKPOINT_SHA256 = "ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25"
YOURMT3_ARGS = [
    f"{YOURMT3_EXPERIMENT}@last.ckpt",
    "-p",
    "2024",
    "-pr",
    "32",
    "-tk",
    "mc13_full_plus_256",
    "-dec",
    "multi-t5",
    "-nl",
    "26",
    "-enc",
    "perceiver-tf",
    "-sqr",
    "1",
    "-ff",
    "moe",
    "-wf",
    "4",
    "-nmoe",
    "8",
    "-kmoe",
    "2",
    "-act",
    "silu",
    "-epe",
    "rope",
    "-rp",
    "1",
    "-ac",
    "spec",
    "-hop",
    "300",
    "-atc",
    "1",
]
INSTRUMENT_GROUPS = MUSCRIPTOR_INSTRUMENT_GROUPS + (
    ("Singing", ("singing_voice", "singing_voice_chorus")),
)
INSTRUMENT_NAMES = tuple(name for _, names in INSTRUMENT_GROUPS for name in names)

SCORE_ENGINE_VERSION = "4.7.4"
SCORE_ENGINE_URL = "https://github.com/musescore/MuseScore/releases/download/v4.7.4/MuseScore-Studio-4.7.4.260706075-x86_64.AppImage"
SCORE_ENGINE_SHA256 = "9233ed1b87d3e6b45722278f3c286dcd41e83da778bd0f80a1dd04949696ad93"
SCORE_ENGINE_COMMAND = "/opt/musescore/AppRun"
