"""Modal resources shared by the transcription pipeline.

Images describe immutable software environments. The Volume stores the static
checkpoint independently from those Images, and the Secret is attached only to
the CPU downloader that needs it.
"""

from pathlib import Path

import modal

from music_transcription.config import (
    APP_NAME,
    ARTIFACT_VOLUME_NAME,
    DENO_VERSION,
    FASTAPI_PACKAGE,
    FRONTEND_MOUNT_PATH,
    HUGGINGFACE_HUB_PACKAGE,
    HUGGINGFACE_SECRET_NAME,
    JOB_DICT_NAME,
    MODEL_PACKAGE,
    MODEL_VOLUME_NAME,
    MULTIPART_PACKAGE,
    PYTHON_VERSION,
    RATE_LIMIT_DICT_NAME,
    URL_DOWNLOAD_PACKAGE,
)

FRONTEND_SOURCE_PATH = Path(__file__).parent / "frontend"

app = modal.App(APP_NAME)

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)
job_states = modal.Dict.from_name(JOB_DICT_NAME, create_if_missing=True)
rate_limit_states = modal.Dict.from_name(RATE_LIMIT_DICT_NAME, create_if_missing=True)
huggingface_secret = modal.Secret.from_name(HUGGINGFACE_SECRET_NAME)

download_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .uv_pip_install(HUGGINGFACE_HUB_PACKAGE)
    .add_local_python_source("music_transcription")
)

audio_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("ffmpeg")
    .add_local_python_source("music_transcription")
)

_deno_archive = "deno-x86_64-unknown-linux-gnu.zip"
_deno_release = f"https://github.com/denoland/deno/releases/download/v{DENO_VERSION}"

ingest_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("ca-certificates", "curl", "ffmpeg", "unzip")
    .run_commands(
        f"curl -fsSLo /tmp/{_deno_archive} {_deno_release}/{_deno_archive}",
        f"curl -fsSLo /tmp/{_deno_archive}.sha256sum "
        f"{_deno_release}/{_deno_archive}.sha256sum",
        f"cd /tmp && sha256sum -c {_deno_archive}.sha256sum",
        f"unzip -q /tmp/{_deno_archive} -d /usr/local/bin",
        "deno --version",
    )
    .uv_pip_install(URL_DOWNLOAD_PACKAGE)
    .add_local_python_source("music_transcription")
)

score_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("musescore3")
    .add_local_python_source("music_transcription")
)

web_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .uv_pip_install(FASTAPI_PACKAGE, MULTIPART_PACKAGE)
    .add_local_python_source("music_transcription")
    .add_local_dir(FRONTEND_SOURCE_PATH, remote_path=str(FRONTEND_MOUNT_PATH))
)

model_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(MODEL_PACKAGE)
    .env(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    .add_local_python_source("music_transcription")
)
