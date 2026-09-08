"""Modal resources shared by the transcription pipeline.

Images describe immutable software environments. The Volume stores the static
checkpoint independently from those Images, and the Secret is attached only to
the one-time model downloader that needs it.
"""

from pathlib import Path

import modal

from music_transcription.config import (
    APP_NAME,
    ARTIFACT_VOLUME_NAME,
    BEAT_PACKAGE,
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


def install_score_engine():
    """Extract the checksum-pinned official build; FUSE is not needed at runtime."""
    import hashlib
    import subprocess
    import urllib.request

    url = "https://github.com/musescore/MuseScore/releases/download/v4.7.4/MuseScore-Studio-4.7.4.260706075-x86_64.AppImage"
    expected = "9233ed1b87d3e6b45722278f3c286dcd41e83da778bd0f80a1dd04949696ad93"
    package = Path("/tmp/musescore.AppImage")
    urllib.request.urlretrieve(url, package)
    if hashlib.sha256(package.read_bytes()).hexdigest() != expected:
        raise RuntimeError("MuseScore release checksum mismatch")
    package.chmod(0o755)
    subprocess.run(
        [str(package), "--appimage-extract"], cwd="/opt", check=True, stdout=subprocess.DEVNULL
    )
    Path("/opt/squashfs-root").rename("/opt/musescore")
    package.unlink()


score_base_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install(
        "xvfb",
        "xauth",
        "libglib2.0-0",
        "libasound2",
        "libdbus-1-3",
        "libnss3",
        "libopengl0",
        "libegl1",
        "libgl1",
        "libxkbcommon-x11-0",
        "libxcb-cursor0",
        "libxcomposite1",
        "libxdamage1",
        "libxrandr2",
        "libxi6",
        "libpulse0",
        "fonts-dejavu-core",
        "fonts-freefont-ttf",
    )
    .uv_pip_install("pypdf==6.9.1")
    .run_function(install_score_engine)
)
score_image = score_base_image.add_local_python_source("music_transcription")

web_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .uv_pip_install(FASTAPI_PACKAGE, MULTIPART_PACKAGE)
    .add_local_python_source("music_transcription")
    .add_local_dir(FRONTEND_SOURCE_PATH, remote_path=str(FRONTEND_MOUNT_PATH))
)

model_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(MODEL_PACKAGE, BEAT_PACKAGE)
    .env(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    .add_local_python_source("music_transcription")
)


def fetch_yourmt3_source():
    """Bake only pinned author code into the image; weights live on the Volume."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        "mimbres/YourMT3",
        repo_type="space",
        revision="5e66c1ea173a8186e0d20432b841d3180cc015b5",
        local_dir="/opt/yourmt3",
        allow_patterns=["amt/src/*", "model_helper.py", "LICENSE*"],
    )


yourmt3_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "numpy==1.26.4",
        "transformers==4.45.1",
        "lightning==2.4.0",
        "librosa==0.10.2.post1",
        "einops==0.8.0",
        "wandb==0.18.7",
        "deprecated==1.2.15",
        "mir-eval==0.8.2",
        "mido==1.3.3",
        "soundfile==0.13.1",
        "huggingface-hub==0.25.2",
        "pretty-midi==0.2.11",
    )
    .run_function(fetch_yourmt3_source)
    .env({"WANDB_MODE": "disabled", "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_OFFLINE": "1"})
    .add_local_python_source("music_transcription")
)
