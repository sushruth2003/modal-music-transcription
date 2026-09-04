"""Protected HTTP entrypoint used only for controlled M2 benchmarks."""

from typing import Any

import modal

from music_transcription.api import create_web_app
from music_transcription.config import (
    WEB_MAX_CONCURRENT_INPUTS,
    WEB_MAX_CONTAINERS,
    WEB_TARGET_CONCURRENT_INPUTS,
)
from music_transcription.resources import app, web_image


@app.function(
    image=web_image,
    cpu=1.0,
    memory=1024,
    timeout=15 * 60,
    max_containers=WEB_MAX_CONTAINERS,
)
@modal.concurrent(
    max_inputs=WEB_MAX_CONCURRENT_INPUTS,
    target_inputs=WEB_TARGET_CONCURRENT_INPUTS,
)
@modal.asgi_app(requires_proxy_auth=True)
def benchmark_web() -> Any:
    """Expose the real pipeline without public-demo quotas, behind Modal's proxy."""

    return create_web_app(enforce_submission_limits=False)
