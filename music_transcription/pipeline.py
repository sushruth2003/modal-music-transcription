"""The deployed transcription workers and M2 web application.

Deploy this module once. The separate local client looks up ``process_job`` by
name, so submissions and status checks do not create new ephemeral App versions.
"""

# Register the web and worker stages on the shared App before Modal builds the graph.
from music_transcription import api as _api  # noqa: F401
from music_transcription import score as _score  # noqa: F401
from music_transcription import transcribe as _transcribe  # noqa: F401
from music_transcription.benchmarks import deployment as _benchmark  # noqa: F401
from music_transcription.preprocess import process_job
from music_transcription.resources import app

__all__ = ["app", "process_job"]
