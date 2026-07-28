"""Pluggable glyph-generation backends.

The package defines the contract every generation backend must satisfy and
ships the implementations.  Nothing here participates in the ``native``
fusion path: ``fusion_inference.generate_fusion_and_select`` is untouched and
remains the default, so switching back for an A/B comparison is always a
configuration change rather than a code change.
"""

from .base import (
    BackendResult,
    BackendUnavailable,
    BackendRequest,
    CandidateGeometry,
    GenerationBackend,
    GlyphRequest,
    candidate_filename,
    codepoint_from_filename,
    normalize_candidate,
    read_candidate_dir,
)
from .dir_backend import DirectoryBackend
from .zi2zi_jit import Zi2ziJitBackend

__all__ = [
    "BackendRequest",
    "BackendResult",
    "BackendUnavailable",
    "CandidateGeometry",
    "DirectoryBackend",
    "GenerationBackend",
    "GlyphRequest",
    "Zi2ziJitBackend",
    "candidate_filename",
    "codepoint_from_filename",
    "normalize_candidate",
    "read_candidate_dir",
]
