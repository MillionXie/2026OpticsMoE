"""Standalone experiment packages.

Fixed-feedback projects were moved physically under ``FixedFeedbackSFT`` in
September 2026.  Their established ``experiments.<name>`` module names remain
the only supported Python interface so old commands, manifests and checkpoint
metadata continue to resolve without duplicate module identities.
"""

from pathlib import Path


_FIXED_FEEDBACK_PROJECTS = (
    Path(__file__).resolve().parent.parent / "FixedFeedbackSFT" / "projects"
)
if _FIXED_FEEDBACK_PROJECTS.is_dir():
    __path__.append(str(_FIXED_FEEDBACK_PROJECTS))
