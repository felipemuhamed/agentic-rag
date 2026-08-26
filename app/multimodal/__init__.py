"""Multimodal input models and visual analysis.

`analyzer` is deliberately not re-exported: the knowledge image installs only
the `knowledge` extra, and importing it here would pull LangChain into that
tier through `app.visual_catalog`.
"""

from app.multimodal.models import ImageInput

__all__ = ["ImageInput"]
