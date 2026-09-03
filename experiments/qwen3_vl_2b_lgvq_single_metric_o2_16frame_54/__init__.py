"""Single-metric, 16-frame LGVQ optical/electronic VQA experiment.

The package keeps only the frozen *front* of Qwen3-VL: official preprocessing,
Vision patch/position embeddings, and Language token embeddings.  Qwen Vision
blocks, merger, Language blocks, attention, and the LM head are deliberately
outside the train/deploy graph.
"""

from .settings import ExperimentSettings, Geometry, load_settings

__all__ = ["ExperimentSettings", "Geometry", "load_settings"]
