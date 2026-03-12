"""
QuakeFlow: A comprehensive template matching and magnitude estimation pipeline.

Features:
- Multi-station / multi-component correlation stacking
- FFT, wavelet, and WST correlation domains
- Detection QC metrics (SNR, CC sharpness)
- Multiple magnitude estimation methods (ratio, Bayesian, robust, spectral)
- Station corrections and magnitude uncertainties
- Relative relocation export (HypoDD / GrowClust)
- Template self-updating from high-quality detections
- Real-time / incremental processing mode
- Optional GPU acceleration (CuPy / PyTorch)
"""

__version__ = "2.0.0"
__author__ = "QuakeFlow Team"

from .cli import app, main
from .config import Config
from .core.template_creator import TemplateCreator
from .core.template_matcher import TemplateMatcher
from .core.evaluator import ResultsEvaluator
from .core.relocator import Relocator
from .core.template_updater import TemplateUpdater
from .core.realtime import RealtimeProcessor
from .core.deduplicate import DetectionDeduplicator

__all__ = [
    "app",
    "main",
    "Config",
    "TemplateCreator",
    "TemplateMatcher",
    "ResultsEvaluator",
    "Relocator",
    "TemplateUpdater",
    "RealtimeProcessor",
    "DetectionDeduplicator",
]