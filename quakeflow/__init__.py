"""
QuakeFlow: A comprehensive template matching and magnitude estimation pipeline.
"""

__version__ = "1.0.0"
__author__ = "QuakeFlow Team"

from .cli import app, main
from .config import Config
from .core.template_creator import TemplateCreator
from .core.template_matcher import TemplateMatcher
from .core.evaluator import ResultsEvaluator

__all__ = [
    "app",
    "main",
    "Config",
    "TemplateCreator",
    "TemplateMatcher",
    "ResultsEvaluator",
]