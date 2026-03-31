# WSI Inference Module
# Makes functions importable from this package

from .classify_wsi import WSIClassifier, get_available_models

__all__ = [
    'WSIClassifier',
    'get_available_models',
]
