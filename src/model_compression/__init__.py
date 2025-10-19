"""
Model Compression Module for Lightweight Deployment
Enables efficient in-vehicle deployment of CANShield
"""

__all__ = ['quantization', 'pruning', 'knowledge_distillation', 'deployment']

from . import quantization
from . import knowledge_distillation
from . import deployment

# Import pruning conditionally to avoid circular import issues
try:
    from . import pruning
except ImportError as e:
    print(f"Warning: Pruning module not available: {e}")

