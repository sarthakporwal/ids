
import sys
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Testing imports...")

try:
    import tensorflow as tf
    print(f"✓ TensorFlow: {tf.__version__}")
except Exception as e:
    print(f"✗ TensorFlow error: {e}")
    sys.exit(1)

try:
    import keras
    print(f"✓ Keras: {keras.__version__}")
except Exception as e:
    print(f"✗ Keras error: {e}")
    sys.exit(1)

try:
    import tensorflow_model_optimization as tfmot
    print("✓ TensorFlow Model Optimization")
except Exception as e:
    print(f"Note: TF-MOT warning (non-critical): {e}")
    print("  Pruning features may not work, but quantization will")

try:
    import numpy, pandas, sklearn, hydra, omegaconf
    print("✓ NumPy, Pandas, Scikit-learn, Hydra")
except Exception as e:
    print(f"✗ Dependencies error: {e}")
    sys.exit(1)

print("\n✓✓✓ All critical dependencies are working! ✓✓✓")
print("\nYou can now run:")
print("  python run_robust_canshield.py training_mode=adversarial")

