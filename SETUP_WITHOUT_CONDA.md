# Setup CANShield Without Conda - macOS Guide

## Option 1: Install Python 3.9 Using Homebrew (Recommended)

### Step 1: Install Homebrew (if not installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Install Python 3.9
```bash
# Install Python 3.9
brew install python@3.9

# Verify installation
/opt/homebrew/bin/python3.9 --version
# or for Intel Mac:
/usr/local/bin/python3.9 --version
```

### Step 3: Create Virtual Environment
```bash
cd /Users/sarthak/Desktop/Projects/CANShield-main

# Create virtual environment with Python 3.9
/opt/homebrew/bin/python3.9 -m venv canshield_env
# or for Intel Mac:
# /usr/local/bin/python3.9 -m venv canshield_env

# Activate it
source canshield_env/bin/activate
```

### Step 4: Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install tensorflow==2.10.0
pip install keras==2.10.0
pip install numpy==1.26.4
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install hydra-core
pip install omegaconf
pip install tqdm

# Install tensorflow-model-optimization
pip install tensorflow-model-optimization
```

### Step 5: Download Dataset
```bash
cd src
chmod +x download_syncan_dataset.sh
./download_syncan_dataset.sh
```

### Step 6: Run Training
```bash
# Make sure environment is activated
source ../canshield_env/bin/activate

# Run training
python run_robust_canshield.py training_mode=adversarial
```

---

## Option 2: Use pyenv (Python Version Manager)

### Step 1: Install pyenv
```bash
# Install pyenv
brew install pyenv

# Add to shell configuration
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init --path)"' >> ~/.zshrc

# Restart terminal or:
source ~/.zshrc
```

### Step 2: Install Python 3.9
```bash
# Install Python 3.9.18 (latest 3.9)
pyenv install 3.9.18

# Set as local version for this project
cd /Users/sarthak/Desktop/Projects/CANShield-main
pyenv local 3.9.18

# Verify
python --version  # Should show Python 3.9.18
```

### Step 3: Create Virtual Environment
```bash
python -m venv canshield_env
source canshield_env/bin/activate
```

### Step 4: Install Dependencies (same as Option 1 Step 4)
```bash
pip install --upgrade pip
pip install tensorflow==2.10.0 keras==2.10.0 numpy==1.26.4
pip install pandas scikit-learn matplotlib seaborn
pip install hydra-core omegaconf tqdm
pip install tensorflow-model-optimization
```

---

## Option 3: Quick Script Installation

Save this as `quick_setup.sh`:

```bash
#!/bin/bash

echo "=== CANShield Quick Setup (No Conda) ==="

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install Python 3.9
echo "Installing Python 3.9..."
brew install python@3.9

# Navigate to project
cd /Users/sarthak/Desktop/Projects/CANShield-main

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "/opt/homebrew/bin/python3.9" ]; then
    /opt/homebrew/bin/python3.9 -m venv canshield_env
else
    /usr/local/bin/python3.9 -m venv canshield_env
fi

# Activate environment
source canshield_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing TensorFlow and dependencies..."
pip install tensorflow==2.10.0
pip install keras==2.10.0
pip install numpy==1.26.4
pip install pandas scikit-learn matplotlib seaborn
pip install hydra-core omegaconf tqdm
pip install tensorflow-model-optimization

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To activate the environment in future:"
echo "  cd /Users/sarthak/Desktop/Projects/CANShield-main"
echo "  source canshield_env/bin/activate"
echo ""
echo "To run training:"
echo "  cd src"
echo "  python run_robust_canshield.py training_mode=adversarial"
```

Run it:
```bash
chmod +x quick_setup.sh
./quick_setup.sh
```

---

## Simplified Training Script (No Pruning/TF-MOT)

If you still have issues with `tensorflow-model-optimization`, you can modify the code to skip pruning:

### Create: `src/run_robust_canshield_simple.py`

```python
"""Simplified Robust CANShield - Without Pruning"""

import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent))

from dataset.load_dataset import *
from training.get_autoencoder import get_autoencoder
from adversarial.adversarial_training import train_robust_autoencoder
from adversarial.robustness_metrics import RobustnessEvaluator
from compression.quantization import ModelQuantizer  # Only quantization, no pruning

@hydra.main(version_base=None, config_path="../config", config_name="robust_canshield")
def train_robust_canshield(args: DictConfig) -> None:
    print("="*70)
    print("ROBUST CANSHIELD - Simplified Training (No Pruning)")
    print("="*70)
    
    root_dir = Path(__file__).resolve().parent
    args.root_dir = root_dir
    args.data_type = "training"
    args.data_dir = args.train_data_dir
    
    # ... rest of training code but skip pruning section ...

if __name__ == "__main__":
    train_robust_canshield()
```

---

## Minimal Setup (Just Run Original CANShield)

If you just want to test the system quickly:

```bash
# Use your current Python 3.13
cd /Users/sarthak/Desktop/Projects/CANShield-main

# Create venv
python3 -m venv simple_env
source simple_env/bin/activate

# Install minimal dependencies
pip install numpy pandas scikit-learn matplotlib tqdm

# Install compatible TensorFlow for Python 3.13 (nightly build)
pip install tf-nightly

# Run original CANShield (without robust features)
cd src
python run_development_canshield.py
```

---

## Troubleshooting

### Issue: "python3.9: command not found"
```bash
# Find Python installations
ls /usr/local/bin/python*
ls /opt/homebrew/bin/python*

# Use full path
/opt/homebrew/bin/python3.9 -m venv canshield_env
```

### Issue: "No module named 'tensorflow'"
```bash
# Make sure venv is activated (you should see (canshield_env) in prompt)
source canshield_env/bin/activate

# Reinstall TensorFlow
pip install --force-reinstall tensorflow==2.10.0
```

### Issue: "SSL Certificate Error"
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org tensorflow==2.10.0
```

---

## Quick Commands Reference

```bash
# Always start with:
cd /Users/sarthak/Desktop/Projects/CANShield-main
source canshield_env/bin/activate

# Check Python version
python --version  # Should be 3.9.x

# Check TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"

# Run training
cd src
python run_robust_canshield.py training_mode=adversarial

# Deactivate when done
deactivate
```

