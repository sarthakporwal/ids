# Installation Fix for macOS

## Problem
You're using Python 3.13, but the project requires Python 3.9 with specific dependencies.

## Solution

### Step 1: Install Mambaforge (Conda Alternative)

```bash
# Download Mambaforge for macOS
cd ~/Downloads
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-MacOSX-$(uname -m).sh"

# Make executable
chmod +x Mambaforge-MacOSX-$(uname -m).sh

# Install (follow prompts, say YES to initialize)
./Mambaforge-MacOSX-$(uname -m).sh

# IMPORTANT: Restart your terminal after installation
```

### Step 2: Verify Installation

Close and reopen your terminal, then:

```bash
conda --version
# Should show: conda 23.x.x or similar
```

### Step 3: Create CANShield Environment

```bash
cd /Users/sarthak/Desktop/Projects/CANShield-main

# Create environment with Python 3.9
conda env create --file dependency/environment_v1.yaml

# This will:
# - Create 'canshield' environment
# - Install Python 3.9
# - Install TensorFlow 2.10
# - Install all required dependencies
```

### Step 4: Activate Environment

```bash
conda activate canshield

# Verify Python version
python --version
# Should show: Python 3.9.x
```

### Step 5: Install Additional Dependency

```bash
# Now install tensorflow-model-optimization
pip install tensorflow-model-optimization

# Should work now because you're in Python 3.9 environment
```

### Step 6: Test Installation

```bash
cd src

# Test GPU (if available)
python test_gpu.py

# Test Keras
python test_keras.py
```

### Step 7: Run Training

```bash
# Now you can run the robust training
python run_robust_canshield.py training_mode=adversarial
```

---

## Alternative: Use Existing Python 3.9

If you already have Python 3.9 installed somewhere, you can use venv:

```bash
# Find Python 3.9
which python3.9

# If it exists, create venv
python3.9 -m venv canshield_env
source canshield_env/bin/activate

# Install dependencies
pip install tensorflow==2.10.0 keras==2.10.0
pip install numpy==1.26.4 pandas scikit-learn
pip install hydra-core omegaconf tqdm
pip install tensorflow-model-optimization
```

---

## Quick Fix Script

Save this as `setup.sh` and run it:

```bash
#!/bin/bash

echo "Installing Mambaforge..."
cd ~/Downloads
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-MacOSX-$(uname -m).sh"
chmod +x Mambaforge-MacOSX-$(uname -m).sh
./Mambaforge-MacOSX-$(uname -m).sh -b -p $HOME/mambaforge

echo "Initializing conda..."
$HOME/mambaforge/bin/conda init zsh

echo "Please restart your terminal and run:"
echo "  cd /Users/sarthak/Desktop/Projects/CANShield-main"
echo "  conda env create --file dependency/environment_v1.yaml"
echo "  conda activate canshield"
echo "  pip install tensorflow-model-optimization"
```

Run it:
```bash
chmod +x setup.sh
./setup.sh
```

---

## Troubleshooting

### Issue: "conda: command not found" after installation
**Solution**: Restart terminal or run:
```bash
source ~/mambaforge/bin/activate
```

### Issue: Environment creation fails
**Solution**: Try manual creation:
```bash
conda create -n canshield python=3.9
conda activate canshield
pip install tensorflow==2.10.0 keras==2.10.0 numpy pandas scikit-learn hydra-core omegaconf tqdm
pip install tensorflow-model-optimization
```

### Issue: NumPy compilation fails
**Solution**: Install Xcode Command Line Tools:
```bash
xcode-select --install
```

---

## After Setup

Once everything is installed:

```bash
# Always activate environment first
conda activate canshield

# Then run training
cd /Users/sarthak/Desktop/Projects/CANShield-main/src
python run_robust_canshield.py training_mode=adversarial
```

