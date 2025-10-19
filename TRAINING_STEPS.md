# Step-by-Step Training Guide for Robust CANShield

This guide provides detailed instructions for training and deploying Robust CANShield.

---

## 📋 Prerequisites Checklist

Before starting, ensure you have:
- [ ] Python 3.9 installed
- [ ] CUDA 11.7+ (for GPU training)
- [ ] 16GB+ RAM
- [ ] 50GB+ free disk space
- [ ] Internet connection for dataset download

---

## 🔧 Step 1: Environment Setup

### 1.1 Clone Repository
```bash
git clone https://github.com/shahriar0651/CANShield.git
cd CANShield
```

### 1.2 Install Mambaforge
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh
chmod +x Mambaforge-$(uname)-$(uname -m).sh
./Mambaforge-$(uname)-$(uname -m).sh
```

**Follow prompts and restart terminal**

### 1.3 Create Conda Environment
```bash
conda env create --file dependency/environment_v1.yaml
conda activate canshield
```

### 1.4 Install Additional Dependencies
```bash
pip install tensorflow-model-optimization scikit-learn
```

### 1.5 Verify Installation
```bash
python src/test_gpu.py
python src/test_keras.py
```

**Expected output**: GPU detected, Keras working

---

## 📥 Step 2: Download Dataset

### 2.1 Download SynCAN Dataset
```bash
cd src
chmod +x download_syncan_dataset.sh
./download_syncan_dataset.sh
```

**This downloads ~2GB of data**

### 2.2 Verify Dataset Structure
```bash
ls -la ../datasets/can-ids/syncan/
```

**Expected structure**:
```
datasets/can-ids/syncan/
├── ambients/         # Training data
│   ├── train_1.csv
│   ├── train_2.csv
│   ├── train_3.csv
│   └── train_4.csv
└── attacks/          # Test data
    ├── test_flooding.csv
    ├── test_suppress.csv
    ├── test_plateau.csv
    ├── test_continuous.csv
    └── test_playback.csv
```

---

## 🎯 Step 3: Choose Training Mode

### Option A: Quick Training (30 minutes)
**For testing and quick results**

Edit `config/robust_canshield.yaml`:
```yaml
max_epoch: 20
time_steps: [50]
sampling_periods: [1]
window_step_train: 10
```

```bash
python run_robust_canshield.py training_mode=adversarial
```

---

### Option B: Standard Training (2-3 hours)
**For good results**

Edit `config/robust_canshield.yaml`:
```yaml
max_epoch: 100
time_steps: [50]
sampling_periods: [1, 5, 10]
window_step_train: 10
```

```bash
python run_robust_canshield.py training_mode=adversarial
```

---

### Option C: Full Training (8-12 hours)
**For best results matching paper**

Edit `config/robust_canshield.yaml`:
```yaml
max_epoch: 500
time_steps: [25, 50, 75, 100]
sampling_periods: [1, 5, 10, 20, 50]
window_step_train: 1
```

```bash
python run_robust_canshield.py training_mode=adversarial
```

---

## 🚀 Step 4: Training Execution

### 4.1 Start Training
```bash
cd src
python run_robust_canshield.py training_mode=adversarial
```

### 4.2 Monitor Training
**You should see**:
```
==================================================
ROBUST CANSHIELD - Adversarially Robust CAN-IDS
==================================================

Training Configuration:
  Dataset: syncan
  Training Mode: adversarial
  Use Compression: True
  Use Uncertainty: True

Loading training data...
Loading file 1/4: train_1
  Loaded 50000 samples
...

Epoch 1/100
Batch 0/100 - Loss: 0.0234, Clean: 0.0198, Adv: 0.0287
...
```

### 4.3 Training Progress
Training will go through multiple phases:

**Phase 1**: Standard Autoencoder Training
- Epochs 1-30: Learning basic reconstruction

**Phase 2**: Adversarial Training
- Epochs 31-70: Training with FGSM attacks (ε=0.005)
- Epochs 71-100: Training with PGD attacks (ε=0.01)

**Phase 3**: Robustness Evaluation
- Testing against multiple attacks
- Computing robustness scores

**Phase 4**: Model Compression
- Quantization (Int8, Float16)
- Pruning (50% sparsity)

### 4.4 Expected Training Time
| Configuration | Time | GPU | CPU |
|--------------|------|-----|-----|
| Quick | 30 min | RTX 2080 Ti | 2 hours |
| Standard | 2-3 hours | RTX 2080 Ti | 8 hours |
| Full | 8-12 hours | RTX 2080 Ti | 48 hours |

---

## 📊 Step 5: Evaluation

### 5.1 Run Comprehensive Evaluation
```bash
python run_robust_evaluation.py
```

### 5.2 Check Results
```bash
cat ../artifacts/evaluation_results/syncan/summary_adversarial.txt
```

**Example output**:
```
Configuration: ts50_sp1

Attack: Flooding
  Standard Performance:
    - Accuracy:  0.9871
    - F1-Score:  0.9682
    - Precision: 0.9745
    - Recall:    0.9621
  Robustness:
    - Overall Score: 0.8245
  Uncertainty:
    - Mean Confidence: 0.8932
  Performance:
    - Inference Time: 8.34 ms
```

### 5.3 View Detailed Results
```bash
# JSON format with all metrics
cat ../artifacts/evaluation_results/syncan/comprehensive_evaluation_adversarial.json
```

---

## 🚀 Step 6: Model Deployment

### 6.1 Locate Trained Models
```bash
ls ../artifacts/models/syncan/
```

You should see:
- `robust_canshield_adversarial_50_20_1.h5` (main model)
- Compressed versions in `../artifacts/compressed/`

### 6.2 Test Deployment (TFLite)
```python
from compression.deployment import EdgeDeployment

# Load compressed model
deployment = EdgeDeployment(
    '../artifacts/compressed/syncan/quantized_50_1/model_int8.tflite',
    model_type='tflite'
)

# Test inference
sample_input = x_test[:1]
prediction = deployment.predict(sample_input)

# Benchmark
benchmark_results = deployment.benchmark(sample_input, num_iterations=100)
print(f"Inference time: {benchmark_results['mean_ms']:.2f} ms")
```

### 6.3 Create Deployment Package
```python
from compression.deployment import create_deployment_package

config = {
    'model_name': 'robust_canshield_adversarial',
    'version': '1.0',
    'threshold': 0.01
}

create_deployment_package(
    model_path='../artifacts/models/syncan/robust_canshield_adversarial_50_20_1.h5',
    config=config,
    output_dir='../deployment_package'
)
```

---

## 🔍 Step 7: Verify Results

### 7.1 Check Model Performance
Expected metrics for adversarial training:
- **F1-Score**: >0.94 on all attack types
- **Robustness Score**: >0.80
- **Inference Time**: <10ms
- **Model Size**: <5MB (compressed)

### 7.2 Compare with Baseline
```python
import json

# Load results
with open('../artifacts/evaluation_results/syncan/comprehensive_evaluation_adversarial.json', 'r') as f:
    results = json.load(f)

# Print comparison
for config, attacks in results.items():
    print(f"\nConfiguration: {config}")
    for attack, metrics in attacks.items():
        f1 = metrics['standard_metrics']['f1_score']
        rob = metrics['robustness']['overall_score']['robustness_score']
        print(f"  {attack}: F1={f1:.4f}, Robustness={rob:.4f}")
```

---

## 🛠️ Step 8: Troubleshooting

### Issue 1: CUDA Out of Memory
**Solution**: Reduce batch size in config
```yaml
# In run_robust_canshield.py, line ~100
batch_size: 64  # Instead of 128
```

### Issue 2: Dataset Not Found
**Solution**: Re-download dataset
```bash
cd src
rm -rf ../datasets/can-ids/syncan
./download_syncan_dataset.sh
```

### Issue 3: Slow Training
**Solution**: Use GPU or reduce data
```yaml
window_step_train: 20  # Instead of 10
per_of_samples: 0.5    # Use 50% of data
```

### Issue 4: Model Not Saving
**Solution**: Create directories manually
```bash
mkdir -p ../artifacts/models/syncan
mkdir -p ../artifacts/compressed/syncan
mkdir -p ../artifacts/evaluation_results/syncan
```

---

## 📈 Step 9: Advanced Training

### 9.1 Domain Adaptive Training (Cross-Vehicle)
```bash
python run_robust_canshield.py training_mode=domain_adaptive
```

**Use case**: When you have data from multiple vehicle types

### 9.2 Bayesian Training (with Uncertainty)
```bash
python run_robust_canshield.py training_mode=bayesian
```

**Use case**: When you need confidence scores for detections

### 9.3 Custom Training
Create custom training script:
```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../config", config_name="robust_canshield")
def custom_training(args: DictConfig):
    # Your custom training logic here
    pass

if __name__ == "__main__":
    custom_training()
```

---

## 🎯 Step 10: Production Deployment

### 10.1 Export for Embedded System
```bash
python -c "
from compression.quantization import ModelQuantizer
import tensorflow as tf

model = tf.keras.models.load_model('../artifacts/models/syncan/robust_canshield_adversarial_50_20_1.h5')
quantizer = ModelQuantizer(model)

# For ARM-based ECU
tflite_model = quantizer.quantize_to_int8(
    representative_dataset=x_train[:1000],
    save_path='../deployment/canshield_embedded.tflite'
)
"
```

### 10.2 Real-time Monitoring Setup
```python
from compression.deployment import RealtimeCANMonitor, EdgeDeployment

# Initialize
deployment = EdgeDeployment('canshield_embedded.tflite', model_type='tflite')
monitor = RealtimeCANMonitor(deployment, threshold=0.01, window_size=50, num_signals=20)

# Process CAN frames
while True:
    can_signals = read_can_bus()  # Your CAN reading function
    result = monitor.process_packet(can_signals)
    
    if result['anomaly']:
        trigger_alert(result)
```

---

## ✅ Success Checklist

After completing all steps, you should have:

- [x] Trained robust model (`.h5` file)
- [x] Compressed models (`.tflite` files)
- [x] Evaluation results (JSON + text reports)
- [x] Deployment package
- [x] Benchmark results

---

## 📚 Next Steps

1. **Fine-tune**: Adjust hyperparameters for your specific use case
2. **Transfer**: Adapt model to new vehicle types
3. **Deploy**: Integrate with your CAN bus monitoring system
4. **Monitor**: Track performance in production
5. **Update**: Retrain with new attack data

---

## 🆘 Getting Help

If you encounter issues:
1. Check troubleshooting section above
2. Review log files in `../artifacts/`
3. Ensure GPU is being used: `nvidia-smi`
4. Verify dataset integrity
5. Open GitHub issue with error logs

---

## 📧 Support

For additional help:
- GitHub Issues: [Project Repository]
- Documentation: `ROBUST_CANSHIELD_GUIDE.md`
- Original Paper: [CANShield Paper](https://arxiv.org/abs/2205.01306)

---

**Happy Training! 🚗💻🔒**

