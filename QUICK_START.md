# 🚀 Quick Start - Robust CANShield

## ✅ Your Setup is Complete!

All errors are resolved. You can now train the model!

---

## 📝 Training Commands

### **Option 1: Quick Test (10 minutes)**

For a quick test to verify everything works:

```bash
cd /Users/sarthak/Desktop/Projects/CANShield-main
source canshield_env/bin/activate
cd src
python run_development_canshield.py
```

### **Option 2: Robust CANShield Training (2-3 hours)**

For full adversarially robust training:

```bash
cd /Users/sarthak/Desktop/Projects/CANShield-main
source canshield_env/bin/activate
cd src
python run_robust_canshield.py training_mode=adversarial
```

### **Option 3: Quick Robust Test (30 minutes)**

Edit the config first for faster testing:

```bash
cd /Users/sarthak/Desktop/Projects/CANShield-main

# Edit config (change max_epoch to 20)
nano config/robust_canshield.yaml
# Change line: max_epoch: 100  →  max_epoch: 20

# Then train
source canshield_env/bin/activate
cd src
python run_robust_canshield.py training_mode=adversarial
```

---

## 🎯 Training Modes

| Mode | Command | Time | Features |
|------|---------|------|----------|
| **Standard** | `python run_development_canshield.py` | 1-2h | Original CANShield |
| **Adversarial** | `python run_robust_canshield.py training_mode=adversarial` | 2-3h | Attack-resistant |
| **Domain Adaptive** | `python run_robust_canshield.py training_mode=domain_adaptive` | 2-3h | Cross-vehicle |
| **Bayesian** | `python run_robust_canshield.py training_mode=bayesian` | 2-3h | With uncertainty |

---

## 📊 What Happens During Training

```
Training Process:
├── Loading dataset...              (2 min)
├── Generating time-series images... (5 min)
├── Training autoencoder...         (1-2 hours)
│   ├── Epoch 1/100
│   ├── Epoch 2/100
│   └── ...
├── Evaluating robustness...        (5 min)
├── Model compression...            (10 min)
│   ├── Quantization (Int8)
│   └── Pruning (50% sparsity)
└── Saving models...                (1 min)

Total: 2-3 hours
```

---

## 📁 Output Files

After training, check these locations:

```bash
artifacts/
├── models/syncan/
│   └── robust_canshield_adversarial_50_20_1.h5  ← Main model
│
├── compressed/syncan/
│   ├── quantized_50_1/
│   │   ├── model_int8.tflite     ← Compressed for deployment
│   │   └── model_float16.tflite
│   └── pruned_50_1.h5
│
├── histories/syncan/
│   └── robust_history_*.json     ← Training logs
│
└── robustness/syncan/
    └── robustness_report_*.json  ← Robustness metrics
```

---

## 🔍 Monitor Training Progress

Open a new terminal and watch the artifacts directory:

```bash
# Terminal 1: Training
cd /Users/sarthak/Desktop/Projects/CANShield-main
source canshield_env/bin/activate
cd src
python run_robust_canshield.py training_mode=adversarial

# Terminal 2: Monitor (in another window)
cd /Users/sarthak/Desktop/Projects/CANShield-main
watch -n 5 'ls -lh artifacts/models/syncan/ 2>/dev/null || echo "No models yet..."'
```

---

## ⚠️ Common Issues & Fixes

### Issue: "No module named 'compression'"
**Fixed!** ✅ We renamed it to `model_compression`

### Issue: Environment not activated
```bash
source canshield_env/bin/activate  # Always run this first!
```

### Issue: Out of memory
Edit `config/robust_canshield.yaml`:
```yaml
window_step_train: 20  # Increase to use less data
```

### Issue: Training too slow
Run quick test first:
```bash
python run_development_canshield.py  # Faster, no robustness features
```

---

## 📈 After Training

### 1. Evaluate Results
```bash
python run_robust_evaluation.py
```

### 2. Visualize Results
```bash
python run_visualization_results.py
```

### 3. Check Model Performance
```bash
cd ../artifacts/evaluation_results/syncan/
cat summary_adversarial.txt
```

---

## 🎓 Understanding the Output

### Training Output:
```
Epoch 1/100
Batch 0/100 - Loss: 0.0234, Clean: 0.0198, Adv: 0.0287
Batch 10/100 - Loss: 0.0189, Clean: 0.0165, Adv: 0.0223
...
Epoch Loss: 0.0145, Val Loss: 0.0167, Adv Loss: 0.0189

Evaluating adversarial robustness...
  Robustness Score: 0.8245
  
MODEL COMPRESSION
1. Quantization...
  Compression: 4.20x
  Accuracy Retention: 97.7%
  
2. Pruning...
  Sparsity: 50.0%
  Accuracy Retention: 98.5%

✓ Model saved to: ../artifacts/models/syncan/robust_canshield_adversarial_50_20_1.h5
✓ Metadata saved

TRAINING COMPLETE!
```

### Good Signs:
- ✅ Loss decreasing over epochs
- ✅ Val Loss close to Training Loss
- ✅ Robustness Score > 0.75
- ✅ Accuracy Retention > 95%

### Warning Signs:
- ⚠️ Loss not decreasing → Reduce learning rate
- ⚠️ Val Loss >> Training Loss → Overfitting
- ⚠️ Robustness Score < 0.50 → Need more adversarial training

---

## 🚀 Next Steps

1. ✅ **Train the model** (you're ready!)
2. ✅ **Evaluate results**
3. ✅ **Deploy to vehicle** (see DEPLOYMENT.md)
4. ✅ **Test on real data**

---

## 📚 Documentation

- **This file** - Quick start
- **SETUP_COMPLETE.md** - Setup guide
- **ROBUST_CANSHIELD_GUIDE.md** - Complete features
- **TRAINING_STEPS.md** - Detailed steps
- **README_ROBUST.md** - Project overview

---

## 🎉 Ready to Train!

**Just run:**

```bash
cd /Users/sarthak/Desktop/Projects/CANShield-main
source canshield_env/bin/activate
cd src
python run_robust_canshield.py training_mode=adversarial
```

**Then grab a coffee ☕ and wait 2-3 hours!**

---

**Good luck! 🚗🔒**

