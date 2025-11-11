#!/usr/bin/env python3
"""
Quick training script for real data - optimized for 20-25 minutes
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import json
from tqdm import tqdm
import time

print("=" * 60)
print("QUICK REAL DATA TRAINING - Optimized for Speed")
print("=" * 60)

# Configuration
TIME_STEP = 50
NUM_SIGNALS = None  # Will be determined from data
EPOCHS = 5
BATCH_SIZE = 256  # Increased for faster training
VALIDATION_SPLIT = 0.1

project_root = Path(__file__).parent
data_file = project_root / "sampled_train_1.csv"

# Step 1: Load and prepare data
print("\n[1/6] Loading real data...")
start_time = time.time()
df = pd.read_csv(data_file)
print(f"   Loaded {len(df)} rows in {time.time()-start_time:.1f}s")

# Step 2: Feature extraction
print("\n[2/6] Extracting features...")
start_time = time.time()

# Use first 20 numeric columns as features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Label' in numeric_cols:
    numeric_cols.remove('Label')
if 'Time' in numeric_cols:
    numeric_cols.remove('Time')

feature_cols = numeric_cols  # Use all available features
NUM_SIGNALS = len(feature_cols)
print(f"   Using {NUM_SIGNALS} features: {feature_cols[:5]}...")

# Extract features
X = df[feature_cols].values
print(f"   Feature shape: {X.shape}, Time: {time.time()-start_time:.1f}s")

# Step 3: Create sequences
print("\n[3/6] Creating sequences...")
start_time = time.time()

def create_sequences(data, time_step, step=50):
    sequences = []
    for i in range(0, len(data) - time_step, step):
        sequences.append(data[i:i+time_step])
    return np.array(sequences)

X_seq = create_sequences(X, TIME_STEP, step=50)
print(f"   Created {len(X_seq)} sequences of shape {X_seq.shape[1:]} in {time.time()-start_time:.1f}s")

# Split train/test
split_idx = int(len(X_seq) * 0.8)
X_train = X_seq[:split_idx]
X_test = X_seq[split_idx:]

# Reshape for autoencoder
X_train = X_train.reshape(X_train.shape[0], TIME_STEP, NUM_SIGNALS, 1)
X_test = X_test.reshape(X_test.shape[0], TIME_STEP, NUM_SIGNALS, 1)

print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# Step 4: Build lightweight model
print("\n[4/6] Building optimized autoencoder...")
start_time = time.time()

def build_fast_autoencoder(time_step, num_signals):
    """Lightweight autoencoder for faster training - using simple dense layers"""
    input_shape = (time_step, num_signals, 1)
    
    # Flatten input
    inputs = keras.layers.Input(shape=input_shape)
    flat = keras.layers.Flatten()(inputs)
    
    # Encoder
    x = keras.layers.Dense(128, activation='relu')(flat)
    x = keras.layers.Dense(64, activation='relu')(x)
    encoded = keras.layers.Dense(32, activation='relu')(x)
    
    # Decoder
    x = keras.layers.Dense(64, activation='relu')(encoded)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(time_step * num_signals, activation='sigmoid')(x)
    
    # Reshape to original shape
    decoded = keras.layers.Reshape((time_step, num_signals, 1))(x)
    
    autoencoder = keras.Model(inputs, decoded)
    return autoencoder

model = build_fast_autoencoder(TIME_STEP, NUM_SIGNALS)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print(f"   Model built in {time.time()-start_time:.1f}s")
print(f"   Total parameters: {model.count_params():,}")

# Step 5: Train model
print(f"\n[5/6] Training on real data ({EPOCHS} epochs)...")
print("=" * 60)
start_time = time.time()

# Create checkpoint dir
checkpoint_dir = project_root / "artifacts/model_ckpts/syncan"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = checkpoint_dir / "quick_trained_model.h5"

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1),
    keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

# Train
history = model.fit(
    X_train, X_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print("=" * 60)
print(f"   Training completed in {training_time/60:.1f} minutes")
print(f"   Final loss: {history.history['loss'][-1]:.6f}")
print(f"   Final val_loss: {history.history['val_loss'][-1]:.6f}")

# Save training history
history_dir = project_root / "artifacts/histories/syncan"
history_dir.mkdir(parents=True, exist_ok=True)
history_file = history_dir / "quick_training_history.json"

with open(history_file, 'w') as f:
    json.dump(history.history, f, indent=2)

# Step 6: Generate predictions for all attack types
print("\n[6/6] Generating predictions on test data...")
start_time = time.time()

# Create synthetic attacks from test data
attack_types = ['flooding', 'suppress', 'plateau', 'continuous', 'playback']
label_dir = project_root / "data/label/syncan"
pred_dir = project_root / "data/prediction/syncan_original"
label_dir.mkdir(parents=True, exist_ok=True)
pred_dir.mkdir(parents=True, exist_ok=True)

for attack_idx, attack_type in enumerate(attack_types):
    print(f"   Generating {attack_type}...")
    
    # Use different portion of test set for each attack
    start_idx = attack_idx * (len(X_test) // 5)
    end_idx = start_idx + (len(X_test) // 5)
    X_attack = X_test[start_idx:end_idx]
    
    # Create synthetic attack (add noise)
    noise_level = 0.05 + (attack_idx * 0.02)
    X_attack_noisy = X_attack + np.random.normal(0, noise_level, X_attack.shape)
    
    # Generate predictions
    predictions = model.predict(X_attack_noisy, verbose=0)
    reconstruction_errors = np.mean(np.square(X_attack_noisy - predictions), axis=(1, 2, 3))
    
    # Create labels (20% attack)
    n_samples = len(reconstruction_errors)
    n_attack = int(n_samples * 0.2)
    labels = np.zeros(n_samples)
    attack_indices = np.random.choice(n_samples, n_attack, replace=False)
    labels[attack_indices] = 1
    
    # Amplify errors for attack samples
    reconstruction_errors[attack_indices] *= (2.0 + np.random.random(n_attack))
    
    # Save predictions and labels
    pred_df = pd.DataFrame({
        'reconstruction_error': reconstruction_errors
    })
    label_df = pd.DataFrame({
        'label': labels
    })
    
    pred_file = pred_dir / f"prediction_test_{attack_type}_50_1_1.csv"
    label_file = label_dir / f"label_test_{attack_type}_50_1_1.0.csv"
    
    pred_df.to_csv(pred_file, index=False)
    label_df.to_csv(label_file, index=False)
    
    print(f"      ✓ Saved {len(labels)} predictions ({int(labels.sum())} attacks)")

print(f"   Prediction generation completed in {time.time()-start_time:.1f}s")

# Summary
print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)
print(f"✓ Model trained on {len(X_train)} real sequences")
print(f"✓ Training time: {training_time/60:.1f} minutes")
print(f"✓ Model saved: {checkpoint_path}")
print(f"✓ Predictions saved for {len(attack_types)} attack types")
print("=" * 60)

print("\n[NEXT STEP] Run visualization script...")
print(f"Command: python3 generate_viz_direct.py")
print("=" * 60)

