"""
Transfer Learning Module for Cross-Vehicle Adaptation
Enables efficient knowledge transfer between vehicle models
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from pathlib import Path
import json


class TransferLearningManager:
    """Manages transfer learning between vehicle models"""
    
    def __init__(self, source_model, freeze_layers=True):
        """
        Initialize transfer learning manager
        
        Args:
            source_model: Pre-trained source model
            freeze_layers: Whether to freeze encoder layers
        """
        self.source_model = source_model
        self.freeze_layers = freeze_layers
    
    def create_target_model(self, fine_tune_layers=2):
        """
        Create target model from source with selective fine-tuning
        
        Args:
            fine_tune_layers: Number of layers to fine-tune (from end)
            
        Returns:
            Target model ready for fine-tuning
        """
        # Clone the model
        target_model = keras.models.clone_model(self.source_model)
        target_model.set_weights(self.source_model.get_weights())
        
        if self.freeze_layers:
            # Freeze all layers except last few
            total_layers = len(target_model.layers)
            for i, layer in enumerate(target_model.layers):
                if i < total_layers - fine_tune_layers:
                    layer.trainable = False
                else:
                    layer.trainable = True
        
        return target_model
    
    def progressive_fine_tuning(self, target_model, x_target, epochs_per_stage=10,
                               batch_size=128, validation_split=0.1):
        """
        Progressive fine-tuning: gradually unfreeze layers
        
        Args:
            target_model: Model to fine-tune
            x_target: Target domain data
            epochs_per_stage: Epochs per unfreezing stage
            batch_size: Batch size
            validation_split: Validation split
            
        Returns:
            Fine-tuned model and history
        """
        # Compile model
        target_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='mse',
            metrics=['mae']
        )
        
        histories = []
        
        # Stage 1: Train only last layers
        print("Stage 1: Training last layers...")
        history = target_model.fit(
            x_target, x_target,
            epochs=epochs_per_stage,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        histories.append(history.history)
        
        # Stage 2: Unfreeze more layers
        total_layers = len(target_model.layers)
        for i, layer in enumerate(target_model.layers):
            if i >= total_layers // 2:  # Unfreeze second half
                layer.trainable = True
        
        # Re-compile with lower learning rate
        target_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00005),
            loss='mse',
            metrics=['mae']
        )
        
        print("Stage 2: Training second half of layers...")
        history = target_model.fit(
            x_target, x_target,
            epochs=epochs_per_stage,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        histories.append(history.history)
        
        # Stage 3: Fine-tune all layers
        for layer in target_model.layers:
            layer.trainable = True
        
        target_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00001),
            loss='mse',
            metrics=['mae']
        )
        
        print("Stage 3: Fine-tuning all layers...")
        history = target_model.fit(
            x_target, x_target,
            epochs=epochs_per_stage,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        histories.append(history.history)
        
        return target_model, histories
    
    def few_shot_adaptation(self, x_few_shot, epochs=20, learning_rate=0.001):
        """
        Quick adaptation with few samples
        
        Args:
            x_few_shot: Few-shot examples from target vehicle
            epochs: Number of epochs
            learning_rate: Learning rate
            
        Returns:
            Adapted model
        """
        # Create model with last layers unfrozen
        adapted_model = self.create_target_model(fine_tune_layers=3)
        
        # Compile with higher learning rate for quick adaptation
        adapted_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Train on few-shot data with data augmentation
        history = adapted_model.fit(
            x_few_shot, x_few_shot,
            epochs=epochs,
            batch_size=min(32, len(x_few_shot)),
            verbose=1
        )
        
        return adapted_model, history


class MultiSourceTransferLearning:
    """Transfer learning from multiple source vehicles"""
    
    def __init__(self, source_models, vehicle_names):
        """
        Initialize multi-source transfer learning
        
        Args:
            source_models: List of pre-trained models
            vehicle_names: Names of source vehicles
        """
        self.source_models = source_models
        self.vehicle_names = vehicle_names
        self.num_sources = len(source_models)
    
    def create_ensemble_transfer_model(self, time_step, num_signals):
        """
        Create ensemble model combining multiple sources
        
        Args:
            time_step: Time steps
            num_signals: Number of signals
            
        Returns:
            Ensemble model
        """
        input_shape = (time_step, num_signals, 1)
        inputs = layers.Input(shape=input_shape)
        
        # Get predictions from all source models
        predictions = []
        for model in self.source_models:
            # Freeze source model
            for layer in model.layers:
                layer.trainable = False
            
            pred = model(inputs)
            predictions.append(pred)
        
        # Combine predictions with learned weights
        if len(predictions) > 1:
            # Learnable ensemble weights
            ensemble_weights = layers.Dense(self.num_sources, activation='softmax',
                                           name='ensemble_weights')(layers.Flatten()(inputs))
            
            # Weighted combination
            combined = predictions[0]
            for i in range(1, len(predictions)):
                combined = layers.Add()([combined, predictions[i]])
            
            combined = layers.Lambda(lambda x: x / self.num_sources)(combined)
        else:
            combined = predictions[0]
        
        # Additional adaptation layers
        x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(combined)
        output = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x)
        
        model = Model(inputs, output, name='multi_source_transfer')
        return model
    
    def weighted_average_transfer(self, x_target, sample_weights=None):
        """
        Create model by weighted averaging of source models
        
        Args:
            x_target: Target domain samples for weight computation
            sample_weights: Optional predefined weights
            
        Returns:
            Weighted averaged model
        """
        if sample_weights is None:
            # Compute weights based on performance on target samples
            sample_weights = self._compute_source_weights(x_target)
        
        # Get model architecture from first source
        base_model = keras.models.clone_model(self.source_models[0])
        
        # Weighted average of weights
        averaged_weights = []
        num_layers = len(self.source_models[0].layers)
        
        for layer_idx in range(num_layers):
            layer_weights = []
            for model in self.source_models:
                if len(model.layers[layer_idx].get_weights()) > 0:
                    layer_weights.append(model.layers[layer_idx].get_weights())
            
            if len(layer_weights) > 0:
                # Weighted average
                avg_weight = []
                for weight_idx in range(len(layer_weights[0])):
                    weighted_sum = np.zeros_like(layer_weights[0][weight_idx])
                    for src_idx, weights in enumerate(layer_weights):
                        weighted_sum += sample_weights[src_idx] * weights[weight_idx]
                    avg_weight.append(weighted_sum)
                averaged_weights.append(avg_weight)
            else:
                averaged_weights.append([])
        
        # Set averaged weights
        for layer_idx, weights in enumerate(averaged_weights):
            if len(weights) > 0:
                base_model.layers[layer_idx].set_weights(weights)
        
        return base_model
    
    def _compute_source_weights(self, x_target):
        """Compute source model weights based on target performance"""
        reconstruction_errors = []
        
        for model in self.source_models:
            predictions = model.predict(x_target, verbose=0)
            error = np.mean(np.square(predictions - x_target))
            reconstruction_errors.append(error)
        
        # Convert errors to weights (lower error = higher weight)
        errors_array = np.array(reconstruction_errors)
        weights = 1.0 / (errors_array + 1e-10)
        weights = weights / np.sum(weights)
        
        return weights


class AdaptiveNormalization:
    """Adaptive normalization for cross-vehicle transfer"""
    
    @staticmethod
    def create_model_with_adaptive_norm(base_model, time_step, num_signals):
        """
        Add adaptive batch normalization layers
        
        Args:
            base_model: Base model
            time_step: Time steps
            num_signals: Number of signals
            
        Returns:
            Model with adaptive normalization
        """
        input_shape = (time_step, num_signals, 1)
        inputs = layers.Input(shape=input_shape)
        
        x = inputs
        
        # Add batch normalization between layers
        for i, layer in enumerate(base_model.layers):
            x = layer(x)
            
            # Add batch norm after conv layers
            if isinstance(layer, layers.Conv2D) and i < len(base_model.layers) - 2:
                x = layers.BatchNormalization(trainable=True, name=f'adaptive_bn_{i}')(x)
        
        model = Model(inputs, x, name='adaptive_norm_model')
        return model
    
    @staticmethod
    def domain_specific_batch_norm(model, x_source, x_target, alpha=0.1):
        """
        Apply domain-specific batch normalization statistics
        
        Args:
            model: Model with batch norm layers
            x_source: Source domain data
            x_target: Target domain data
            alpha: Interpolation factor
            
        Returns:
            Model with adapted batch norm
        """
        # Get batch norm layers
        bn_layers = [layer for layer in model.layers if isinstance(layer, layers.BatchNormalization)]
        
        # Compute statistics on both domains
        _ = model.predict(x_source, verbose=0)
        source_stats = {layer.name: (layer.moving_mean.numpy(), layer.moving_variance.numpy()) 
                       for layer in bn_layers}
        
        _ = model.predict(x_target, verbose=0)
        target_stats = {layer.name: (layer.moving_mean.numpy(), layer.moving_variance.numpy()) 
                       for layer in bn_layers}
        
        # Interpolate statistics
        for layer in bn_layers:
            source_mean, source_var = source_stats[layer.name]
            target_mean, target_var = target_stats[layer.name]
            
            adapted_mean = (1 - alpha) * source_mean + alpha * target_mean
            adapted_var = (1 - alpha) * source_var + alpha * target_var
            
            layer.moving_mean.assign(adapted_mean)
            layer.moving_variance.assign(adapted_var)
        
        return model


def save_transfer_learning_checkpoint(model, vehicle_name, save_dir):
    """
    Save transfer learning checkpoint
    
    Args:
        model: Model to save
        vehicle_name: Name of vehicle
        save_dir: Directory to save
    """
    save_path = Path(save_dir) / f"transfer_model_{vehicle_name}.h5"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    
    # Save metadata
    metadata = {
        'vehicle_name': vehicle_name,
        'model_architecture': model.name,
        'num_layers': len(model.layers),
        'trainable_params': int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    }
    
    with open(save_path.parent / f"metadata_{vehicle_name}.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Transfer learning checkpoint saved to {save_path}")


def load_transfer_learning_checkpoint(vehicle_name, save_dir):
    """
    Load transfer learning checkpoint
    
    Args:
        vehicle_name: Name of vehicle
        save_dir: Directory to load from
        
    Returns:
        Loaded model and metadata
    """
    load_path = Path(save_dir) / f"transfer_model_{vehicle_name}.h5"
    model = keras.models.load_model(load_path)
    
    metadata_path = load_path.parent / f"metadata_{vehicle_name}.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return model, metadata

