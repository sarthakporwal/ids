"""
Model Pruning for Size Reduction
Implements structured and unstructured pruning techniques
"""

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
from pathlib import Path
import json


class ModelPruner:
    """Handles model pruning operations"""
    
    def __init__(self, model):
        """
        Initialize model pruner
        
        Args:
            model: Keras model to prune
        """
        self.model = model
        self.pruning_info = {}
    
    def magnitude_based_pruning(self, x_train, target_sparsity=0.5, 
                                epochs=50, batch_size=128):
        """
        Apply magnitude-based weight pruning
        
        Args:
            x_train: Training data
            target_sparsity: Target sparsity (0.5 = 50% weights pruned)
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Pruned model and history
        """
        print(f"Applying magnitude-based pruning (target sparsity: {target_sparsity})...")
        
        # Define pruning schedule
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=0,
                end_step=len(x_train) // batch_size * epochs
            )
        }
        
        # Apply pruning
        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
            self.model,
            **pruning_params
        )
        
        # Compile
        model_for_pruning.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
        ]
        
        # Train with pruning
        history = model_for_pruning.fit(
            x_train, x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )
        
        # Strip pruning wrappers
        pruned_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        
        self.pruning_info['magnitude'] = {
            'target_sparsity': target_sparsity,
            'epochs': epochs,
            'actual_sparsity': self._compute_sparsity(pruned_model)
        }
        
        return pruned_model, history
    
    def structured_pruning(self, x_train, pruning_ratio=0.3, epochs=50, batch_size=128):
        """
        Apply structured pruning (prune entire filters/neurons)
        
        Args:
            x_train: Training data
            pruning_ratio: Ratio of filters to prune
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Pruned model
        """
        print(f"Applying structured pruning (ratio: {pruning_ratio})...")
        
        # Clone model
        pruned_model = tf.keras.models.clone_model(self.model)
        pruned_model.set_weights(self.model.get_weights())
        
        # Identify layers to prune (Conv2D layers)
        conv_layers = [i for i, layer in enumerate(pruned_model.layers) 
                      if isinstance(layer, tf.keras.layers.Conv2D)]
        
        for layer_idx in conv_layers:
            layer = pruned_model.layers[layer_idx]
            weights = layer.get_weights()
            
            if len(weights) > 0:
                kernel = weights[0]  # Shape: (h, w, in_channels, out_channels)
                
                # Compute filter importance (L1 norm)
                filter_norms = np.sum(np.abs(kernel), axis=(0, 1, 2))
                
                # Determine filters to keep
                num_filters = len(filter_norms)
                num_keep = int(num_filters * (1 - pruning_ratio))
                keep_indices = np.argsort(filter_norms)[-num_keep:]
                
                # Prune filters
                new_kernel = kernel[:, :, :, keep_indices]
                
                if len(weights) > 1:
                    new_bias = weights[1][keep_indices]
                    layer.set_weights([new_kernel, new_bias])
                else:
                    layer.set_weights([new_kernel])
        
        # Recompile and fine-tune
        pruned_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        pruned_model.fit(
            x_train, x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        
        self.pruning_info['structured'] = {
            'pruning_ratio': pruning_ratio,
            'pruned_layers': len(conv_layers)
        }
        
        return pruned_model
    
    def iterative_pruning(self, x_train, x_val, num_iterations=5, 
                         sparsity_per_iteration=0.2, epochs_per_iter=20, batch_size=128):
        """
        Iterative pruning with fine-tuning
        
        Args:
            x_train: Training data
            x_val: Validation data
            num_iterations: Number of pruning iterations
            sparsity_per_iteration: Sparsity increase per iteration
            epochs_per_iter: Epochs per iteration
            batch_size: Batch size
            
        Returns:
            Final pruned model and history
        """
        print(f"Starting iterative pruning ({num_iterations} iterations)...")
        
        current_model = self.model
        history_all = []
        
        for iteration in range(num_iterations):
            current_sparsity = sparsity_per_iteration * (iteration + 1)
            current_sparsity = min(current_sparsity, 0.9)  # Cap at 90%
            
            print(f"\nIteration {iteration+1}/{num_iterations} - Target sparsity: {current_sparsity:.2f}")
            
            pruner = ModelPruner(current_model)
            current_model, history = pruner.magnitude_based_pruning(
                x_train,
                target_sparsity=current_sparsity,
                epochs=epochs_per_iter,
                batch_size=batch_size
            )
            
            # Evaluate on validation
            val_loss = current_model.evaluate(x_val, x_val, verbose=0)
            print(f"Validation loss: {val_loss}")
            
            history_all.append({
                'iteration': iteration + 1,
                'sparsity': current_sparsity,
                'val_loss': float(val_loss),
                'history': history.history
            })
        
        self.pruning_info['iterative'] = {
            'num_iterations': num_iterations,
            'final_sparsity': current_sparsity,
            'history': history_all
        }
        
        return current_model, history_all
    
    def _compute_sparsity(self, model):
        """Compute actual sparsity of model"""
        total_weights = 0
        zero_weights = 0
        
        for layer in model.layers:
            weights = layer.get_weights()
            for w in weights:
                total_weights += w.size
                zero_weights += np.sum(w == 0)
        
        sparsity = zero_weights / total_weights if total_weights > 0 else 0
        return float(sparsity)
    
    def evaluate_pruned_model(self, pruned_model, x_test):
        """
        Evaluate pruned model performance
        
        Args:
            pruned_model: Pruned model
            x_test: Test data
            
        Returns:
            Evaluation metrics
        """
        # Get predictions
        predictions = pruned_model.predict(x_test, verbose=0)
        original_predictions = self.model.predict(x_test, verbose=0)
        
        # Compute metrics
        mse = np.mean(np.square(predictions - x_test))
        mae = np.mean(np.abs(predictions - x_test))
        
        original_mse = np.mean(np.square(original_predictions - x_test))
        
        # Model sizes
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as tmp1:
            pruned_model.save(tmp1.name)
            pruned_size = Path(tmp1.name).stat().st_size
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as tmp2:
            self.model.save(tmp2.name)
            original_size = Path(tmp2.name).stat().st_size
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'original_mse': float(original_mse),
            'accuracy_retention_%': float((1 - abs(mse - original_mse) / original_mse) * 100),
            'pruned_size_mb': float(pruned_size / (1024 * 1024)),
            'original_size_mb': float(original_size / (1024 * 1024)),
            'compression_ratio': float(original_size / pruned_size),
            'sparsity': self._compute_sparsity(pruned_model)
        }
        
        print(f"\nPruned Model Evaluation:")
        print(f"  MSE: {mse:.6f} (Original: {original_mse:.6f})")
        print(f"  Accuracy Retention: {metrics['accuracy_retention_%']:.2f}%")
        print(f"  Size: {metrics['pruned_size_mb']:.2f} MB (Original: {metrics['original_size_mb']:.2f} MB)")
        print(f"  Compression: {metrics['compression_ratio']:.2f}x")
        print(f"  Sparsity: {metrics['sparsity']:.2%}")
        
        return metrics
    
    def export_pruned_model(self, pruned_model, save_path, convert_to_tflite=True):
        """
        Export pruned model
        
        Args:
            pruned_model: Pruned model
            save_path: Save path
            convert_to_tflite: Whether to also export TFLite version
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        pruned_model.save(save_path)
        print(f"Pruned model saved to {save_path}")
        
        # Convert to TFLite
        if convert_to_tflite:
            converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            tflite_path = save_path.with_suffix('.tflite')
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            print(f"TFLite model saved to {tflite_path}")
        
        # Save pruning info
        info_path = save_path.with_suffix('.json')
        with open(info_path, 'w') as f:
            json.dump(self.pruning_info, f, indent=2)
        print(f"Pruning info saved to {info_path}")


def compare_pruning_strategies(model, x_train, x_val, x_test, save_dir):
    """
    Compare different pruning strategies
    
    Args:
        model: Original model
        x_train: Training data
        x_val: Validation data
        x_test: Test data
        save_dir: Save directory
        
    Returns:
        Comparison results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Magnitude-based pruning (50% sparsity)
    print("\n" + "="*70)
    print("Testing Magnitude-Based Pruning (50% sparsity)")
    print("="*70)
    
    pruner1 = ModelPruner(model)
    pruned_model1, _ = pruner1.magnitude_based_pruning(x_train, target_sparsity=0.5, epochs=30)
    results['magnitude_50'] = pruner1.evaluate_pruned_model(pruned_model1, x_test)
    pruner1.export_pruned_model(pruned_model1, save_dir / 'magnitude_50.h5')
    
    # 2. Magnitude-based pruning (70% sparsity)
    print("\n" + "="*70)
    print("Testing Magnitude-Based Pruning (70% sparsity)")
    print("="*70)
    
    pruner2 = ModelPruner(model)
    pruned_model2, _ = pruner2.magnitude_based_pruning(x_train, target_sparsity=0.7, epochs=30)
    results['magnitude_70'] = pruner2.evaluate_pruned_model(pruned_model2, x_test)
    pruner2.export_pruned_model(pruned_model2, save_dir / 'magnitude_70.h5')
    
    # 3. Iterative pruning
    print("\n" + "="*70)
    print("Testing Iterative Pruning")
    print("="*70)
    
    pruner3 = ModelPruner(model)
    pruned_model3, _ = pruner3.iterative_pruning(x_train, x_val, num_iterations=3, 
                                                 sparsity_per_iteration=0.2, epochs_per_iter=15)
    results['iterative'] = pruner3.evaluate_pruned_model(pruned_model3, x_test)
    pruner3.export_pruned_model(pruned_model3, save_dir / 'iterative.h5')
    
    # Save comparison
    with open(save_dir / 'pruning_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("PRUNING COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Method':<20} {'Size (MB)':<12} {'Compression':<12} {'Accuracy %':<12} {'Sparsity':<12}")
    print("-"*70)
    
    for method, metrics in results.items():
        print(f"{method:<20} {metrics['pruned_size_mb']:<12.2f} {metrics['compression_ratio']:<12.2f}x "
              f"{metrics['accuracy_retention_%']:<12.2f} {metrics['sparsity']:<12.2%}")
    
    return results

