"""
Model Quantization for Efficient Deployment
Reduces model size and inference time through quantization
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import json


class ModelQuantizer:
    """Handles various quantization strategies"""
    
    def __init__(self, model):
        """
        Initialize model quantizer
        
        Args:
            model: Keras model to quantize
        """
        self.model = model
        self.quantization_info = {}
    
    def quantize_to_float16(self, save_path=None):
        """
        Convert model to float16 precision
        
        Args:
            save_path: Optional path to save quantized model
            
        Returns:
            Float16 model
        """
        print("Quantizing model to float16...")
        
        # Convert to float16
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        self.quantization_info['float16'] = {
            'size_bytes': len(tflite_model),
            'precision': 'float16'
        }
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(tflite_model)
            print(f"Float16 model saved to {save_path}")
        
        return tflite_model
    
    def quantize_to_int8(self, representative_dataset, save_path=None):
        """
        Convert model to int8 quantization (full integer quantization)
        
        Args:
            representative_dataset: Dataset for calibration
            save_path: Optional path to save
            
        Returns:
            Int8 quantized model
        """
        print("Quantizing model to int8...")
        
        def representative_data_gen():
            for i in range(min(100, len(representative_dataset))):
                sample = representative_dataset[i:i+1]
                yield [sample.astype(np.float32)]
        
        # Convert to int8
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()
        
        self.quantization_info['int8'] = {
            'size_bytes': len(tflite_model),
            'precision': 'int8',
            'calibration_samples': min(100, len(representative_dataset))
        }
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(tflite_model)
            print(f"Int8 model saved to {save_path}")
        
        return tflite_model
    
    def dynamic_range_quantization(self, save_path=None):
        """
        Apply dynamic range quantization (weights only)
        
        Args:
            save_path: Optional path to save
            
        Returns:
            Quantized model
        """
        print("Applying dynamic range quantization...")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        self.quantization_info['dynamic'] = {
            'size_bytes': len(tflite_model),
            'type': 'dynamic_range'
        }
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(tflite_model)
            print(f"Dynamic quantized model saved to {save_path}")
        
        return tflite_model
    
    def quantization_aware_training(self, x_train, epochs=50, batch_size=128):
        """
        Perform quantization-aware training
        
        Args:
            x_train: Training data
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Quantization-aware trained model
        """
        print("Starting quantization-aware training...")
        
        import tensorflow_model_optimization as tfmot
        
        # Apply quantization-aware training
        quantize_model = tfmot.quantization.keras.quantize_model
        
        q_aware_model = quantize_model(self.model)
        
        # Compile
        q_aware_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Train
        history = q_aware_model.fit(
            x_train, x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        self.quantization_info['qat'] = {
            'size_bytes': len(tflite_model),
            'type': 'quantization_aware_training',
            'epochs': epochs
        }
        
        return q_aware_model, tflite_model, history
    
    def evaluate_quantized_model(self, tflite_model, x_test, model_type='float32'):
        """
        Evaluate quantized model performance
        
        Args:
            tflite_model: TFLite model bytes
            x_test: Test data
            model_type: Type of quantization
            
        Returns:
            Evaluation metrics
        """
        print(f"Evaluating {model_type} quantized model...")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Evaluate on test set
        predictions = []
        sample_size = min(500, len(x_test))
        
        for i in range(sample_size):
            # Prepare input
            input_data = x_test[i:i+1].astype(input_details[0]['dtype'])
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predictions.append(output_data)
        
        predictions = np.concatenate(predictions, axis=0)
        x_test_sample = x_test[:sample_size]
        
        # Compute metrics
        mse = np.mean(np.square(predictions - x_test_sample))
        mae = np.mean(np.abs(predictions - x_test_sample))
        
        # Compare with original model
        original_predictions = self.model.predict(x_test_sample, verbose=0)
        original_mse = np.mean(np.square(original_predictions - x_test_sample))
        
        accuracy_retention = (1 - abs(mse - original_mse) / original_mse) * 100
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'original_mse': float(original_mse),
            'accuracy_retention_%': float(accuracy_retention),
            'model_size_bytes': len(tflite_model),
            'model_size_kb': len(tflite_model) / 1024,
            'compression_ratio': self._get_original_size() / len(tflite_model)
        }
        
        print(f"MSE: {mse:.6f}, MAE: {mae:.6f}")
        print(f"Accuracy retention: {accuracy_retention:.2f}%")
        print(f"Model size: {metrics['model_size_kb']:.2f} KB")
        print(f"Compression ratio: {metrics['compression_ratio']:.2f}x")
        
        return metrics
    
    def _get_original_size(self):
        """Get original model size in bytes"""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as tmp:
            self.model.save(tmp.name)
            return Path(tmp.name).stat().st_size
    
    def compare_quantization_methods(self, x_train, x_test, save_dir=None):
        """
        Compare different quantization methods
        
        Args:
            x_train: Training data (for calibration)
            x_test: Test data
            save_dir: Directory to save results
            
        Returns:
            Comparison results
        """
        print("Comparing quantization methods...\n")
        
        results = {}
        
        # 1. Float16
        print("=" * 50)
        print("Testing Float16 Quantization")
        print("=" * 50)
        tflite_fp16 = self.quantize_to_float16()
        results['float16'] = self.evaluate_quantized_model(tflite_fp16, x_test, 'float16')
        
        # 2. Dynamic Range
        print("\n" + "=" * 50)
        print("Testing Dynamic Range Quantization")
        print("=" * 50)
        tflite_dynamic = self.dynamic_range_quantization()
        results['dynamic'] = self.evaluate_quantized_model(tflite_dynamic, x_test, 'dynamic')
        
        # 3. Int8
        print("\n" + "=" * 50)
        print("Testing Int8 Quantization")
        print("=" * 50)
        tflite_int8 = self.quantize_to_int8(x_train[:1000])
        results['int8'] = self.evaluate_quantized_model(tflite_int8, x_test, 'int8')
        
        # Save results
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            with open(save_dir / 'quantization_comparison.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save models
            with open(save_dir / 'model_float16.tflite', 'wb') as f:
                f.write(tflite_fp16)
            with open(save_dir / 'model_dynamic.tflite', 'wb') as f:
                f.write(tflite_dynamic)
            with open(save_dir / 'model_int8.tflite', 'wb') as f:
                f.write(tflite_int8)
            
            print(f"\nQuantization comparison saved to {save_dir}")
        
        # Print summary
        print("\n" + "=" * 70)
        print("QUANTIZATION COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Method':<15} {'Size (KB)':<12} {'MSE':<12} {'Retention %':<15} {'Compression':<12}")
        print("-" * 70)
        
        for method, metrics in results.items():
            print(f"{method:<15} {metrics['model_size_kb']:<12.2f} {metrics['mse']:<12.6f} "
                  f"{metrics['accuracy_retention_%']:<15.2f} {metrics['compression_ratio']:<12.2f}x")
        
        return results


def quantize_ensemble_models(models, model_names, x_train, x_test, save_dir):
    """
    Quantize multiple models in an ensemble
    
    Args:
        models: List of models
        model_names: Names of models
        x_train: Training data
        x_test: Test data
        save_dir: Save directory
        
    Returns:
        Quantization results for all models
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for model, name in zip(models, model_names):
        print(f"\n{'='*70}")
        print(f"Quantizing model: {name}")
        print(f"{'='*70}")
        
        quantizer = ModelQuantizer(model)
        
        model_save_dir = save_dir / name
        results = quantizer.compare_quantization_methods(x_train, x_test, model_save_dir)
        
        all_results[name] = results
    
    # Save combined results
    with open(save_dir / 'ensemble_quantization_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nEnsemble quantization results saved to {save_dir}")
    
    return all_results


class TFLiteInference:
    """Efficient inference with TFLite models"""
    
    def __init__(self, tflite_model_path):
        """
        Initialize TFLite inference
        
        Args:
            tflite_model_path: Path to TFLite model
        """
        self.interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def predict(self, x_input):
        """
        Run inference
        
        Args:
            x_input: Input data
            
        Returns:
            Predictions
        """
        # Prepare input
        input_data = x_input.astype(self.input_details[0]['dtype'])
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return output_data
    
    def batch_predict(self, x_batch, batch_size=32):
        """
        Batch inference
        
        Args:
            x_batch: Batch of inputs
            batch_size: Batch size
            
        Returns:
            Predictions
        """
        predictions = []
        
        for i in range(0, len(x_batch), batch_size):
            batch = x_batch[i:i+batch_size]
            for sample in batch:
                pred = self.predict(sample[np.newaxis, ...])
                predictions.append(pred)
        
        return np.concatenate(predictions, axis=0)
    
    def benchmark_inference_time(self, x_sample, num_runs=100):
        """
        Benchmark inference time
        
        Args:
            x_sample: Sample input
            num_runs: Number of runs
            
        Returns:
            Timing statistics
        """
        import time
        
        # Warmup
        for _ in range(10):
            _ = self.predict(x_sample)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = self.predict(x_sample)
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
        
        stats = {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'median_ms': float(np.median(times))
        }
        
        print(f"Inference time: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
        
        return stats

