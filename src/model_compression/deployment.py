
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import time


class EdgeDeployment:
    
    def __init__(self, model_path, model_type='tflite'):
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.model = None
        self.interpreter = None
        
        self._load_model()
    
    def _load_model(self):
        if self.model_type == 'tflite':
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        elif self.model_type == 'h5':
            self.model = tf.keras.models.load_model(str(self.model_path))
        else:
            self.model = tf.saved_model.load(str(self.model_path))
    
    def predict(self, x_input):
        if self.model_type == 'tflite':
            input_data = x_input.astype(self.input_details[0]['dtype'])
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            return output
        else:
            return self.model.predict(x_input, verbose=0)
    
    def benchmark(self, sample_input, num_iterations=100, warmup=10):
        print(f"Benchmarking {self.model_type} model...")
        
        for _ in range(warmup):
            _ = self.predict(sample_input)
        
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = self.predict(sample_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        results = {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'median_ms': float(np.median(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'throughput_samples_per_sec': float(1000 / np.mean(times))
        }
        
        print(f"\nBenchmark Results:")
        print(f"  Mean inference time: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms")
        print(f"  Median: {results['median_ms']:.2f} ms")
        print(f"  P95: {results['p95_ms']:.2f} ms")
        print(f"  P99: {results['p99_ms']:.2f} ms")
        print(f"  Throughput: {results['throughput_samples_per_sec']:.2f} samples/sec")
        
        return results
    
    def get_model_info(self):
        info = {
            'model_path': str(self.model_path),
            'model_type': self.model_type,
            'model_size_bytes': self.model_path.stat().st_size,
            'model_size_mb': self.model_path.stat().st_size / (1024 * 1024)
        }
        
        if self.model_type == 'tflite':
            info['input_shape'] = self.input_details[0]['shape'].tolist()
            info['input_dtype'] = str(self.input_details[0]['dtype'])
            info['output_shape'] = self.output_details[0]['shape'].tolist()
            info['output_dtype'] = str(self.output_details[0]['dtype'])
        
        return info


class RealtimeCANMonitor:
    
    def __init__(self, model_deployment, threshold, window_size=50, num_signals=20):
        self.deployment = model_deployment
        self.threshold = threshold
        self.window_size = window_size
        self.num_signals = num_signals
        
        self.buffer = np.zeros((window_size, num_signals))
        self.buffer_idx = 0
        self.buffer_full = False
        
        self.stats = {
            'total_packets': 0,
            'anomalies_detected': 0,
            'avg_inference_time_ms': 0,
            'inference_times': []
        }
    
    def process_packet(self, can_signals):
        self.buffer[self.buffer_idx] = can_signals
        self.buffer_idx = (self.buffer_idx + 1) % self.window_size
        
        if self.buffer_idx == 0:
            self.buffer_full = True
        
        self.stats['total_packets'] += 1
        
        if not self.buffer_full:
            return {'status': 'buffering', 'anomaly': False}
        
        window = np.roll(self.buffer, -self.buffer_idx, axis=0)
        x_input = window.reshape(1, self.window_size, self.num_signals, 1).astype(np.float32)
        
        start = time.perf_counter()
        prediction = self.deployment.predict(x_input)
        inference_time = (time.perf_counter() - start) * 1000
        
        reconstruction_error = np.mean(np.square(prediction - x_input))
        
        is_anomaly = reconstruction_error > self.threshold
        
        if is_anomaly:
            self.stats['anomalies_detected'] += 1
        
        self.stats['inference_times'].append(inference_time)
        if len(self.stats['inference_times']) > 1000:
            self.stats['inference_times'].pop(0)
        self.stats['avg_inference_time_ms'] = np.mean(self.stats['inference_times'])
        
        return {
            'status': 'active',
            'anomaly': bool(is_anomaly),
            'reconstruction_error': float(reconstruction_error),
            'threshold': self.threshold,
            'inference_time_ms': inference_time
        }
    
    def get_statistics(self):
        detection_rate = (self.stats['anomalies_detected'] / max(1, self.stats['total_packets'])) * 100
        
        return {
            **self.stats,
            'detection_rate_%': detection_rate,
            'avg_inference_time_ms': self.stats['avg_inference_time_ms']
        }
    
    def reset_statistics(self):
        self.stats = {
            'total_packets': 0,
            'anomalies_detected': 0,
            'avg_inference_time_ms': 0,
            'inference_times': []
        }


def create_deployment_package(model_path, config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating deployment package in {output_dir}...")
    
    model = tf.keras.models.load_model(model_path)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    
    with open(output_dir / 'model_float16.tflite', 'wb') as f:
        f.write(tflite_model)
    
    tf.saved_model.save(model, str(output_dir / 'saved_model'))
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    readme_content = f"""

- Model: {Path(model_path).name}
- Input Shape: {model.input_shape}
- Output Shape: {model.output_shape}

- `model_float16.tflite`: TensorFlow Lite model (Float16)
- `saved_model/`: TensorFlow SavedModel format
- `config.json`: Model configuration
- `README.md`: This file


```python
from deployment import EdgeDeployment

deployment = EdgeDeployment('model_float16.tflite', model_type='tflite')

prediction = deployment.predict(input_data)
```

```python
from deployment import RealtimeCANMonitor

monitor = RealtimeCANMonitor(deployment, threshold=0.01)

result = monitor.process_packet(can_signals)
if result['anomaly']:
    print("Attack detected!")
```

- Inference time: < 10ms (typical)
- Model size: {Path(output_dir / 'model_float16.tflite').stat().st_size / 1024:.2f} KB
- Suitable for embedded deployment

- TensorFlow Lite Runtime
- NumPy
- Python 3.7+
    Benchmark different deployment options
    
    Args:
        model_path: Path to model
        x_test: Test data
        
    Returns:
        Benchmark results