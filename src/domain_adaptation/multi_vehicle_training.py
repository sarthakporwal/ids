
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path
import json


class MultiVehicleDataset:
    
    def __init__(self):
        self.vehicles = {}
        self.vehicle_names = []
    
    def add_vehicle_data(self, vehicle_name, x_data, metadata=None):
        self.vehicles[vehicle_name] = {
            'data': x_data,
            'metadata': metadata or {},
            'num_samples': len(x_data)
        }
        self.vehicle_names.append(vehicle_name)
        print(f"Added vehicle '{vehicle_name}' with {len(x_data)} samples")
    
    def get_balanced_batch(self, batch_size):
        samples_per_vehicle = batch_size // len(self.vehicles)
        
        batch_data = []
        batch_labels = []
        
        for vid, vehicle_name in enumerate(self.vehicle_names):
            vehicle_data = self.vehicles[vehicle_name]['data']
            
            indices = np.random.choice(len(vehicle_data), samples_per_vehicle, replace=False)
            samples = vehicle_data[indices]
            
            batch_data.append(samples)
            batch_labels.extend([vid] * samples_per_vehicle)
        
        batch_data = np.concatenate(batch_data, axis=0)
        batch_labels = np.array(batch_labels)
        
        return batch_data, batch_labels
    
    def get_vehicle_batches(self, batch_size):
        batches = {}
        
        for vehicle_name in self.vehicle_names:
            vehicle_data = self.vehicles[vehicle_name]['data']
            indices = np.random.choice(len(vehicle_data), batch_size, replace=False)
            batches[vehicle_name] = vehicle_data[indices]
        
        return batches


class MultiVehicleTrainer:
    
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.num_vehicles = len(dataset.vehicles)
    
    def train_with_vehicle_mixing(self, epochs=100, batch_size=128, 
                                  validation_split=0.1):
        all_data = []
        for vehicle_name in self.dataset.vehicle_names:
            all_data.append(self.dataset.vehicles[vehicle_name]['data'])
        
        x_combined = np.concatenate(all_data, axis=0)
        
        indices = np.random.permutation(len(x_combined))
        x_combined = x_combined[indices]
        
        history = self.model.fit(
            x_combined, x_combined,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        return history
    
    def train_with_curriculum_learning(self, epochs=100, batch_size=128):
        history = {'loss': [], 'vehicle_losses': {v: [] for v in self.dataset.vehicle_names}}
        
        vehicle_difficulties = {}
        for vehicle_name in self.dataset.vehicle_names:
            data = self.dataset.vehicles[vehicle_name]['data']
            variance = np.var(data)
            vehicle_difficulties[vehicle_name] = variance
        
        sorted_vehicles = sorted(vehicle_difficulties.items(), key=lambda x: x[1])
        
        epochs_per_vehicle = epochs // self.num_vehicles
        
        for stage, (vehicle_name, difficulty) in enumerate(sorted_vehicles):
            print(f"\nStage {stage+1}: Training on {vehicle_name} (difficulty: {difficulty:.4f})")
            
            x_vehicle = self.dataset.vehicles[vehicle_name]['data']
            
            if stage > 0:
                prev_data = []
                for prev_vehicle, _ in sorted_vehicles[:stage]:
                    prev_data.append(self.dataset.vehicles[prev_vehicle]['data'])
                x_vehicle = np.concatenate([x_vehicle] + prev_data, axis=0)
            
            stage_history = self.model.fit(
                x_vehicle, x_vehicle,
                epochs=epochs_per_vehicle,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=1
            )
            
            history['loss'].extend(stage_history.history['loss'])
            history['vehicle_losses'][vehicle_name] = stage_history.history['loss']
        
        return history
    
    def train_with_alternating_vehicles(self, epochs=100, batch_size=128):
        history = {'loss': [], 'vehicle_losses': {v: [] for v in self.dataset.vehicle_names}}
        
        for epoch in range(epochs):
            vehicle_idx = epoch % self.num_vehicles
            vehicle_name = self.dataset.vehicle_names[vehicle_idx]
            
            x_vehicle = self.dataset.vehicles[vehicle_name]['data']
            
            indices = np.random.choice(len(x_vehicle), batch_size, replace=False)
            x_batch = x_vehicle[indices]
            
            loss = self.model.train_on_batch(x_batch, x_batch)
            
            history['loss'].append(float(loss))
            history['vehicle_losses'][vehicle_name].append(float(loss))
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Vehicle: {vehicle_name}, Loss: {loss:.4f}")
        
        return history
    
    def evaluate_per_vehicle(self):
        results = {}
        
        for vehicle_name in self.dataset.vehicle_names:
            x_vehicle = self.dataset.vehicles[vehicle_name]['data']
            
            predictions = self.model.predict(x_vehicle, verbose=0)
            
            mse = np.mean(np.square(predictions - x_vehicle))
            mae = np.mean(np.abs(predictions - x_vehicle))
            
            results[vehicle_name] = {
                'mse': float(mse),
                'mae': float(mae),
                'num_samples': len(x_vehicle)
            }
            
            print(f"{vehicle_name}: MSE={mse:.6f}, MAE={mae:.6f}")
        
        overall_mse = np.mean([v['mse'] for v in results.values()])
        overall_mae = np.mean([v['mae'] for v in results.values()])
        
        results['overall'] = {
            'mse': float(overall_mse),
            'mae': float(overall_mae)
        }
        
        return results


class VehicleAgnosticTraining:
    
    @staticmethod
    def train_with_mixup(model, dataset, epochs=100, batch_size=128, alpha=0.2):
        history = {'loss': []}
        
        for epoch in range(epochs):
            x_batch, vehicle_labels = dataset.get_balanced_batch(batch_size)
            
            batch_size_actual = len(x_batch)
            indices = np.random.permutation(batch_size_actual)
            x_batch_shuffled = x_batch[indices]
            
            lam = np.random.beta(alpha, alpha, size=(batch_size_actual, 1, 1, 1))
            
            x_mixed = lam * x_batch + (1 - lam) * x_batch_shuffled
            
            loss = model.train_on_batch(x_mixed, x_mixed)
            history['loss'].append(float(loss))
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")
        
        return history
    
    @staticmethod
    def train_with_vehicle_dropout(model, dataset, epochs=100, batch_size=128, 
                                   dropout_rate=0.2):
        history = {'loss': []}
        vehicle_names = dataset.vehicle_names.copy()
        
        for epoch in range(epochs):
            num_active = max(1, int(len(vehicle_names) * (1 - dropout_rate)))
            active_vehicles = np.random.choice(vehicle_names, num_active, replace=False)
            
            vehicle_data = []
            for vehicle_name in active_vehicles:
                data = dataset.vehicles[vehicle_name]['data']
                samples_per_vehicle = batch_size // num_active
                indices = np.random.choice(len(data), samples_per_vehicle, replace=False)
                vehicle_data.append(data[indices])
            
            x_batch = np.concatenate(vehicle_data, axis=0)
            
            loss = model.train_on_batch(x_batch, x_batch)
            history['loss'].append(float(loss))
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Active vehicles: {num_active}, Loss: {loss:.4f}")
        
        return history


def create_universal_model(time_step, num_signals, num_vehicles):
    from tensorflow.keras import Model, layers
    
    input_shape = (time_step, num_signals, 1)
    inputs = layers.Input(shape=input_shape)
    vehicle_id = layers.Input(shape=(num_vehicles,), name='vehicle_id')
    
    x = layers.ZeroPadding2D((2, 2))(inputs)
    x = layers.Conv2D(32, (5, 5), padding='same', name='shared_conv1')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    x = layers.Conv2D(16, (5, 5), padding='same', name='shared_conv2')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    vehicle_embedding = layers.Dense(64, activation='relu')(vehicle_id)
    gamma = layers.Dense(encoded.shape[-1])(vehicle_embedding)
    beta = layers.Dense(encoded.shape[-1])(vehicle_embedding)
    
    gamma = layers.Reshape((1, 1, encoded.shape[-1]))(gamma)
    beta = layers.Reshape((1, 1, encoded.shape[-1]))(beta)
    
    x = layers.Multiply()([encoded, gamma])
    x = layers.Add()([x, beta])
    
    x = layers.Conv2D(16, (5, 5), padding='same', name='shared_dec1')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2D(32, (5, 5), padding='same', name='shared_dec2')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    temp_shape = x.shape
    diff_h = temp_shape[1] - input_shape[0]
    diff_w = temp_shape[2] - input_shape[1]
    if diff_h > 0 or diff_w > 0:
        x = layers.Cropping2D(cropping=((diff_h//2, diff_h-diff_h//2), 
                                       (diff_w//2, diff_w-diff_w//2)))(x)
    
    model = Model(inputs=[inputs, vehicle_id], outputs=x, name='universal_model')
    return model


def save_multi_vehicle_checkpoint(model, dataset, save_dir, checkpoint_name='multi_vehicle'):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.save(save_dir / f"{checkpoint_name}_model.h5")
    
    metadata = {
        'num_vehicles': len(dataset.vehicles),
        'vehicle_names': dataset.vehicle_names,
        'vehicle_info': {
            name: {
                'num_samples': info['num_samples'],
                'metadata': info['metadata']
            }
            for name, info in dataset.vehicles.items()
        }
    }
    
    with open(save_dir / f"{checkpoint_name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Multi-vehicle checkpoint saved to {save_dir}")

