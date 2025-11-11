
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback
from .attacks import AdversarialAttacks, generate_mixed_adversarial_batch
import json
from pathlib import Path


class AdversarialTrainer:
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.attacker = AdversarialAttacks(model)
        
        self.adv_ratio = config.get('adversarial_ratio', 0.3)
        self.epsilon_schedule = config.get('epsilon_schedule', [0.005, 0.01, 0.02])
        self.current_epsilon = self.epsilon_schedule[0]
    
    def train_with_adversarial_examples(self, x_train, epochs=100, batch_size=128,
                                       validation_split=0.1, callbacks=None):
        val_size = int(len(x_train) * validation_split)
        x_val = x_train[-val_size:]
        x_train = x_train[:-val_size]
        
        history = {
            'loss': [],
            'val_loss': [],
            'adv_loss': [],
            'clean_loss': []
        }
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            self._update_epsilon(epoch, epochs)
            
            indices = np.random.permutation(len(x_train))
            x_train_shuffled = x_train[indices]
            
            epoch_losses = []
            epoch_adv_losses = []
            epoch_clean_losses = []
            
            num_batches = len(x_train) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                x_batch = x_train_shuffled[start_idx:end_idx]
                
                x_adv_batch = self.attacker.fgsm_attack(x_batch, epsilon=self.current_epsilon)
                
                x_mixed = np.concatenate([x_batch, x_adv_batch], axis=0)
                
                loss = self.model.train_on_batch(x_mixed, x_mixed)
                
                clean_loss = self.model.test_on_batch(x_batch, x_batch)
                adv_loss = self.model.test_on_batch(x_adv_batch, x_adv_batch)
                
                loss_value = loss[0] if isinstance(loss, list) else loss
                clean_loss_value = clean_loss[0] if isinstance(clean_loss, list) else clean_loss
                adv_loss_value = adv_loss[0] if isinstance(adv_loss, list) else adv_loss
                
                epoch_losses.append(loss_value)
                epoch_clean_losses.append(clean_loss_value)
                epoch_adv_losses.append(adv_loss_value)
                
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}/{num_batches} - "
                          f"Loss: {loss_value:.4f}, Clean: {clean_loss_value:.4f}, Adv: {adv_loss_value:.4f}")
            
            val_loss = self.model.evaluate(x_val, x_val, verbose=0)
            val_loss_value = val_loss[0] if isinstance(val_loss, list) else val_loss
            
            history['loss'].append(float(np.mean(epoch_losses)))
            history['val_loss'].append(float(val_loss_value))
            history['adv_loss'].append(float(np.mean(epoch_adv_losses)))
            history['clean_loss'].append(float(np.mean(epoch_clean_losses)))
            
            print(f"Epoch Loss: {history['loss'][-1]:.4f}, "
                  f"Val Loss: {history['val_loss'][-1]:.4f}, "
                  f"Adv Loss: {history['adv_loss'][-1]:.4f}")
            
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, history)
        
        return history
    
    def train_with_multiple_attacks(self, x_train, epochs=100, batch_size=128,
                                   validation_split=0.1, callbacks=None):
        val_size = int(len(x_train) * validation_split)
        x_val = x_train[-val_size:]
        x_train = x_train[:-val_size]
        
        history = {'loss': [], 'val_loss': [], 'fgsm_loss': [], 'pgd_loss': [], 'auto_loss': []}
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            self._update_epsilon(epoch, epochs)
            
            indices = np.random.permutation(len(x_train))
            x_train_shuffled = x_train[indices]
            
            epoch_losses = {'total': [], 'fgsm': [], 'pgd': [], 'auto': []}
            
            num_batches = len(x_train) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                x_batch = x_train_shuffled[start_idx:end_idx]
                
                batch_size_per_attack = len(x_batch) // 4
                
                x_clean = x_batch[:batch_size_per_attack]
                x_fgsm = self.attacker.fgsm_attack(x_batch[batch_size_per_attack:2*batch_size_per_attack], 
                                                   epsilon=self.current_epsilon)
                x_pgd = self.attacker.pgd_attack(x_batch[2*batch_size_per_attack:3*batch_size_per_attack],
                                                epsilon=self.current_epsilon, num_iter=10)
                x_auto = self.attacker.automotive_masquerade_attack(x_batch[3*batch_size_per_attack:],
                                                                    epsilon=self.current_epsilon)
                
                x_mixed = np.concatenate([x_clean, x_fgsm, x_pgd, x_auto], axis=0)
                
                loss = self.model.train_on_batch(x_mixed, x_mixed)
                
                loss_value = loss[0] if isinstance(loss, list) else loss
                epoch_losses['total'].append(loss_value)
                
                fgsm_loss = self.model.test_on_batch(x_fgsm, x_fgsm)
                pgd_loss = self.model.test_on_batch(x_pgd, x_pgd)
                auto_loss = self.model.test_on_batch(x_auto, x_auto)
                
                epoch_losses['fgsm'].append(fgsm_loss[0] if isinstance(fgsm_loss, list) else fgsm_loss)
                epoch_losses['pgd'].append(pgd_loss[0] if isinstance(pgd_loss, list) else pgd_loss)
                epoch_losses['auto'].append(auto_loss[0] if isinstance(auto_loss, list) else auto_loss)
                
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}/{num_batches} - Loss: {loss_value:.4f}")
            
            val_loss = self.model.evaluate(x_val, x_val, verbose=0)
            val_loss_value = val_loss[0] if isinstance(val_loss, list) else val_loss
            
            history['loss'].append(float(np.mean(epoch_losses['total'])))
            history['val_loss'].append(float(val_loss_value))
            history['fgsm_loss'].append(float(np.mean(epoch_losses['fgsm'])))
            history['pgd_loss'].append(float(np.mean(epoch_losses['pgd'])))
            history['auto_loss'].append(float(np.mean(epoch_losses['auto'])))
            
            print(f"Epoch Loss: {history['loss'][-1]:.4f}, Val Loss: {history['val_loss'][-1]:.4f}")
        
        return history
    
    def _update_epsilon(self, current_epoch, total_epochs):
        if current_epoch < total_epochs * 0.3:
            self.current_epsilon = self.epsilon_schedule[0]
        elif current_epoch < total_epochs * 0.7:
            self.current_epsilon = self.epsilon_schedule[1]
        else:
            self.current_epsilon = self.epsilon_schedule[2] if len(self.epsilon_schedule) > 2 else self.epsilon_schedule[1]


class RobustnessCallback(Callback):
    
    def __init__(self, x_val, save_dir, epsilon=0.01):
        super().__init__()
        self.x_val = x_val
        self.save_dir = Path(save_dir)
        self.epsilon = epsilon
        self.robustness_history = []
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        attacker = AdversarialAttacks(self.model)
        
        sample_size = min(500, len(self.x_val))
        x_sample = self.x_val[:sample_size]
        
        x_adv = attacker.fgsm_attack(x_sample, epsilon=self.epsilon)
        
        metrics = attacker.evaluate_robustness(x_sample, x_adv)
        metrics['epoch'] = epoch
        
        self.robustness_history.append(metrics)
        
        print(f"\nRobustness - L2 Pert: {metrics['l2_perturbation']:.4f}, "
              f"Degradation: {metrics['degradation_ratio']:.4f}")
        
        with open(self.save_dir / 'robustness_history.json', 'w') as f:
            json.dump(self.robustness_history, f, indent=2)


def train_robust_autoencoder(model, x_train, args, file_index, use_adversarial=True):
    root_dir = args.root_dir
    dataset_name = args.dataset_name
    time_step = args.time_step
    num_signals = args.num_signals
    sampling_period = args.sampling_period
    max_epoch = args.max_epoch
    
    checkpoint_path = f"{root_dir}/../artifacts/model_ckpts/{dataset_name}/robust_autoencoder_{dataset_name}_{time_step}_{num_signals}_{sampling_period}.h5"
    history_dir = Path(f"{root_dir}/../artifacts/histories/{dataset_name}/")
    history_dir.mkdir(parents=True, exist_ok=True)
    
    keras_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, mode='min'),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', 
                                          save_best_only=True, mode='auto', verbose=1),
        RobustnessCallback(x_train[-500:], history_dir, epsilon=0.01)
    ]
    
    if use_adversarial:
        adv_config = {
            'adversarial_ratio': 0.3,
            'epsilon_schedule': [0.005, 0.01, 0.02]
        }
        
        trainer = AdversarialTrainer(model, adv_config)
        
        history = trainer.train_with_multiple_attacks(
            x_train,
            epochs=max_epoch,
            batch_size=128,
            validation_split=0.1,
            callbacks=keras_callbacks
        )
    else:
        history = model.fit(
            x_train,
            x_train,
            epochs=max_epoch,
            batch_size=128,
            validation_split=0.1,
            callbacks=keras_callbacks,
            verbose=1
        )
        history = history.history
    
    history_file = history_dir / f"robust_history_{dataset_name}_{time_step}_{num_signals}_{sampling_period}_{file_index}.json"
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history

