
import tensorflow as tf
import numpy as np
from typing import Tuple, Optional


class AdversarialAttacks:
    
    def __init__(self, model, loss_fn='mse'):
        self.model = model
        self.loss_fn = tf.keras.losses.MeanSquaredError() if loss_fn == 'mse' else tf.keras.losses.MeanAbsoluteError()
    
    def compute_gradient(self, x, y=None):
        if y is None:
            y = x
            
        with tf.GradientTape() as tape:
            tape.watch(x)
            predictions = self.model(x, training=False)
            loss = self.loss_fn(y, predictions)
        
        gradient = tape.gradient(loss, x)
        return gradient
    
    def fgsm_attack(self, x, epsilon=0.01, targeted=False):
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        
        gradient = self.compute_gradient(x_tensor)
        
        signed_grad = tf.sign(gradient)
        
        if targeted:
            x_adv = x_tensor - epsilon * signed_grad
        else:
            x_adv = x_tensor + epsilon * signed_grad
        
        x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
        
        return x_adv.numpy()
    
    def pgd_attack(self, x, epsilon=0.01, alpha=0.001, num_iter=40, random_start=True):
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        
        if random_start:
            x_adv = x_tensor + tf.random.uniform(x_tensor.shape, -epsilon, epsilon)
            x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
        else:
            x_adv = tf.identity(x_tensor)
        
        for i in range(num_iter):
            gradient = self.compute_gradient(x_adv)
            
            x_adv = x_adv + alpha * tf.sign(gradient)
            
            perturbation = tf.clip_by_value(x_adv - x_tensor, -epsilon, epsilon)
            x_adv = x_tensor + perturbation
            
            x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
        
        return x_adv.numpy()
    
    def carlini_wagner_attack(self, x, c=1.0, kappa=0, max_iter=100, learning_rate=0.01):
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        batch_size = x_tensor.shape[0]
        
        w = tf.Variable(tf.zeros_like(x_tensor), trainable=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        best_adv = x_tensor.numpy()
        best_l2 = np.inf * np.ones(batch_size)
        
        for iteration in range(max_iter):
            with tf.GradientTape() as tape:
                x_adv = 0.5 * (tf.tanh(w) + 1.0)
                
                predictions = self.model(x_adv, training=False)
                recon_loss = tf.reduce_sum(tf.square(predictions - x_adv), axis=[1, 2, 3])
                
                l2_dist = tf.reduce_sum(tf.square(x_adv - x_tensor), axis=[1, 2, 3])
                
                loss = tf.reduce_mean(l2_dist + c * recon_loss)
            
            gradients = tape.gradient(loss, [w])
            optimizer.apply_gradients(zip(gradients, [w]))
            
            x_adv_np = x_adv.numpy()
            l2_np = l2_dist.numpy()
            
            for i in range(batch_size):
                if l2_np[i] < best_l2[i]:
                    best_l2[i] = l2_np[i]
                    best_adv[i] = x_adv_np[i]
        
        return best_adv
    
    def automotive_masquerade_attack(self, x, target_signal_idx=None, epsilon=0.02):
        x_adv = x.copy()
        
        if target_signal_idx is None:
            num_signals = x.shape[2]
            num_targets = max(1, num_signals // 4)
            target_signal_idx = np.random.choice(num_signals, num_targets, replace=False)
        
        for idx in target_signal_idx:
            noise = np.random.uniform(-epsilon, epsilon, size=x[:, :, idx:idx+1, :].shape)
            x_adv[:, :, idx:idx+1, :] = np.clip(x[:, :, idx:idx+1, :] + noise, 0, 1)
        
        return x_adv
    
    def temporal_attack(self, x, delay_steps=5, epsilon=0.01):
        x_adv = x.copy()
        time_steps = x.shape[1]
        
        if delay_steps < time_steps:
            x_shifted = np.roll(x, shift=delay_steps, axis=1)
            
            alpha = 0.7
            x_adv = alpha * x + (1 - alpha) * x_shifted
            
            noise = np.random.uniform(-epsilon, epsilon, size=x.shape)
            x_adv = np.clip(x_adv + noise, 0, 1)
        
        return x_adv
    
    def evaluate_robustness(self, x_clean, x_adv):
        pred_clean = self.model.predict(x_clean, verbose=0)
        pred_adv = self.model.predict(x_adv, verbose=0)
        
        recon_error_clean = np.mean(np.square(pred_clean - x_clean))
        recon_error_adv = np.mean(np.square(pred_adv - x_adv))
        
        l2_perturbation = np.mean(np.sqrt(np.sum(np.square(x_adv - x_clean), axis=(1, 2, 3))))
        linf_perturbation = np.max(np.abs(x_adv - x_clean))
        
        degradation_ratio = recon_error_adv / (recon_error_clean + 1e-10)
        
        metrics = {
            'clean_reconstruction_error': float(recon_error_clean),
            'adversarial_reconstruction_error': float(recon_error_adv),
            'l2_perturbation': float(l2_perturbation),
            'linf_perturbation': float(linf_perturbation),
            'degradation_ratio': float(degradation_ratio),
            'avg_perturbation_per_signal': float(np.mean(np.abs(x_adv - x_clean)))
        }
        
        return metrics


def generate_mixed_adversarial_batch(x, model, attack_ratio=0.3, epsilon=0.01):
    batch_size = x.shape[0]
    num_adv = int(batch_size * attack_ratio)
    
    adv_indices = np.random.choice(batch_size, num_adv, replace=False)
    
    x_mixed = x.copy()
    labels = np.zeros(batch_size)
    
    if num_adv > 0:
        attacker = AdversarialAttacks(model)
        
        attack_type = np.random.choice(['fgsm', 'pgd', 'automotive'])
        
        if attack_type == 'fgsm':
            x_adv = attacker.fgsm_attack(x[adv_indices], epsilon=epsilon)
        elif attack_type == 'pgd':
            x_adv = attacker.pgd_attack(x[adv_indices], epsilon=epsilon, num_iter=20)
        else:
            x_adv = attacker.automotive_masquerade_attack(x[adv_indices], epsilon=epsilon)
        
        x_mixed[adv_indices] = x_adv
        labels[adv_indices] = 1
    
    return x_mixed, labels

