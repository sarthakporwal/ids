"""
Robustness Evaluation Metrics for CANShield
Comprehensive metrics for adversarial robustness assessment
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import pandas as pd
from typing import Dict, List, Tuple


class RobustnessEvaluator:
    """Comprehensive robustness evaluation for CAN-IDS"""
    
    def __init__(self, model):
        """
        Initialize robustness evaluator
        
        Args:
            model: Trained autoencoder model
        """
        self.model = model
        self.results = {}
    
    def compute_reconstruction_errors(self, x_data):
        """
        Compute reconstruction errors for input data
        
        Args:
            x_data: Input samples
            
        Returns:
            Array of reconstruction errors per sample
        """
        predictions = self.model.predict(x_data, verbose=0)
        errors = np.mean(np.square(predictions - x_data), axis=(1, 2, 3))
        return errors
    
    def evaluate_attack_success_rate(self, x_clean, x_adv, threshold):
        """
        Evaluate attack success rate (ASR)
        ASR = % of adversarial examples misclassified
        
        Args:
            x_clean: Clean samples
            x_adv: Adversarial samples
            threshold: Detection threshold
            
        Returns:
            Attack success rate
        """
        # Reconstruction errors
        errors_clean = self.compute_reconstruction_errors(x_clean)
        errors_adv = self.compute_reconstruction_errors(x_adv)
        
        # Clean samples should be below threshold
        clean_detected_correctly = np.sum(errors_clean < threshold)
        
        # Adversarial samples should be above threshold but aren't
        adv_not_detected = np.sum(errors_adv < threshold)
        
        asr = adv_not_detected / len(x_adv)
        
        return {
            'attack_success_rate': float(asr),
            'clean_accuracy': float(clean_detected_correctly / len(x_clean)),
            'avg_clean_error': float(np.mean(errors_clean)),
            'avg_adv_error': float(np.mean(errors_adv))
        }
    
    def evaluate_perturbation_sensitivity(self, x_data, epsilon_range=[0.001, 0.005, 0.01, 0.02, 0.05]):
        """
        Evaluate model sensitivity to different perturbation magnitudes
        
        Args:
            x_data: Input samples
            epsilon_range: Range of epsilon values to test
            
        Returns:
            Sensitivity results
        """
        from .attacks import AdversarialAttacks
        
        attacker = AdversarialAttacks(self.model)
        results = []
        
        errors_clean = self.compute_reconstruction_errors(x_data)
        
        for epsilon in epsilon_range:
            # Generate adversarial examples
            x_adv_fgsm = attacker.fgsm_attack(x_data, epsilon=epsilon)
            x_adv_pgd = attacker.pgd_attack(x_data, epsilon=epsilon, num_iter=20)
            
            # Compute errors
            errors_fgsm = self.compute_reconstruction_errors(x_adv_fgsm)
            errors_pgd = self.compute_reconstruction_errors(x_adv_pgd)
            
            # Compute metrics
            result = {
                'epsilon': epsilon,
                'fgsm_error_increase': float(np.mean(errors_fgsm) - np.mean(errors_clean)),
                'pgd_error_increase': float(np.mean(errors_pgd) - np.mean(errors_clean)),
                'fgsm_l2_perturbation': float(np.mean(np.sqrt(np.sum(np.square(x_adv_fgsm - x_data), axis=(1,2,3))))),
                'pgd_l2_perturbation': float(np.mean(np.sqrt(np.sum(np.square(x_adv_pgd - x_data), axis=(1,2,3))))),
                'fgsm_max_error': float(np.max(errors_fgsm)),
                'pgd_max_error': float(np.max(errors_pgd))
            }
            results.append(result)
        
        return results
    
    def evaluate_cross_attack_robustness(self, x_data, attack_types=['fgsm', 'pgd', 'automotive', 'temporal']):
        """
        Evaluate robustness across different attack types
        
        Args:
            x_data: Input samples
            attack_types: List of attack types to evaluate
            
        Returns:
            Cross-attack robustness metrics
        """
        from .attacks import AdversarialAttacks
        
        attacker = AdversarialAttacks(self.model)
        results = {}
        
        errors_clean = self.compute_reconstruction_errors(x_data)
        base_error = np.mean(errors_clean)
        
        for attack_type in attack_types:
            if attack_type == 'fgsm':
                x_adv = attacker.fgsm_attack(x_data, epsilon=0.01)
            elif attack_type == 'pgd':
                x_adv = attacker.pgd_attack(x_data, epsilon=0.01, num_iter=40)
            elif attack_type == 'automotive':
                x_adv = attacker.automotive_masquerade_attack(x_data, epsilon=0.02)
            elif attack_type == 'temporal':
                x_adv = attacker.temporal_attack(x_data, delay_steps=5, epsilon=0.01)
            else:
                continue
            
            errors_adv = self.compute_reconstruction_errors(x_adv)
            
            results[attack_type] = {
                'avg_error': float(np.mean(errors_adv)),
                'max_error': float(np.max(errors_adv)),
                'error_increase': float(np.mean(errors_adv) - base_error),
                'detection_rate': float(np.sum(errors_adv > base_error * 1.5) / len(errors_adv))
            }
        
        return results
    
    def compute_robustness_score(self, x_data, epsilon=0.01):
        """
        Compute overall robustness score (0-1, higher is better)
        
        Args:
            x_data: Input samples
            epsilon: Attack strength
            
        Returns:
            Robustness score and components
        """
        from .attacks import AdversarialAttacks
        
        attacker = AdversarialAttacks(self.model)
        
        # Generate attacks
        x_fgsm = attacker.fgsm_attack(x_data, epsilon=epsilon)
        x_pgd = attacker.pgd_attack(x_data, epsilon=epsilon, num_iter=20)
        
        # Compute errors
        errors_clean = self.compute_reconstruction_errors(x_data)
        errors_fgsm = self.compute_reconstruction_errors(x_fgsm)
        errors_pgd = self.compute_reconstruction_errors(x_pgd)
        
        # Component scores (normalized)
        # 1. Error stability (how much error changes under attack)
        error_stability = 1.0 / (1.0 + np.mean(errors_fgsm) / (np.mean(errors_clean) + 1e-10))
        
        # 2. Perturbation resistance (how much perturbation needed)
        l2_pert = np.mean(np.sqrt(np.sum(np.square(x_fgsm - x_data), axis=(1,2,3))))
        perturbation_resistance = np.clip(l2_pert * 10, 0, 1)  # Scale factor
        
        # 3. Attack diversity robustness
        pgd_stability = 1.0 / (1.0 + np.mean(errors_pgd) / (np.mean(errors_clean) + 1e-10))
        
        # Overall robustness score
        robustness_score = (error_stability + perturbation_resistance + pgd_stability) / 3.0
        
        return {
            'robustness_score': float(robustness_score),
            'error_stability': float(error_stability),
            'perturbation_resistance': float(perturbation_resistance),
            'attack_diversity_robustness': float(pgd_stability),
            'clean_error': float(np.mean(errors_clean)),
            'fgsm_error': float(np.mean(errors_fgsm)),
            'pgd_error': float(np.mean(errors_pgd))
        }
    
    def generate_robustness_report(self, x_data, save_path=None):
        """
        Generate comprehensive robustness report
        
        Args:
            x_data: Test data
            save_path: Path to save report
            
        Returns:
            Complete robustness report dictionary
        """
        print("Generating comprehensive robustness report...")
        
        report = {
            'overall_score': self.compute_robustness_score(x_data),
            'perturbation_sensitivity': self.evaluate_perturbation_sensitivity(x_data),
            'cross_attack_robustness': self.evaluate_cross_attack_robustness(x_data),
        }
        
        # Save if path provided
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {save_path}")
        
        return report
    
    def compare_models_robustness(self, models_dict, x_data, epsilon=0.01):
        """
        Compare robustness of multiple models
        
        Args:
            models_dict: Dictionary of model_name: model
            x_data: Test data
            epsilon: Attack strength
            
        Returns:
            Comparison dataframe
        """
        from .attacks import AdversarialAttacks
        
        results = []
        
        for model_name, model in models_dict.items():
            print(f"Evaluating {model_name}...")
            
            evaluator = RobustnessEvaluator(model)
            score = evaluator.compute_robustness_score(x_data, epsilon=epsilon)
            
            result = {
                'model': model_name,
                **score
            }
            results.append(result)
        
        df = pd.DataFrame(results)
        return df


def compute_certified_robustness(model, x_data, epsilon, num_samples=100):
    """
    Compute certified robustness using randomized smoothing
    
    Args:
        model: Model to evaluate
        x_data: Input samples
        epsilon: Perturbation bound
        num_samples: Number of noise samples
        
    Returns:
        Certified robustness metrics
    """
    certified_correct = 0
    
    for i in range(len(x_data)):
        x_sample = x_data[i:i+1]
        
        # Add random noise multiple times
        predictions = []
        for _ in range(num_samples):
            noise = np.random.normal(0, epsilon, x_sample.shape)
            x_noisy = np.clip(x_sample + noise, 0, 1)
            pred = model.predict(x_noisy, verbose=0)
            predictions.append(pred)
        
        # Check consistency
        predictions = np.array(predictions)
        variance = np.var(predictions, axis=0)
        
        # If variance is low, prediction is certified
        if np.mean(variance) < epsilon:
            certified_correct += 1
    
    certified_accuracy = certified_correct / len(x_data)
    
    return {
        'certified_accuracy': float(certified_accuracy),
        'epsilon': epsilon,
        'num_samples': num_samples
    }

