"""
Ensemble-based Uncertainty Quantification
Uses multiple models for uncertainty estimation
"""

import numpy as np
from scipy.stats import entropy


class EnsembleUncertainty:
    """Uncertainty estimation using ensemble of models"""
    
    def __init__(self, models, model_names=None):
        """
        Initialize ensemble uncertainty estimator
        
        Args:
            models: List of trained models
            model_names: Optional names for models
        """
        self.models = models
        self.num_models = len(models)
        self.model_names = model_names or [f"model_{i}" for i in range(self.num_models)]
    
    def predict_ensemble(self, x_input, return_individual=False):
        """
        Get ensemble predictions
        
        Args:
            x_input: Input data
            return_individual: Whether to return individual model predictions
            
        Returns:
            Ensemble prediction and uncertainty
        """
        predictions = []
        
        for model in self.models:
            pred = model.predict(x_input, verbose=0)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Ensemble statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Variance-based uncertainty
        variance_uncertainty = np.mean(np.var(predictions, axis=0))
        
        results = {
            'mean': mean_pred,
            'std': std_pred,
            'variance_uncertainty': float(variance_uncertainty),
            'predictions': predictions if return_individual else None
        }
        
        return results
    
    def model_disagreement(self, x_input):
        """
        Measure disagreement between models
        
        Args:
            x_input: Input data
            
        Returns:
            Disagreement metrics
        """
        predictions = []
        reconstruction_errors = []
        
        for model in self.models:
            pred = model.predict(x_input, verbose=0)
            predictions.append(pred)
            
            # Reconstruction error per model
            error = np.mean(np.square(pred - x_input), axis=(1, 2, 3))
            reconstruction_errors.append(error)
        
        predictions = np.array(predictions)
        reconstruction_errors = np.array(reconstruction_errors)
        
        # Coefficient of variation (std/mean) for each sample
        mean_errors = np.mean(reconstruction_errors, axis=0)
        std_errors = np.std(reconstruction_errors, axis=0)
        cv = std_errors / (mean_errors + 1e-10)
        
        # Pairwise disagreement
        pairwise_disagreements = []
        for i in range(self.num_models):
            for j in range(i+1, self.num_models):
                disagreement = np.mean(np.square(predictions[i] - predictions[j]))
                pairwise_disagreements.append(disagreement)
        
        results = {
            'mean_disagreement': float(np.mean(pairwise_disagreements)),
            'max_disagreement': float(np.max(pairwise_disagreements)),
            'coefficient_of_variation': cv,
            'mean_cv': float(np.mean(cv)),
            'reconstruction_errors': reconstruction_errors,
            'mean_reconstruction_error': float(np.mean(mean_errors))
        }
        
        return results
    
    def entropy_based_uncertainty(self, x_input, threshold):
        """
        Compute entropy-based uncertainty from ensemble
        
        Args:
            x_input: Input data
            threshold: Detection threshold
            
        Returns:
            Entropy-based uncertainty
        """
        reconstruction_errors = []
        
        for model in self.models:
            pred = model.predict(x_input, verbose=0)
            error = np.mean(np.square(pred - x_input), axis=(1, 2, 3))
            reconstruction_errors.append(error)
        
        reconstruction_errors = np.array(reconstruction_errors)
        
        # Convert to binary decisions
        decisions = reconstruction_errors > threshold  # Shape: (num_models, num_samples)
        
        # Compute entropy for each sample
        entropies = []
        for sample_idx in range(decisions.shape[1]):
            sample_decisions = decisions[:, sample_idx]
            
            # Probability of attack/normal
            p_attack = np.mean(sample_decisions)
            p_normal = 1 - p_attack
            
            # Entropy
            if p_attack == 0 or p_attack == 1:
                ent = 0
            else:
                ent = -p_attack * np.log2(p_attack + 1e-10) - p_normal * np.log2(p_normal + 1e-10)
            
            entropies.append(ent)
        
        entropies = np.array(entropies)
        
        # High entropy = high uncertainty (models disagree)
        # Low entropy = low uncertainty (models agree)
        
        results = {
            'entropies': entropies,
            'mean_entropy': float(np.mean(entropies)),
            'high_uncertainty_samples': int(np.sum(entropies > 0.5)),
            'agreement_rate': float(np.mean(np.all(decisions == decisions[0], axis=0)))
        }
        
        return results
    
    def weighted_ensemble_prediction(self, x_input, weights=None):
        """
        Weighted ensemble prediction
        
        Args:
            x_input: Input data
            weights: Optional model weights (default: equal)
            
        Returns:
            Weighted prediction
        """
        if weights is None:
            weights = np.ones(self.num_models) / self.num_models
        else:
            weights = np.array(weights)
            weights = weights / np.sum(weights)
        
        predictions = []
        for model in self.models:
            pred = model.predict(x_input, verbose=0)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Weighted average
        weighted_pred = np.sum(predictions * weights[:, None, None, None, None], axis=0)
        
        return weighted_pred
    
    def compute_model_reliability(self, x_val, y_val, threshold):
        """
        Compute reliability scores for each model
        
        Args:
            x_val: Validation data
            y_val: True labels
            threshold: Detection threshold
            
        Returns:
            Reliability scores
        """
        reliabilities = []
        
        for i, model in enumerate(self.models):
            pred = model.predict(x_val, verbose=0)
            errors = np.mean(np.square(pred - x_val), axis=(1, 2, 3))
            
            # Predictions
            y_pred = (errors > threshold).astype(int)
            
            # Accuracy
            accuracy = np.mean(y_pred == y_val)
            
            # F1 score
            tp = np.sum((y_pred == 1) & (y_val == 1))
            fp = np.sum((y_pred == 1) & (y_val == 0))
            fn = np.sum((y_pred == 0) & (y_val == 1))
            
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            reliabilities.append({
                'model_name': self.model_names[i],
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            })
        
        return reliabilities
    
    def selective_prediction(self, x_input, threshold, min_agreement=0.7):
        """
        Make predictions only when models agree sufficiently
        
        Args:
            x_input: Input data
            threshold: Detection threshold
            min_agreement: Minimum agreement ratio required
            
        Returns:
            Predictions with abstention
        """
        reconstruction_errors = []
        
        for model in self.models:
            pred = model.predict(x_input, verbose=0)
            error = np.mean(np.square(pred - x_input), axis=(1, 2, 3))
            reconstruction_errors.append(error)
        
        reconstruction_errors = np.array(reconstruction_errors)
        
        # Binary decisions
        decisions = reconstruction_errors > threshold
        
        # Agreement for each sample
        agreements = np.mean(decisions, axis=0)
        
        # Final decisions (only when agreement meets threshold)
        final_decisions = np.zeros(len(x_input), dtype=int)
        abstain = np.zeros(len(x_input), dtype=bool)
        
        # Attack if majority agrees on attack AND agreement is high
        attack_votes = agreements >= 0.5
        high_agreement = (agreements >= min_agreement) | (agreements <= (1 - min_agreement))
        
        final_decisions[attack_votes & high_agreement] = 1
        abstain[~high_agreement] = True
        
        results = {
            'predictions': final_decisions,
            'abstain': abstain,
            'agreements': agreements,
            'coverage': float(np.mean(~abstain)),
            'num_abstained': int(np.sum(abstain))
        }
        
        return results


class AdaptiveEnsemble:
    """Adaptive ensemble that weights models based on performance"""
    
    def __init__(self, models, model_names=None):
        """
        Initialize adaptive ensemble
        
        Args:
            models: List of models
            model_names: Optional model names
        """
        self.models = models
        self.num_models = len(models)
        self.model_names = model_names or [f"model_{i}" for i in range(self.num_models)]
        self.weights = np.ones(self.num_models) / self.num_models
    
    def update_weights(self, x_recent, window_size=100):
        """
        Update model weights based on recent performance
        
        Args:
            x_recent: Recent data samples
            window_size: Size of window for weight computation
            
        Returns:
            Updated weights
        """
        x_window = x_recent[-window_size:]
        
        # Compute reconstruction errors for each model
        errors = []
        for model in self.models:
            pred = model.predict(x_window, verbose=0)
            error = np.mean(np.square(pred - x_window))
            errors.append(error)
        
        errors = np.array(errors)
        
        # Inverse error as weight (lower error = higher weight)
        weights = 1.0 / (errors + 1e-10)
        weights = weights / np.sum(weights)
        
        self.weights = weights
        
        return weights
    
    def predict(self, x_input):
        """
        Adaptive weighted prediction
        
        Args:
            x_input: Input data
            
        Returns:
            Weighted ensemble prediction
        """
        predictions = []
        
        for model in self.models:
            pred = model.predict(x_input, verbose=0)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Weighted combination
        weighted_pred = np.sum(predictions * self.weights[:, None, None, None, None], axis=0)
        
        return weighted_pred


def create_diversity_ensemble(base_model, x_train, num_models=5, diversity_method='bootstrap'):
    """
    Create diverse ensemble of models
    
    Args:
        base_model: Base model architecture
        x_train: Training data
        num_models: Number of models in ensemble
        diversity_method: Method to ensure diversity ('bootstrap', 'dropout', 'initialization')
        
    Returns:
        List of trained diverse models
    """
    import tensorflow as tf
    
    models = []
    
    for i in range(num_models):
        print(f"\nTraining ensemble model {i+1}/{num_models}...")
        
        # Clone model
        model = tf.keras.models.clone_model(base_model)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        if diversity_method == 'bootstrap':
            # Bootstrap sampling
            indices = np.random.choice(len(x_train), len(x_train), replace=True)
            x_train_bootstrap = x_train[indices]
            training_data = x_train_bootstrap
        elif diversity_method == 'dropout':
            # Use full data but with different dropout
            training_data = x_train
        else:  # random initialization
            training_data = x_train
        
        # Train
        model.fit(
            training_data, training_data,
            epochs=50,
            batch_size=128,
            validation_split=0.1,
            verbose=0
        )
        
        models.append(model)
    
    return models

