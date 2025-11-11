
import numpy as np
from scipy.stats import entropy


class EnsembleUncertainty:
    
    def __init__(self, models, model_names=None):
        self.models = models
        self.num_models = len(models)
        self.model_names = model_names or [f"model_{i}" for i in range(self.num_models)]
    
    def predict_ensemble(self, x_input, return_individual=False):
        predictions = []
        
        for model in self.models:
            pred = model.predict(x_input, verbose=0)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        variance_uncertainty = np.mean(np.var(predictions, axis=0))
        
        results = {
            'mean': mean_pred,
            'std': std_pred,
            'variance_uncertainty': float(variance_uncertainty),
            'predictions': predictions if return_individual else None
        }
        
        return results
    
    def model_disagreement(self, x_input):
        predictions = []
        reconstruction_errors = []
        
        for model in self.models:
            pred = model.predict(x_input, verbose=0)
            predictions.append(pred)
            
            error = np.mean(np.square(pred - x_input), axis=(1, 2, 3))
            reconstruction_errors.append(error)
        
        predictions = np.array(predictions)
        reconstruction_errors = np.array(reconstruction_errors)
        
        mean_errors = np.mean(reconstruction_errors, axis=0)
        std_errors = np.std(reconstruction_errors, axis=0)
        cv = std_errors / (mean_errors + 1e-10)
        
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
        reconstruction_errors = []
        
        for model in self.models:
            pred = model.predict(x_input, verbose=0)
            error = np.mean(np.square(pred - x_input), axis=(1, 2, 3))
            reconstruction_errors.append(error)
        
        reconstruction_errors = np.array(reconstruction_errors)
        
        decisions = reconstruction_errors > threshold
        
        entropies = []
        for sample_idx in range(decisions.shape[1]):
            sample_decisions = decisions[:, sample_idx]
            
            p_attack = np.mean(sample_decisions)
            p_normal = 1 - p_attack
            
            if p_attack == 0 or p_attack == 1:
                ent = 0
            else:
                ent = -p_attack * np.log2(p_attack + 1e-10) - p_normal * np.log2(p_normal + 1e-10)
            
            entropies.append(ent)
        
        entropies = np.array(entropies)
        
        
        results = {
            'entropies': entropies,
            'mean_entropy': float(np.mean(entropies)),
            'high_uncertainty_samples': int(np.sum(entropies > 0.5)),
            'agreement_rate': float(np.mean(np.all(decisions == decisions[0], axis=0)))
        }
        
        return results
    
    def weighted_ensemble_prediction(self, x_input, weights=None):
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
        
        weighted_pred = np.sum(predictions * weights[:, None, None, None, None], axis=0)
        
        return weighted_pred
    
    def compute_model_reliability(self, x_val, y_val, threshold):
        reliabilities = []
        
        for i, model in enumerate(self.models):
            pred = model.predict(x_val, verbose=0)
            errors = np.mean(np.square(pred - x_val), axis=(1, 2, 3))
            
            y_pred = (errors > threshold).astype(int)
            
            accuracy = np.mean(y_pred == y_val)
            
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
        reconstruction_errors = []
        
        for model in self.models:
            pred = model.predict(x_input, verbose=0)
            error = np.mean(np.square(pred - x_input), axis=(1, 2, 3))
            reconstruction_errors.append(error)
        
        reconstruction_errors = np.array(reconstruction_errors)
        
        decisions = reconstruction_errors > threshold
        
        agreements = np.mean(decisions, axis=0)
        
        final_decisions = np.zeros(len(x_input), dtype=int)
        abstain = np.zeros(len(x_input), dtype=bool)
        
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
    
    def __init__(self, models, model_names=None):
        self.models = models
        self.num_models = len(models)
        self.model_names = model_names or [f"model_{i}" for i in range(self.num_models)]
        self.weights = np.ones(self.num_models) / self.num_models
    
    def update_weights(self, x_recent, window_size=100):
        x_window = x_recent[-window_size:]
        
        errors = []
        for model in self.models:
            pred = model.predict(x_window, verbose=0)
            error = np.mean(np.square(pred - x_window))
            errors.append(error)
        
        errors = np.array(errors)
        
        weights = 1.0 / (errors + 1e-10)
        weights = weights / np.sum(weights)
        
        self.weights = weights
        
        return weights
    
    def predict(self, x_input):
        predictions = []
        
        for model in self.models:
            pred = model.predict(x_input, verbose=0)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        weighted_pred = np.sum(predictions * self.weights[:, None, None, None, None], axis=0)
        
        return weighted_pred


def create_diversity_ensemble(base_model, x_train, num_models=5, diversity_method='bootstrap'):
    import tensorflow as tf
    
    models = []
    
    for i in range(num_models):
        print(f"\nTraining ensemble model {i+1}/{num_models}...")
        
        model = tf.keras.models.clone_model(base_model)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        if diversity_method == 'bootstrap':
            indices = np.random.choice(len(x_train), len(x_train), replace=True)
            x_train_bootstrap = x_train[indices]
            training_data = x_train_bootstrap
        elif diversity_method == 'dropout':
            training_data = x_train
        else:
            training_data = x_train
        
        model.fit(
            training_data, training_data,
            epochs=50,
            batch_size=128,
            validation_split=0.1,
            verbose=0
        )
        
        models.append(model)
    
    return models

