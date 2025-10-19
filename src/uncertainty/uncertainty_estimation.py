"""
Uncertainty Estimation Methods
Implements various uncertainty quantification techniques
"""

import tensorflow as tf
import numpy as np
from scipy.stats import entropy


class UncertaintyEstimator:
    """Estimate uncertainty in model predictions"""
    
    def __init__(self, model, method='monte_carlo'):
        """
        Initialize uncertainty estimator
        
        Args:
            model: Trained model
            method: Uncertainty estimation method ('monte_carlo', 'ensemble', 'dropout')
        """
        self.model = model
        self.method = method
    
    def monte_carlo_dropout(self, x_input, num_samples=30, dropout_rate=0.1):
        """
        Monte Carlo Dropout for uncertainty estimation
        
        Args:
            x_input: Input data
            num_samples: Number of stochastic forward passes
            dropout_rate: Dropout rate to apply
            
        Returns:
            Mean prediction, uncertainty (std), and all predictions
        """
        # Enable dropout during inference by setting training=True
        predictions = []
        
        for _ in range(num_samples):
            pred = self.model(x_input, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        # Compute statistics
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)
        
        # Uncertainty metrics
        epistemic_uncertainty = np.mean(std_prediction)  # Model uncertainty
        
        return {
            'mean': mean_prediction,
            'std': std_prediction,
            'epistemic_uncertainty': float(epistemic_uncertainty),
            'all_predictions': predictions
        }
    
    def bootstrap_uncertainty(self, x_input, num_bootstrap=30):
        """
        Bootstrap sampling for uncertainty estimation
        
        Args:
            x_input: Input data
            num_bootstrap: Number of bootstrap samples
            
        Returns:
            Uncertainty metrics
        """
        n_samples = len(x_input)
        predictions = []
        
        for _ in range(num_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            x_bootstrap = x_input[indices]
            
            # Predict
            pred = self.model.predict(x_bootstrap, verbose=0)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'uncertainty': float(np.mean(std_pred))
        }
    
    def reconstruction_confidence(self, x_input, threshold):
        """
        Compute confidence based on reconstruction error
        
        Args:
            x_input: Input data
            threshold: Detection threshold
            
        Returns:
            Confidence scores and classification
        """
        # Get predictions
        predictions = self.model.predict(x_input, verbose=0)
        
        # Reconstruction errors
        reconstruction_errors = np.mean(np.square(predictions - x_input), axis=(1, 2, 3))
        
        # Confidence: how far from threshold (normalized)
        # High confidence = error far from threshold (either way)
        distance_from_threshold = np.abs(reconstruction_errors - threshold)
        max_distance = np.max(distance_from_threshold)
        
        confidence = distance_from_threshold / (max_distance + 1e-10)
        
        # Classification
        is_anomaly = reconstruction_errors > threshold
        
        return {
            'reconstruction_errors': reconstruction_errors,
            'confidence': confidence,
            'is_anomaly': is_anomaly,
            'mean_confidence': float(np.mean(confidence))
        }
    
    def prediction_interval(self, x_input, confidence_level=0.95, num_samples=50):
        """
        Compute prediction intervals
        
        Args:
            x_input: Input data
            confidence_level: Confidence level (e.g., 0.95 for 95% interval)
            num_samples: Number of samples for interval estimation
            
        Returns:
            Prediction intervals
        """
        predictions = []
        
        for _ in range(num_samples):
            pred = self.model(x_input, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        # Compute percentiles
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(predictions, upper_percentile, axis=0)
        mean_pred = np.mean(predictions, axis=0)
        
        # Interval width (measure of uncertainty)
        interval_width = np.mean(upper_bound - lower_bound)
        
        return {
            'mean': mean_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': float(interval_width),
            'confidence_level': confidence_level
        }
    
    def epistemic_aleatoric_decomposition(self, x_input, num_samples=30):
        """
        Decompose uncertainty into epistemic and aleatoric components
        
        Args:
            x_input: Input data
            num_samples: Number of samples
            
        Returns:
            Decomposed uncertainty
        """
        # Multiple forward passes
        predictions = []
        for _ in range(num_samples):
            pred = self.model(x_input, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        # Epistemic uncertainty (model uncertainty)
        mean_predictions = np.mean(predictions, axis=0)
        epistemic = np.var(predictions, axis=0)
        
        # Aleatoric uncertainty (data uncertainty) - estimated from prediction variance
        # For autoencoder, we use reconstruction error as proxy
        aleatoric = np.mean(np.square(mean_predictions - x_input), axis=(1, 2, 3))
        
        total_uncertainty = np.mean(epistemic) + np.mean(aleatoric)
        
        return {
            'epistemic': float(np.mean(epistemic)),
            'aleatoric': float(np.mean(aleatoric)),
            'total': float(total_uncertainty),
            'epistemic_ratio': float(np.mean(epistemic) / (total_uncertainty + 1e-10))
        }


class BayesianAutoencoder:
    """Bayesian autoencoder for uncertainty quantification"""
    
    def __init__(self, time_step, num_signals):
        """
        Initialize Bayesian autoencoder
        
        Args:
            time_step: Number of time steps
            num_signals: Number of signals
        """
        self.time_step = time_step
        self.num_signals = num_signals
        self.model = self._build_model()
    
    def _build_model(self):
        """Build Bayesian autoencoder with dropout layers"""
        from tensorflow.keras import layers, Model
        
        input_shape = (self.time_step, self.num_signals, 1)
        inputs = layers.Input(shape=input_shape)
        
        # Encoder with dropout
        x = layers.ZeroPadding2D((2, 2))(inputs)
        x = layers.Conv2D(32, (5, 5), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.2)(x, training=True)  # Always active
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        x = layers.Conv2D(16, (5, 5), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.2)(x, training=True)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        x = layers.Conv2D(16, (3, 3), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.2)(x, training=True)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Decoder with dropout
        x = layers.Conv2D(16, (3, 3), padding='same')(encoded)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.2)(x, training=True)
        x = layers.UpSampling2D((2, 2))(x)
        
        x = layers.Conv2D(16, (5, 5), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.2)(x, training=True)
        x = layers.UpSampling2D((2, 2))(x)
        
        x = layers.Conv2D(32, (5, 5), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.2)(x, training=True)
        x = layers.UpSampling2D((2, 2))(x)
        
        x = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
        # Crop to match input
        temp_shape = x.shape
        diff_h = temp_shape[1] - input_shape[0]
        diff_w = temp_shape[2] - input_shape[1]
        if diff_h > 0 or diff_w > 0:
            x = layers.Cropping2D(cropping=((diff_h//2, diff_h-diff_h//2),
                                           (diff_w//2, diff_w-diff_w//2)))(x)
        
        model = Model(inputs, x, name='bayesian_autoencoder')
        return model
    
    def compile_and_train(self, x_train, epochs=100, batch_size=128):
        """Train Bayesian autoencoder"""
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        history = self.model.fit(
            x_train, x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        
        return history
    
    def predict_with_uncertainty(self, x_input, num_samples=30):
        """
        Predict with uncertainty estimates
        
        Args:
            x_input: Input data
            num_samples: Number of stochastic forward passes
            
        Returns:
            Predictions with uncertainty
        """
        estimator = UncertaintyEstimator(self.model)
        return estimator.monte_carlo_dropout(x_input, num_samples=num_samples)


class ConfidenceCalibration:
    """Calibrate confidence scores"""
    
    def __init__(self, model):
        """
        Initialize calibration
        
        Args:
            model: Trained model
        """
        self.model = model
        self.calibration_params = None
    
    def calibrate(self, x_cal, y_cal, threshold, method='isotonic'):
        """
        Calibrate confidence scores using calibration set
        
        Args:
            x_cal: Calibration data
            y_cal: True labels (0=normal, 1=attack)
            threshold: Detection threshold
            method: Calibration method ('isotonic', 'platt')
            
        Returns:
            Calibration parameters
        """
        from sklearn.calibration import calibration_curve
        
        # Get predictions
        predictions = self.model.predict(x_cal, verbose=0)
        reconstruction_errors = np.mean(np.square(predictions - x_cal), axis=(1, 2, 3))
        
        # Convert to probabilities
        # High error = high probability of attack
        probs = reconstruction_errors / (np.max(reconstruction_errors) + 1e-10)
        
        # Compute calibration curve
        true_probs, pred_probs = calibration_curve(y_cal, probs, n_bins=10)
        
        self.calibration_params = {
            'true_probs': true_probs.tolist(),
            'pred_probs': pred_probs.tolist(),
            'method': method
        }
        
        return self.calibration_params
    
    def get_calibrated_confidence(self, x_input):
        """
        Get calibrated confidence scores
        
        Args:
            x_input: Input data
            
        Returns:
            Calibrated confidence scores
        """
        if self.calibration_params is None:
            raise ValueError("Model not calibrated. Call calibrate() first.")
        
        predictions = self.model.predict(x_input, verbose=0)
        reconstruction_errors = np.mean(np.square(predictions - x_input), axis=(1, 2, 3))
        
        # Normalize
        probs = reconstruction_errors / (np.max(reconstruction_errors) + 1e-10)
        
        # Apply calibration (simple linear interpolation)
        calibrated_probs = np.interp(probs, 
                                     self.calibration_params['pred_probs'],
                                     self.calibration_params['true_probs'])
        
        return calibrated_probs


def uncertainty_aware_detection(model, x_input, threshold, confidence_threshold=0.8):
    """
    Perform detection with uncertainty awareness
    
    Args:
        model: Trained model
        x_input: Input data
        threshold: Detection threshold
        confidence_threshold: Minimum confidence for detection
        
    Returns:
        Detection results with confidence
    """
    estimator = UncertaintyEstimator(model)
    
    # Get predictions with uncertainty
    mc_results = estimator.monte_carlo_dropout(x_input, num_samples=30)
    
    # Reconstruction errors
    mean_pred = mc_results['mean']
    std_pred = mc_results['std']
    
    reconstruction_errors = np.mean(np.square(mean_pred - x_input), axis=(1, 2, 3))
    uncertainty = np.mean(std_pred, axis=(1, 2, 3))
    
    # Confidence: inverse of uncertainty
    max_uncertainty = np.max(uncertainty)
    confidence = 1 - (uncertainty / (max_uncertainty + 1e-10))
    
    # Detection with confidence filtering
    is_anomaly = reconstruction_errors > threshold
    high_confidence_detections = (is_anomaly) & (confidence > confidence_threshold)
    
    results = {
        'is_anomaly': is_anomaly,
        'high_confidence_detections': high_confidence_detections,
        'reconstruction_errors': reconstruction_errors,
        'confidence': confidence,
        'uncertainty': uncertainty,
        'num_detections': int(np.sum(is_anomaly)),
        'num_high_confidence': int(np.sum(high_confidence_detections))
    }
    
    return results

