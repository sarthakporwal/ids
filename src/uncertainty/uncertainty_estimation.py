
import tensorflow as tf
import numpy as np
from scipy.stats import entropy


class UncertaintyEstimator:
    
    def __init__(self, model, method='monte_carlo'):
        self.model = model
        self.method = method
    
    def monte_carlo_dropout(self, x_input, num_samples=30, dropout_rate=0.1):
        predictions = []
        
        for _ in range(num_samples):
            pred = self.model(x_input, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)
        
        epistemic_uncertainty = np.mean(std_prediction)
        
        return {
            'mean': mean_prediction,
            'std': std_prediction,
            'epistemic_uncertainty': float(epistemic_uncertainty),
            'all_predictions': predictions
        }
    
    def bootstrap_uncertainty(self, x_input, num_bootstrap=30):
        n_samples = len(x_input)
        predictions = []
        
        for _ in range(num_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            x_bootstrap = x_input[indices]
            
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
        predictions = self.model.predict(x_input, verbose=0)
        
        reconstruction_errors = np.mean(np.square(predictions - x_input), axis=(1, 2, 3))
        
        distance_from_threshold = np.abs(reconstruction_errors - threshold)
        max_distance = np.max(distance_from_threshold)
        
        confidence = distance_from_threshold / (max_distance + 1e-10)
        
        is_anomaly = reconstruction_errors > threshold
        
        return {
            'reconstruction_errors': reconstruction_errors,
            'confidence': confidence,
            'is_anomaly': is_anomaly,
            'mean_confidence': float(np.mean(confidence))
        }
    
    def prediction_interval(self, x_input, confidence_level=0.95, num_samples=50):
        predictions = []
        
        for _ in range(num_samples):
            pred = self.model(x_input, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(predictions, upper_percentile, axis=0)
        mean_pred = np.mean(predictions, axis=0)
        
        interval_width = np.mean(upper_bound - lower_bound)
        
        return {
            'mean': mean_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': float(interval_width),
            'confidence_level': confidence_level
        }
    
    def epistemic_aleatoric_decomposition(self, x_input, num_samples=30):
        predictions = []
        for _ in range(num_samples):
            pred = self.model(x_input, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        mean_predictions = np.mean(predictions, axis=0)
        epistemic = np.var(predictions, axis=0)
        
        aleatoric = np.mean(np.square(mean_predictions - x_input), axis=(1, 2, 3))
        
        total_uncertainty = np.mean(epistemic) + np.mean(aleatoric)
        
        return {
            'epistemic': float(np.mean(epistemic)),
            'aleatoric': float(np.mean(aleatoric)),
            'total': float(total_uncertainty),
            'epistemic_ratio': float(np.mean(epistemic) / (total_uncertainty + 1e-10))
        }


class BayesianAutoencoder:
    
    def __init__(self, time_step, num_signals):
        self.time_step = time_step
        self.num_signals = num_signals
        self.model = self._build_model()
    
    def _build_model(self):
        from tensorflow.keras import layers, Model
        
        input_shape = (self.time_step, self.num_signals, 1)
        inputs = layers.Input(shape=input_shape)
        
        x = layers.ZeroPadding2D((2, 2))(inputs)
        x = layers.Conv2D(32, (5, 5), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.2)(x, training=True)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        x = layers.Conv2D(16, (5, 5), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.2)(x, training=True)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        x = layers.Conv2D(16, (3, 3), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.2)(x, training=True)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
        
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
        
        temp_shape = x.shape
        diff_h = temp_shape[1] - input_shape[0]
        diff_w = temp_shape[2] - input_shape[1]
        if diff_h > 0 or diff_w > 0:
            x = layers.Cropping2D(cropping=((diff_h//2, diff_h-diff_h//2),
                                           (diff_w//2, diff_w-diff_w//2)))(x)
        
        model = Model(inputs, x, name='bayesian_autoencoder')
        return model
    
    def compile_and_train(self, x_train, epochs=100, batch_size=128):
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
        estimator = UncertaintyEstimator(self.model)
        return estimator.monte_carlo_dropout(x_input, num_samples=num_samples)


class ConfidenceCalibration:
    
    def __init__(self, model):
        self.model = model
        self.calibration_params = None
    
    def calibrate(self, x_cal, y_cal, threshold, method='isotonic'):
        from sklearn.calibration import calibration_curve
        
        predictions = self.model.predict(x_cal, verbose=0)
        reconstruction_errors = np.mean(np.square(predictions - x_cal), axis=(1, 2, 3))
        
        probs = reconstruction_errors / (np.max(reconstruction_errors) + 1e-10)
        
        true_probs, pred_probs = calibration_curve(y_cal, probs, n_bins=10)
        
        self.calibration_params = {
            'true_probs': true_probs.tolist(),
            'pred_probs': pred_probs.tolist(),
            'method': method
        }
        
        return self.calibration_params
    
    def get_calibrated_confidence(self, x_input):
        if self.calibration_params is None:
            raise ValueError("Model not calibrated. Call calibrate() first.")
        
        predictions = self.model.predict(x_input, verbose=0)
        reconstruction_errors = np.mean(np.square(predictions - x_input), axis=(1, 2, 3))
        
        probs = reconstruction_errors / (np.max(reconstruction_errors) + 1e-10)
        
        calibrated_probs = np.interp(probs, 
                                     self.calibration_params['pred_probs'],
                                     self.calibration_params['true_probs'])
        
        return calibrated_probs


def uncertainty_aware_detection(model, x_input, threshold, confidence_threshold=0.8):
    estimator = UncertaintyEstimator(model)
    
    mc_results = estimator.monte_carlo_dropout(x_input, num_samples=30)
    
    mean_pred = mc_results['mean']
    std_pred = mc_results['std']
    
    reconstruction_errors = np.mean(np.square(mean_pred - x_input), axis=(1, 2, 3))
    uncertainty = np.mean(std_pred, axis=(1, 2, 3))
    
    max_uncertainty = np.max(uncertainty)
    confidence = 1 - (uncertainty / (max_uncertainty + 1e-10))
    
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

