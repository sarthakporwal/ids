"""
Domain Adversarial Neural Network (DANN) for Cross-Vehicle Generalization
Implements domain-invariant feature learning for CAN-IDS
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np


@tf.custom_gradient
def gradient_reversal(x, alpha=1.0):
    """
    Gradient Reversal Layer
    Forward pass: identity
    Backward pass: multiply gradient by -alpha
    """
    def grad(dy):
        return -alpha * dy, None
    return x, grad


class GradientReversalLayer(layers.Layer):
    """
    Gradient Reversal Layer for Domain Adversarial Training
    """
    def __init__(self, alpha=1.0, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.alpha = alpha
    
    def call(self, x):
        return gradient_reversal(x, self.alpha)
    
    def get_config(self):
        config = super().get_config()
        config.update({'alpha': self.alpha})
        return config


class DomainAdaptiveAutoencoder:
    """
    Domain Adaptive Autoencoder for Cross-Vehicle IDS
    Uses DANN to learn vehicle-agnostic features
    """
    
    def __init__(self, time_step, num_signals, num_domains=2, alpha=1.0):
        """
        Initialize domain adaptive autoencoder
        
        Args:
            time_step: Number of time steps
            num_signals: Number of CAN signals
            num_domains: Number of vehicle types/domains
            alpha: Gradient reversal strength
        """
        self.time_step = time_step
        self.num_signals = num_signals
        self.num_domains = num_domains
        self.alpha = alpha
        
        # Build model components
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.domain_classifier = self._build_domain_classifier()
        
        # Build combined model
        self.model = self._build_model()
    
    def _build_encoder(self):
        """Build encoder network"""
        input_shape = (self.time_step, self.num_signals, 1)
        
        inputs = layers.Input(shape=input_shape, name='encoder_input')
        
        # Encoder layers
        x = layers.ZeroPadding2D((2, 2))(inputs)
        x = layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='enc_conv1')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        x = layers.Conv2D(16, (5, 5), strides=(1, 1), padding='same', name='enc_conv2')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        x = layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', name='enc_conv3')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.MaxPooling2D((2, 2), padding='same', name='encoder_output')(x)
        
        encoder = Model(inputs, x, name='encoder')
        return encoder
    
    def _build_decoder(self):
        """Build decoder network"""
        # Get encoder output shape
        encoder_output_shape = self.encoder.output_shape[1:]
        
        inputs = layers.Input(shape=encoder_output_shape, name='decoder_input')
        
        # Decoder layers
        x = layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', name='dec_conv1')(inputs)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        x = layers.Conv2D(16, (5, 5), strides=(1, 1), padding='same', name='dec_conv2')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        x = layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='dec_conv3')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        # Output layer
        x = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
        # Crop to match input size
        temp_shape = x.shape
        input_shape = (self.time_step, self.num_signals, 1)
        diff_h = temp_shape[1] - input_shape[0]
        diff_w = temp_shape[2] - input_shape[1]
        
        top = diff_h // 2
        bottom = diff_h - top
        left = diff_w // 2
        right = diff_w - left
        
        if diff_h > 0 or diff_w > 0:
            x = layers.Cropping2D(cropping=((top, bottom), (left, right)))(x)
        
        decoder = Model(inputs, x, name='decoder')
        return decoder
    
    def _build_domain_classifier(self):
        """Build domain classifier network"""
        encoder_output_shape = self.encoder.output_shape[1:]
        
        inputs = layers.Input(shape=encoder_output_shape, name='domain_input')
        
        # Gradient reversal
        x = GradientReversalLayer(alpha=self.alpha)(inputs)
        
        # Domain classification layers
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu', name='domain_fc1')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu', name='domain_fc2')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(self.num_domains, activation='softmax', name='domain_output')(x)
        
        domain_classifier = Model(inputs, x, name='domain_classifier')
        return domain_classifier
    
    def _build_model(self):
        """Build complete DANN model"""
        input_shape = (self.time_step, self.num_signals, 1)
        inputs = layers.Input(shape=input_shape, name='input')
        
        # Encoder
        encoded = self.encoder(inputs)
        
        # Decoder (reconstruction)
        reconstructed = self.decoder(encoded)
        
        # Domain classifier
        domain_pred = self.domain_classifier(encoded)
        
        # Create model with multiple outputs
        model = Model(inputs=inputs, 
                     outputs=[reconstructed, domain_pred],
                     name='domain_adaptive_autoencoder')
        
        return model
    
    def compile(self, learning_rate=0.0002, reconstruction_weight=1.0, domain_weight=0.5):
        """
        Compile model with multiple loss functions
        
        Args:
            learning_rate: Learning rate
            reconstruction_weight: Weight for reconstruction loss
            domain_weight: Weight for domain classification loss
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.99)
        
        self.model.compile(
            optimizer=optimizer,
            loss={
                'decoder': 'mse',
                'domain_classifier': 'categorical_crossentropy'
            },
            loss_weights={
                'decoder': reconstruction_weight,
                'domain_classifier': domain_weight
            },
            metrics={
                'decoder': ['mae'],
                'domain_classifier': ['accuracy']
            }
        )
    
    def train(self, x_source, x_target, domain_labels_source, domain_labels_target,
              epochs=100, batch_size=128, validation_split=0.1):
        """
        Train domain adaptive autoencoder
        
        Args:
            x_source: Source domain data
            x_target: Target domain data
            domain_labels_source: Domain labels for source (one-hot)
            domain_labels_target: Domain labels for target (one-hot)
            epochs: Number of epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            
        Returns:
            Training history
        """
        # Combine source and target data
        x_combined = np.concatenate([x_source, x_target], axis=0)
        domain_labels = np.concatenate([domain_labels_source, domain_labels_target], axis=0)
        
        # Training
        history = self.model.fit(
            x_combined,
            [x_combined, domain_labels],  # Reconstruction and domain labels
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        return history
    
    def get_autoencoder(self):
        """Get the autoencoder (encoder + decoder) without domain classifier"""
        input_shape = (self.time_step, self.num_signals, 1)
        inputs = layers.Input(shape=input_shape)
        
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        
        autoencoder = Model(inputs, decoded, name='autoencoder')
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.99),
            loss='mse',
            metrics=['mae']
        )
        
        return autoencoder


class MetaLearningAutoencoder:
    """
    Meta-learning approach for quick adaptation to new vehicles
    Uses Model-Agnostic Meta-Learning (MAML) style training
    """
    
    def __init__(self, base_model, meta_lr=0.001, inner_lr=0.01):
        """
        Initialize meta-learning autoencoder
        
        Args:
            base_model: Base autoencoder model
            meta_lr: Meta learning rate
            inner_lr: Inner loop learning rate
        """
        self.base_model = base_model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.meta_optimizer = keras.optimizers.Adam(learning_rate=meta_lr)
    
    def adapt_to_vehicle(self, x_vehicle, num_steps=5):
        """
        Quick adaptation to new vehicle with few samples
        
        Args:
            x_vehicle: Small dataset from new vehicle
            num_steps: Number of adaptation steps
            
        Returns:
            Adapted model
        """
        # Clone model
        adapted_model = keras.models.clone_model(self.base_model)
        adapted_model.set_weights(self.base_model.get_weights())
        
        # Compile with inner learning rate
        adapted_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.inner_lr),
            loss='mse',
            metrics=['mae']
        )
        
        # Fine-tune on vehicle-specific data
        adapted_model.fit(
            x_vehicle,
            x_vehicle,
            epochs=num_steps,
            batch_size=32,
            verbose=0
        )
        
        return adapted_model
    
    def meta_train(self, vehicle_datasets, num_episodes=100, support_size=100, query_size=50):
        """
        Meta-training across multiple vehicles
        
        Args:
            vehicle_datasets: List of datasets from different vehicles
            num_episodes: Number of meta-training episodes
            support_size: Size of support set per vehicle
            query_size: Size of query set per vehicle
            
        Returns:
            Meta-trained model
        """
        for episode in range(num_episodes):
            # Sample random vehicle
            vehicle_idx = np.random.randint(0, len(vehicle_datasets))
            vehicle_data = vehicle_datasets[vehicle_idx]
            
            # Split into support and query
            indices = np.random.permutation(len(vehicle_data))
            support_indices = indices[:support_size]
            query_indices = indices[support_size:support_size+query_size]
            
            x_support = vehicle_data[support_indices]
            x_query = vehicle_data[query_indices]
            
            # Inner loop: adapt to support set
            adapted_model = self.adapt_to_vehicle(x_support, num_steps=5)
            
            # Outer loop: evaluate on query set and update meta-parameters
            with tf.GradientTape() as tape:
                predictions = adapted_model(x_query, training=True)
                loss = tf.reduce_mean(tf.square(predictions - x_query))
            
            # Update base model
            gradients = tape.gradient(loss, self.base_model.trainable_variables)
            self.meta_optimizer.apply_gradients(zip(gradients, self.base_model.trainable_variables))
            
            if episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes}, Loss: {loss.numpy():.4f}")
        
        return self.base_model


def create_domain_adaptive_model(time_step, num_signals, num_domains=2):
    """
    Factory function to create domain adaptive autoencoder
    
    Args:
        time_step: Number of time steps
        num_signals: Number of CAN signals
        num_domains: Number of vehicle domains
        
    Returns:
        Domain adaptive autoencoder instance
    """
    model = DomainAdaptiveAutoencoder(time_step, num_signals, num_domains)
    model.compile()
    return model

