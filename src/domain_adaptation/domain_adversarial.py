
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np


@tf.custom_gradient
def gradient_reversal(x, alpha=1.0):
    def grad(dy):
        return -alpha * dy, None
    return x, grad


class GradientReversalLayer(layers.Layer):
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
    
    def __init__(self, time_step, num_signals, num_domains=2, alpha=1.0):
        self.time_step = time_step
        self.num_signals = num_signals
        self.num_domains = num_domains
        self.alpha = alpha
        
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.domain_classifier = self._build_domain_classifier()
        
        self.model = self._build_model()
    
    def _build_encoder(self):
        input_shape = (self.time_step, self.num_signals, 1)
        
        inputs = layers.Input(shape=input_shape, name='encoder_input')
        
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
        encoder_output_shape = self.encoder.output_shape[1:]
        
        inputs = layers.Input(shape=encoder_output_shape, name='decoder_input')
        
        x = layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', name='dec_conv1')(inputs)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        x = layers.Conv2D(16, (5, 5), strides=(1, 1), padding='same', name='dec_conv2')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        x = layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='dec_conv3')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        x = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
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
        encoder_output_shape = self.encoder.output_shape[1:]
        
        inputs = layers.Input(shape=encoder_output_shape, name='domain_input')
        
        x = GradientReversalLayer(alpha=self.alpha)(inputs)
        
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu', name='domain_fc1')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu', name='domain_fc2')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(self.num_domains, activation='softmax', name='domain_output')(x)
        
        domain_classifier = Model(inputs, x, name='domain_classifier')
        return domain_classifier
    
    def _build_model(self):
        input_shape = (self.time_step, self.num_signals, 1)
        inputs = layers.Input(shape=input_shape, name='input')
        
        encoded = self.encoder(inputs)
        
        reconstructed = self.decoder(encoded)
        
        domain_pred = self.domain_classifier(encoded)
        
        model = Model(inputs=inputs, 
                     outputs=[reconstructed, domain_pred],
                     name='domain_adaptive_autoencoder')
        
        return model
    
    def compile(self, learning_rate=0.0002, reconstruction_weight=1.0, domain_weight=0.5):
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
        x_combined = np.concatenate([x_source, x_target], axis=0)
        domain_labels = np.concatenate([domain_labels_source, domain_labels_target], axis=0)
        
        history = self.model.fit(
            x_combined,
            [x_combined, domain_labels],
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        return history
    
    def get_autoencoder(self):
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
    
    def __init__(self, base_model, meta_lr=0.001, inner_lr=0.01):
        self.base_model = base_model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.meta_optimizer = keras.optimizers.Adam(learning_rate=meta_lr)
    
    def adapt_to_vehicle(self, x_vehicle, num_steps=5):
        adapted_model = keras.models.clone_model(self.base_model)
        adapted_model.set_weights(self.base_model.get_weights())
        
        adapted_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.inner_lr),
            loss='mse',
            metrics=['mae']
        )
        
        adapted_model.fit(
            x_vehicle,
            x_vehicle,
            epochs=num_steps,
            batch_size=32,
            verbose=0
        )
        
        return adapted_model
    
    def meta_train(self, vehicle_datasets, num_episodes=100, support_size=100, query_size=50):
        for episode in range(num_episodes):
            vehicle_idx = np.random.randint(0, len(vehicle_datasets))
            vehicle_data = vehicle_datasets[vehicle_idx]
            
            indices = np.random.permutation(len(vehicle_data))
            support_indices = indices[:support_size]
            query_indices = indices[support_size:support_size+query_size]
            
            x_support = vehicle_data[support_indices]
            x_query = vehicle_data[query_indices]
            
            adapted_model = self.adapt_to_vehicle(x_support, num_steps=5)
            
            with tf.GradientTape() as tape:
                predictions = adapted_model(x_query, training=True)
                loss = tf.reduce_mean(tf.square(predictions - x_query))
            
            gradients = tape.gradient(loss, self.base_model.trainable_variables)
            self.meta_optimizer.apply_gradients(zip(gradients, self.base_model.trainable_variables))
            
            if episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes}, Loss: {loss.numpy():.4f}")
        
        return self.base_model


def create_domain_adaptive_model(time_step, num_signals, num_domains=2):
    model = DomainAdaptiveAutoencoder(time_step, num_signals, num_domains)
    model.compile()
    return model

