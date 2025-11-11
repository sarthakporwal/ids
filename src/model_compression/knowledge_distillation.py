
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class KnowledgeDistillation:
    
    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.5):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        for layer in self.teacher_model.layers:
            layer.trainable = False
    
    def distillation_loss(self, y_true, y_student, y_teacher):
        student_loss = tf.reduce_mean(tf.square(y_student - y_true))
        
        distillation_loss = tf.reduce_mean(tf.square(y_student - y_teacher))
        
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        
        return total_loss
    
    def train(self, x_train, epochs=100, batch_size=128, validation_split=0.1):
        print("Training student model with knowledge distillation...")
        
        val_size = int(len(x_train) * validation_split)
        x_val = x_train[-val_size:]
        x_train = x_train[:-val_size]
        
        print("Generating teacher predictions...")
        teacher_preds_train = self.teacher_model.predict(x_train, verbose=0)
        teacher_preds_val = self.teacher_model.predict(x_val, verbose=0)
        
        self.student_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        history = {'loss': [], 'val_loss': [], 'distillation_loss': [], 'student_loss': []}
        
        for epoch in range(epochs):
            indices = np.random.permutation(len(x_train))
            x_train_shuffled = x_train[indices]
            teacher_preds_shuffled = teacher_preds_train[indices]
            
            epoch_losses = {'total': [], 'distillation': [], 'student': []}
            
            num_batches = len(x_train) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                x_batch = x_train_shuffled[start_idx:end_idx]
                teacher_batch = teacher_preds_shuffled[start_idx:end_idx]
                
                with tf.GradientTape() as tape:
                    student_pred = self.student_model(x_batch, training=True)
                    
                    student_loss = tf.reduce_mean(tf.square(student_pred - x_batch))
                    distill_loss = tf.reduce_mean(tf.square(student_pred - teacher_batch))
                    total_loss = self.alpha * distill_loss + (1 - self.alpha) * student_loss
                
                gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
                self.student_model.optimizer.apply_gradients(
                    zip(gradients, self.student_model.trainable_variables)
                )
                
                epoch_losses['total'].append(float(total_loss.numpy()))
                epoch_losses['distillation'].append(float(distill_loss.numpy()))
                epoch_losses['student'].append(float(student_loss.numpy()))
            
            val_pred = self.student_model.predict(x_val, verbose=0)
            val_loss = np.mean(np.square(val_pred - x_val))
            
            history['loss'].append(np.mean(epoch_losses['total']))
            history['val_loss'].append(float(val_loss))
            history['distillation_loss'].append(np.mean(epoch_losses['distillation']))
            history['student_loss'].append(np.mean(epoch_losses['student']))
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - "
                      f"Loss: {history['loss'][-1]:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"Distill: {history['distillation_loss'][-1]:.6f}")
        
        return history
    
    def evaluate_student(self, x_test):
        teacher_pred = self.teacher_model.predict(x_test, verbose=0)
        student_pred = self.student_model.predict(x_test, verbose=0)
        
        teacher_mse = np.mean(np.square(teacher_pred - x_test))
        teacher_mae = np.mean(np.abs(teacher_pred - x_test))
        
        student_mse = np.mean(np.square(student_pred - x_test))
        student_mae = np.mean(np.abs(student_pred - x_test))
        
        agreement = np.mean(np.square(teacher_pred - student_pred))
        
        import tempfile
        from pathlib import Path
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as tmp1:
            self.teacher_model.save(tmp1.name)
            teacher_size = Path(tmp1.name).stat().st_size
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as tmp2:
            self.student_model.save(tmp2.name)
            student_size = Path(tmp2.name).stat().st_size
        
        metrics = {
            'teacher_mse': float(teacher_mse),
            'teacher_mae': float(teacher_mae),
            'student_mse': float(student_mse),
            'student_mae': float(student_mae),
            'teacher_student_agreement': float(agreement),
            'performance_ratio': float(student_mse / teacher_mse),
            'teacher_size_mb': float(teacher_size / (1024 * 1024)),
            'student_size_mb': float(student_size / (1024 * 1024)),
            'compression_ratio': float(teacher_size / student_size)
        }
        
        print("\nKnowledge Distillation Results:")
        print(f"  Teacher MSE: {teacher_mse:.6f}")
        print(f"  Student MSE: {student_mse:.6f}")
        print(f"  Performance Ratio: {metrics['performance_ratio']:.2f}x")
        print(f"  Compression Ratio: {metrics['compression_ratio']:.2f}x")
        print(f"  Teacher-Student Agreement: {agreement:.6f}")
        
        return metrics


def create_student_model(time_step, num_signals, compression_ratio=4):
    input_shape = (time_step, num_signals, 1)
    
    base_filters = 32 // compression_ratio
    
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        layers.ZeroPadding2D((2, 2)),
        layers.Conv2D(base_filters, (3, 3), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D((2, 2), padding='same'),
        
        layers.Conv2D(base_filters // 2, (3, 3), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D((2, 2), padding='same'),
        
        layers.Conv2D(base_filters // 2, (3, 3), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.UpSampling2D((2, 2)),
        
        layers.Conv2D(base_filters, (3, 3), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.UpSampling2D((2, 2)),
        
        layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
    ])
    
    
    return model


def distill_ensemble_to_single(teacher_models, time_step, num_signals, 
                               x_train, x_test, save_path=None):
    print("Distilling ensemble into single student model...")
    
    student_model = create_student_model(time_step, num_signals, compression_ratio=2)
    
    print("Computing ensemble predictions...")
    ensemble_preds = []
    for model in teacher_models:
        preds = model.predict(x_train, verbose=0)
        ensemble_preds.append(preds)
    
    ensemble_preds = np.mean(ensemble_preds, axis=0)
    
    print("Training student model...")
    student_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    history = student_model.fit(
        x_train, ensemble_preds,
        epochs=100,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )
    
    test_preds = []
    for model in teacher_models:
        test_preds.append(model.predict(x_test, verbose=0))
    
    ensemble_test_preds = np.mean(test_preds, axis=0)
    student_test_preds = student_model.predict(x_test, verbose=0)
    
    ensemble_mse = np.mean(np.square(ensemble_test_preds - x_test))
    student_mse = np.mean(np.square(student_test_preds - x_test))
    
    metrics = {
        'ensemble_mse': float(ensemble_mse),
        'student_mse': float(student_mse),
        'performance_ratio': float(student_mse / ensemble_mse),
        'num_teachers': len(teacher_models)
    }
    
    print(f"\nEnsemble to Single Distillation:")
    print(f"  Ensemble MSE: {ensemble_mse:.6f}")
    print(f"  Student MSE: {student_mse:.6f}")
    print(f"  Performance Ratio: {metrics['performance_ratio']:.2f}x")
    
    if save_path:
        student_model.save(save_path)
        print(f"Student model saved to {save_path}")
    
    return student_model, metrics

