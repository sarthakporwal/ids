
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent))

from dataset.load_dataset import *
from training.get_autoencoder import get_autoencoder
from adversarial.adversarial_training import train_robust_autoencoder
from adversarial.attacks import AdversarialAttacks
from adversarial.robustness_metrics import RobustnessEvaluator
from model_compression.quantization import ModelQuantizer
from model_compression.pruning import ModelPruner
from uncertainty.uncertainty_estimation import BayesianAutoencoder, UncertaintyEstimator
from domain_adaptation.domain_adversarial import create_domain_adaptive_model


@hydra.main(version_base=None, config_path="../config", config_name="robust_canshield")
def train_robust_canshield(args: DictConfig) -> None:
    print("="*70)
    print("ROBUST CANSHIELD - Adversarially Robust CAN-IDS")
    print("="*70)
    
    root_dir = Path(__file__).resolve().parent
    args.root_dir = root_dir
    args.data_type = "training"
    args.data_dir = args.train_data_dir
    
    dataset_name = args.dataset_name
    num_signals = args.num_signals
    
    training_mode = args.get('training_mode', 'adversarial')
    use_compression = args.get('use_compression', True)
    use_uncertainty = args.get('use_uncertainty', True)
    
    print(f"\nTraining Configuration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Training Mode: {training_mode}")
    print(f"  Use Compression: {use_compression}")
    print(f"  Use Uncertainty: {use_uncertainty}")
    print(f"  Root Directory: {root_dir}")
    
    for time_step in args.time_steps:
        for sampling_period in args.sampling_periods:
            print(f"\n{'='*70}")
            print(f"Training Model: TimeStep={time_step}, SamplingPeriod={sampling_period}")
            print(f"{'='*70}")
            
            args.time_step = time_step
            args.sampling_period = sampling_period
            args.window_step = args.window_step_train
            
            autoencoder, retrain = get_autoencoder(args)
            
            print("\nLoading training data...")
            file_dir_dict = get_list_of_files(args)
            
            if len(file_dir_dict) == 0:
                print("ERROR: No training files found!")
                continue
            
            print("\n⚠️  Memory optimization: Using first training file only")
            print("   (To use all files, increase your RAM or reduce window_step_train)")
            
            file_name, file_path = list(file_dir_dict.items())[0]
            print(f"\nLoading file: {file_name}")
            
            try:
                x_train_combined, _ = load_data_create_images(args, file_name, file_path)
                print(f"  Loaded {len(x_train_combined)} samples")
            except Exception as error:
                print(f"  ERROR: {error}")
                continue
            
            if len(x_train_combined) == 0:
                print("ERROR: No data loaded successfully!")
                continue
            
            
            if training_mode == 'adversarial':
                print("\n" + "="*70)
                print("ADVERSARIAL ROBUST TRAINING")
                print("="*70)
                
                trained_model, history = train_robust_autoencoder(
                    autoencoder,
                    x_train_combined,
                    args,
                    file_index=0,
                    use_adversarial=True
                )
                
                print("\nEvaluating adversarial robustness...")
                evaluator = RobustnessEvaluator(trained_model)
                
                sample_size = min(1000, len(x_train_combined))
                x_eval = x_train_combined[:sample_size]
                
                robustness_report = evaluator.generate_robustness_report(
                    x_eval,
                    save_path=f"{root_dir}/../artifacts/robustness/{dataset_name}/"
                                f"robustness_report_{time_step}_{sampling_period}.json"
                )
                
                print(f"\nRobustness Score: {robustness_report['overall_score']['robustness_score']:.4f}")
            
            elif training_mode == 'domain_adaptive':
                print("\n" + "="*70)
                print("DOMAIN ADAPTIVE TRAINING")
                print("="*70)
                
                da_model = create_domain_adaptive_model(time_step, num_signals, num_domains=2)
                
                split_idx = len(x_train_combined) // 2
                x_source = x_train_combined[:split_idx]
                x_target = x_train_combined[split_idx:]
                
                domain_labels_source = np.zeros((len(x_source), 2))
                domain_labels_source[:, 0] = 1
                
                domain_labels_target = np.zeros((len(x_target), 2))
                domain_labels_target[:, 1] = 1
                
                history = da_model.train(
                    x_source, x_target,
                    domain_labels_source, domain_labels_target,
                    epochs=args.max_epoch,
                    batch_size=128
                )
                
                trained_model = da_model.get_autoencoder()
            
            elif training_mode == 'bayesian':
                print("\n" + "="*70)
                print("BAYESIAN TRAINING (with Uncertainty)")
                print("="*70)
                
                bayesian_ae = BayesianAutoencoder(time_step, num_signals)
                history = bayesian_ae.compile_and_train(
                    x_train_combined,
                    epochs=args.max_epoch,
                    batch_size=128
                )
                
                trained_model = bayesian_ae.model
                
                print("\nTesting uncertainty estimation...")
                x_test_sample = x_train_combined[:100]
                uncertainty_results = bayesian_ae.predict_with_uncertainty(x_test_sample, num_samples=30)
                print(f"  Epistemic Uncertainty: {uncertainty_results['epistemic_uncertainty']:.6f}")
            
            else:
                print("\n" + "="*70)
                print("STANDARD TRAINING")
                print("="*70)
                
                trained_model, history = train_robust_autoencoder(
                    autoencoder,
                    x_train_combined,
                    args,
                    file_index=0,
                    use_adversarial=False
                )
            
            
            if use_compression:
                print("\n" + "="*70)
                print("MODEL COMPRESSION")
                print("="*70)
                
                print("\n1. Quantization...")
                quantizer = ModelQuantizer(trained_model)
                
                quantization_save_dir = f"{root_dir}/../artifacts/compressed/{dataset_name}/" \
                                       f"quantized_{time_step}_{sampling_period}"
                
                x_test_sample = x_train_combined[-1000:]
                x_train_sample = x_train_combined[:1000]
                
                quantization_results = quantizer.compare_quantization_methods(
                    x_train_sample,
                    x_test_sample,
                    save_dir=quantization_save_dir
                )
                
                print(f"\nBest quantization: Int8")
                print(f"  Compression: {quantization_results['int8']['compression_ratio']:.2f}x")
                print(f"  Accuracy Retention: {quantization_results['int8']['accuracy_retention_%']:.2f}%")
                
                print("\n2. Pruning...")
                pruner = ModelPruner(trained_model)
                pruned_model, _ = pruner.magnitude_based_pruning(
                    x_train_combined[:5000],
                    target_sparsity=0.5,
                    epochs=20
                )
                
                pruning_metrics = pruner.evaluate_pruned_model(pruned_model, x_test_sample)
                
                pruning_save_path = f"{root_dir}/../artifacts/compressed/{dataset_name}/" \
                                   f"pruned_{time_step}_{sampling_period}.h5"
                pruner.export_pruned_model(pruned_model, pruning_save_path)
            
            
            model_dir = f"{root_dir}/../artifacts/models/{dataset_name}/" \
                       f"robust_canshield_{training_mode}_{time_step}_{num_signals}_{sampling_period}.h5"
            
            Path(model_dir).parent.mkdir(parents=True, exist_ok=True)
            trained_model.save(model_dir)
            print(f"\n✓ Model saved to: {model_dir}")
            
            metadata = {
                'dataset': dataset_name,
                'time_step': time_step,
                'num_signals': num_signals,
                'sampling_period': sampling_period,
                'training_mode': training_mode,
                'num_samples': len(x_train_combined),
                'use_compression': use_compression,
                'use_uncertainty': use_uncertainty
            }
            
            metadata_path = Path(model_dir).with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Metadata saved to: {metadata_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    train_robust_canshield()

