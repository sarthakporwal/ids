"""
Robust CANShield Evaluation
Comprehensive evaluation including robustness, uncertainty, and performance metrics
"""

import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
import numpy as np
import json
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset.load_dataset import *
from adversarial.attacks import AdversarialAttacks
from adversarial.robustness_metrics import RobustnessEvaluator
from uncertainty.uncertainty_estimation import UncertaintyEstimator, uncertainty_aware_detection
from uncertainty.ensemble_uncertainty import EnsembleUncertainty
from model_compression.quantization import TFLiteInference
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
import tensorflow as tf


@hydra.main(version_base=None, config_path="../config", config_name="robust_canshield")
def evaluate_robust_canshield(args: DictConfig) -> None:
    """
    Comprehensive evaluation of robust CANShield
    
    Args:
        args: Configuration arguments
    """
    print("="*70)
    print("ROBUST CANSHIELD - COMPREHENSIVE EVALUATION")
    print("="*70)
    
    root_dir = Path(__file__).resolve().parent
    args.root_dir = root_dir
    args.data_type = "testing"
    
    dataset_name = args.dataset_name
    training_mode = args.get('training_mode', 'adversarial')
    
    results_all = {}
    
    for time_step in args.time_steps:
        for sampling_period in args.sampling_periods:
            print(f"\n{'='*70}")
            print(f"Evaluating: TimeStep={time_step}, SamplingPeriod={sampling_period}")
            print(f"{'='*70}")
            
            args.time_step = time_step
            args.sampling_period = sampling_period
            args.window_step = args.window_step_test
            
            # Load model
            model_path = f"{root_dir}/../artifacts/models/{dataset_name}/" \
                        f"robust_canshield_{training_mode}_{time_step}_{args.num_signals}_{sampling_period}.h5"
            
            if not Path(model_path).exists():
                print(f"WARNING: Model not found at {model_path}")
                continue
            
            print(f"\nLoading model from: {model_path}")
            model = tf.keras.models.load_model(model_path)
            
            # Load test data for different attack types
            args.data_dir = args.test_data_dir
            test_file_dict = get_list_of_files(args)
            
            attack_results = {}
            
            for attack_name, file_prefix in args.attacks_dict.items():
                print(f"\n--- Evaluating on {attack_name} Attack ---")
                
                # Find matching file
                matching_files = {k: v for k, v in test_file_dict.items() if file_prefix in k}
                
                if not matching_files:
                    print(f"WARNING: No test file found for {attack_name}")
                    continue
                
                file_name, file_path = list(matching_files.items())[0]
                
                try:
                    # Load test data
                    x_test, y_test = load_data_create_images(args, file_name, file_path)
                    print(f"Loaded {len(x_test)} test samples")
                    
                    # ========== 1. STANDARD PERFORMANCE ==========
                    print("\n1. Standard Performance Metrics...")
                    
                    predictions = model.predict(x_test, verbose=0)
                    reconstruction_errors = np.mean(np.square(predictions - x_test), axis=(1, 2, 3))
                    
                    # Compute optimal threshold
                    thresholds = np.percentile(reconstruction_errors, [90, 95, 99])
                    best_f1 = 0
                    best_threshold = thresholds[0]
                    
                    for threshold in thresholds:
                        y_pred = (reconstruction_errors > threshold).astype(int)
                        
                        tp = np.sum((y_pred == 1) & (y_test == 1))
                        fp = np.sum((y_pred == 1) & (y_test == 0))
                        fn = np.sum((y_pred == 0) & (y_test == 1))
                        
                        precision = tp / (tp + fp + 1e-10)
                        recall = tp / (tp + fn + 1e-10)
                        f1 = 2 * precision * recall / (precision + recall + 1e-10)
                        
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = threshold
                    
                    # Compute final metrics with best threshold
                    y_pred = (reconstruction_errors > best_threshold).astype(int)
                    
                    tp = np.sum((y_pred == 1) & (y_test == 1))
                    fp = np.sum((y_pred == 1) & (y_test == 0))
                    fn = np.sum((y_pred == 0) & (y_test == 1))
                    tn = np.sum((y_pred == 0) & (y_test == 0))
                    
                    accuracy = (tp + tn) / len(y_test)
                    precision = tp / (tp + fp + 1e-10)
                    recall = tp / (tp + fn + 1e-10)
                    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
                    fpr = fp / (fp + tn + 1e-10)
                    
                    # ROC-AUC
                    try:
                        roc_auc = roc_auc_score(y_test, reconstruction_errors)
                        precision_curve, recall_curve, _ = precision_recall_curve(y_test, reconstruction_errors)
                        pr_auc = auc(recall_curve, precision_curve)
                    except:
                        roc_auc = 0.0
                        pr_auc = 0.0
                    
                    standard_metrics = {
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1_score),
                        'fpr': float(fpr),
                        'roc_auc': float(roc_auc),
                        'pr_auc': float(pr_auc),
                        'threshold': float(best_threshold)
                    }
                    
                    print(f"  Accuracy: {accuracy:.4f}")
                    print(f"  F1-Score: {f1_score:.4f}")
                    print(f"  ROC-AUC: {roc_auc:.4f}")
                    
                    # ========== 2. ADVERSARIAL ROBUSTNESS ==========
                    print("\n2. Adversarial Robustness Evaluation...")
                    
                    evaluator = RobustnessEvaluator(model)
                    
                    # Sample for robustness testing
                    sample_size = min(500, len(x_test))
                    x_test_sample = x_test[:sample_size]
                    
                    # Robustness score
                    robustness_metrics = evaluator.compute_robustness_score(x_test_sample, epsilon=0.01)
                    
                    # Cross-attack robustness
                    cross_attack = evaluator.evaluate_cross_attack_robustness(x_test_sample)
                    
                    print(f"  Robustness Score: {robustness_metrics['robustness_score']:.4f}")
                    print(f"  Error Stability: {robustness_metrics['error_stability']:.4f}")
                    
                    # ========== 3. UNCERTAINTY QUANTIFICATION ==========
                    print("\n3. Uncertainty Quantification...")
                    
                    uncertainty_estimator = UncertaintyEstimator(model)
                    
                    # Uncertainty-aware detection
                    uncertainty_results = uncertainty_aware_detection(
                        model, x_test_sample, best_threshold, confidence_threshold=0.8
                    )
                    
                    print(f"  High Confidence Detections: {uncertainty_results['num_high_confidence']}/{uncertainty_results['num_detections']}")
                    print(f"  Mean Confidence: {np.mean(uncertainty_results['confidence']):.4f}")
                    print(f"  Mean Uncertainty: {uncertainty_results['uncertainty'].mean():.6f}")
                    
                    # ========== 4. INFERENCE TIME ==========
                    print("\n4. Inference Time Benchmark...")
                    
                    import time
                    sample_input = x_test[:1]
                    
                    # Warmup
                    for _ in range(10):
                        _ = model.predict(sample_input, verbose=0)
                    
                    # Benchmark
                    times = []
                    for _ in range(100):
                        start = time.perf_counter()
                        _ = model.predict(sample_input, verbose=0)
                        end = time.perf_counter()
                        times.append((end - start) * 1000)
                    
                    inference_metrics = {
                        'mean_ms': float(np.mean(times)),
                        'std_ms': float(np.std(times)),
                        'median_ms': float(np.median(times)),
                        'p95_ms': float(np.percentile(times, 95))
                    }
                    
                    print(f"  Mean Inference Time: {inference_metrics['mean_ms']:.2f} ± {inference_metrics['std_ms']:.2f} ms")
                    
                    # ========== AGGREGATE RESULTS ==========
                    
                    attack_results[attack_name] = {
                        'standard_metrics': standard_metrics,
                        'robustness': {
                            'overall_score': robustness_metrics,
                            'cross_attack': cross_attack
                        },
                        'uncertainty': {
                            'high_confidence_detections': int(uncertainty_results['num_high_confidence']),
                            'total_detections': int(uncertainty_results['num_detections']),
                            'mean_confidence': float(np.mean(uncertainty_results['confidence'])),
                            'mean_uncertainty': float(uncertainty_results['uncertainty'].mean())
                        },
                        'inference_time': inference_metrics
                    }
                
                except Exception as error:
                    print(f"ERROR evaluating {attack_name}: {error}")
                    import traceback
                    traceback.print_exc()
            
            # Save results for this configuration
            config_key = f"ts{time_step}_sp{sampling_period}"
            results_all[config_key] = attack_results
            
            # Print summary
            print(f"\n{'='*70}")
            print(f"SUMMARY - TimeStep={time_step}, SamplingPeriod={sampling_period}")
            print(f"{'='*70}")
            print(f"{'Attack':<15} {'F1-Score':<12} {'Robustness':<12} {'Confidence':<12} {'Inference(ms)':<15}")
            print("-"*70)
            
            for attack_name, results in attack_results.items():
                f1 = results['standard_metrics']['f1_score']
                rob = results['robustness']['overall_score']['robustness_score']
                conf = results['uncertainty']['mean_confidence']
                inf_time = results['inference_time']['mean_ms']
                
                print(f"{attack_name:<15} {f1:<12.4f} {rob:<12.4f} {conf:<12.4f} {inf_time:<15.2f}")
    
    # ========== SAVE COMPREHENSIVE RESULTS ==========
    
    results_dir = Path(f"{root_dir}/../artifacts/evaluation_results/{dataset_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / f"comprehensive_evaluation_{training_mode}.json"
    with open(results_file, 'w') as f:
        json.dump(results_all, f, indent=2)
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {results_file}")
    
    # Generate summary report
    generate_summary_report(results_all, results_dir / f"summary_{training_mode}.txt")


def generate_summary_report(results, save_path):
    """Generate human-readable summary report"""
    
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ROBUST CANSHIELD - EVALUATION SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        for config_key, attack_results in results.items():
            f.write(f"\nConfiguration: {config_key}\n")
            f.write("-"*80 + "\n\n")
            
            # Aggregate metrics across attacks
            f1_scores = []
            robustness_scores = []
            inference_times = []
            
            for attack_name, metrics in attack_results.items():
                f.write(f"Attack: {attack_name}\n")
                f.write(f"  Standard Performance:\n")
                f.write(f"    - Accuracy:  {metrics['standard_metrics']['accuracy']:.4f}\n")
                f.write(f"    - F1-Score:  {metrics['standard_metrics']['f1_score']:.4f}\n")
                f.write(f"    - Precision: {metrics['standard_metrics']['precision']:.4f}\n")
                f.write(f"    - Recall:    {metrics['standard_metrics']['recall']:.4f}\n")
                f.write(f"    - ROC-AUC:   {metrics['standard_metrics']['roc_auc']:.4f}\n")
                f.write(f"  Robustness:\n")
                f.write(f"    - Overall Score: {metrics['robustness']['overall_score']['robustness_score']:.4f}\n")
                f.write(f"  Uncertainty:\n")
                f.write(f"    - Mean Confidence: {metrics['uncertainty']['mean_confidence']:.4f}\n")
                f.write(f"  Performance:\n")
                f.write(f"    - Inference Time: {metrics['inference_time']['mean_ms']:.2f} ms\n")
                f.write("\n")
                
                f1_scores.append(metrics['standard_metrics']['f1_score'])
                robustness_scores.append(metrics['robustness']['overall_score']['robustness_score'])
                inference_times.append(metrics['inference_time']['mean_ms'])
            
            f.write(f"\nAggregate Metrics:\n")
            f.write(f"  - Average F1-Score:       {np.mean(f1_scores):.4f}\n")
            f.write(f"  - Average Robustness:     {np.mean(robustness_scores):.4f}\n")
            f.write(f"  - Average Inference Time: {np.mean(inference_times):.2f} ms\n")
            f.write("\n" + "="*80 + "\n")
    
    print(f"Summary report saved to: {save_path}")


if __name__ == "__main__":
    evaluate_robust_canshield()

