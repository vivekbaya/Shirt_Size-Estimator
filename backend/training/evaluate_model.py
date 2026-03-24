"""
Model Evaluation Script

This script evaluates the trained shirt size prediction model on the test set
and generates detailed performance metrics and visualizations.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from .train_model import SizeDataset, SizeClassifier


class ModelEvaluator:
    """Evaluate trained model performance"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        size_classes: list,
        fit_classes: list,
        save_dir: str = 'evaluation_results'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.size_classes = size_classes
        self.fit_classes = fit_classes
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def evaluate(self, test_loader) -> dict:
        """Evaluate model on test set"""
        
        all_size_preds = []
        all_size_labels = []
        all_fit_preds = []
        all_fit_labels = []
        all_size_probs = []
        all_fit_probs = []
        
        print("Evaluating model on test set...")
        
        with torch.no_grad():
            for features, size_labels, fit_labels in test_loader:
                features = features.to(self.device)
                
                size_logits, fit_logits = self.model(features)
                
                size_probs = torch.softmax(size_logits, dim=1)
                fit_probs = torch.softmax(fit_logits, dim=1)
                
                size_preds = size_logits.argmax(dim=1)
                fit_preds = fit_logits.argmax(dim=1)
                
                all_size_preds.extend(size_preds.cpu().numpy())
                all_size_labels.extend(size_labels.numpy())
                all_fit_preds.extend(fit_preds.cpu().numpy())
                all_fit_labels.extend(fit_labels.numpy())
                all_size_probs.extend(size_probs.cpu().numpy())
                all_fit_probs.extend(fit_probs.cpu().numpy())
        
        results = {
            'size_preds': np.array(all_size_preds),
            'size_labels': np.array(all_size_labels),
            'fit_preds': np.array(all_fit_preds),
            'fit_labels': np.array(all_fit_labels),
            'size_probs': np.array(all_size_probs),
            'fit_probs': np.array(all_fit_probs)
        }
        
        return results
    
    def calculate_metrics(self, results: dict) -> dict:
        """Calculate detailed metrics"""
        
        size_accuracy = accuracy_score(results['size_labels'], results['size_preds'])
        fit_accuracy = accuracy_score(results['fit_labels'], results['fit_preds'])
        
        # Size classification report
        size_report = classification_report(
            results['size_labels'],
            results['size_preds'],
            target_names=self.size_classes,
            output_dict=True
        )
        
        # Fit classification report
        fit_report = classification_report(
            results['fit_labels'],
            results['fit_preds'],
            target_names=self.fit_classes,
            output_dict=True
        )
        
        # Average confidence
        size_confidences = results['size_probs'].max(axis=1)
        fit_confidences = results['fit_probs'].max(axis=1)
        
        metrics = {
            'size_accuracy': float(size_accuracy),
            'fit_accuracy': float(fit_accuracy),
            'avg_size_confidence': float(size_confidences.mean()),
            'avg_fit_confidence': float(fit_confidences.mean()),
            'size_report': size_report,
            'fit_report': fit_report
        }
        
        return metrics
    
    def plot_confusion_matrices(self, results: dict):
        """Plot confusion matrices"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Size confusion matrix
        size_cm = confusion_matrix(results['size_labels'], results['size_preds'])
        sns.heatmap(
            size_cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.size_classes,
            yticklabels=self.size_classes,
            ax=axes[0]
        )
        axes[0].set_title('Size Classification Confusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # Fit confusion matrix
        fit_cm = confusion_matrix(results['fit_labels'], results['fit_preds'])
        sns.heatmap(
            fit_cm,
            annot=True,
            fmt='d',
            cmap='Greens',
            xticklabels=self.fit_classes,
            yticklabels=self.fit_classes,
            ax=axes[1]
        )
        axes[1].set_title('Fit Type Classification Confusion Matrix')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrices.png', dpi=300)
        print(f"Confusion matrices saved to {self.save_dir / 'confusion_matrices.png'}")
    
    def plot_confidence_distribution(self, results: dict):
        """Plot confidence distribution"""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        size_confidences = results['size_probs'].max(axis=1)
        fit_confidences = results['fit_probs'].max(axis=1)
        
        # Size confidence
        axes[0].hist(size_confidences, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(size_confidences.mean(), color='red', linestyle='--', 
                       label=f'Mean: {size_confidences.mean():.3f}')
        axes[0].set_xlabel('Confidence')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Size Prediction Confidence Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Fit confidence
        axes[1].hist(fit_confidences, bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[1].axvline(fit_confidences.mean(), color='red', linestyle='--',
                       label=f'Mean: {fit_confidences.mean():.3f}')
        axes[1].set_xlabel('Confidence')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Fit Type Prediction Confidence Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confidence_distribution.png', dpi=300)
        print(f"Confidence distribution saved to {self.save_dir / 'confidence_distribution.png'}")
    
    def analyze_errors(self, results: dict, test_df: pd.DataFrame):
        """Analyze prediction errors"""
        
        # Size errors
        size_errors = results['size_labels'] != results['size_preds']
        size_error_rate = size_errors.mean()
        
        # Fit errors
        fit_errors = results['fit_labels'] != results['fit_preds']
        fit_error_rate = fit_errors.mean()
        
        print(f"\n{'='*60}")
        print(f"ERROR ANALYSIS")
        print(f"{'='*60}")
        print(f"\nSize Error Rate: {size_error_rate:.2%}")
        print(f"Fit Error Rate: {fit_error_rate:.2%}")
        
        # Analyze off-by-one errors for sizes
        size_labels = results['size_labels']
        size_preds = results['size_preds']
        
        off_by_one = np.abs(size_labels - size_preds) == 1
        off_by_one_rate = off_by_one[size_errors].mean()
        
        print(f"\nOf the size errors, {off_by_one_rate:.2%} are off-by-one")
        print("(e.g., predicting M when truth is L)")
        
        # Most confused pairs
        print(f"\nMost Confused Size Pairs:")
        for i, true_class in enumerate(self.size_classes):
            for j, pred_class in enumerate(self.size_classes):
                if i != j:
                    count = ((size_labels == i) & (size_preds == j)).sum()
                    if count > 0:
                        print(f"  {true_class} → {pred_class}: {count} times")
    
    def save_metrics(self, metrics: dict):
        """Save metrics to JSON"""
        
        filepath = self.save_dir / 'metrics.json'
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {filepath}")
    
    def print_report(self, metrics: dict):
        """Print evaluation report"""
        
        print(f"\n{'='*60}")
        print(f"MODEL EVALUATION REPORT")
        print(f"{'='*60}")
        
        print(f"\n📊 OVERALL METRICS")
        print(f"  Size Classification Accuracy: {metrics['size_accuracy']:.2%}")
        print(f"  Fit Type Classification Accuracy: {metrics['fit_accuracy']:.2%}")
        print(f"  Avg Size Confidence: {metrics['avg_size_confidence']:.3f}")
        print(f"  Avg Fit Confidence: {metrics['avg_fit_confidence']:.3f}")
        
        print(f"\n📏 SIZE CLASSIFICATION METRICS")
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
        print("-" * 60)
        for size in self.size_classes:
            report = metrics['size_report'][size]
            print(f"{size:<10} {report['precision']:<12.3f} {report['recall']:<12.3f} "
                  f"{report['f1-score']:<12.3f} {int(report['support'])}")
        
        print(f"\n👕 FIT TYPE CLASSIFICATION METRICS")
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
        print("-" * 60)
        for fit in self.fit_classes:
            report = metrics['fit_report'][fit]
            print(f"{fit:<10} {report['precision']:<12.3f} {report['recall']:<12.3f} "
                  f"{report['f1-score']:<12.3f} {int(report['support'])}")


def main():
    """Main evaluation pipeline"""
    
    # Configuration
    DATA_DIR = 'data/synthetic_sizes'
    MODEL_DIR = 'models'
    EVAL_DIR = 'evaluation_results'
    BATCH_SIZE = 64
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load metadata
    with open(Path(DATA_DIR) / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    size_classes = metadata['size_classes']
    fit_classes = metadata['fit_classes']
    
    size_to_idx = {size: idx for idx, size in enumerate(size_classes)}
    fit_to_idx = {fit: idx for idx, fit in enumerate(fit_classes)}
    
    # Load test data
    print("Loading test dataset...")
    test_df = pd.read_csv(Path(DATA_DIR) / 'test.csv')
    print(f"Test samples: {len(test_df)}\n")
    
    # Create dataset
    test_dataset = SizeDataset(test_df, size_to_idx, fit_to_idx)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Load model
    print("Loading trained model...")
    checkpoint = torch.load(Path(MODEL_DIR) / 'best_model.pth', map_location=device)
    
    model = SizeClassifier(
        input_dim=4,
        hidden_dim=64,
        num_sizes=len(size_classes),
        num_fits=len(fit_classes),
        dropout=0.3
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully!\n")
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        size_classes=size_classes,
        fit_classes=fit_classes,
        save_dir=EVAL_DIR
    )
    
    # Evaluate
    results = evaluator.evaluate(test_loader)
    metrics = evaluator.calculate_metrics(results)
    
    # Print report
    evaluator.print_report(metrics)
    
    # Analyze errors
    evaluator.analyze_errors(results, test_df)
    
    # Generate visualizations
    print(f"\n{'='*60}")
    print(f"GENERATING VISUALIZATIONS")
    print(f"{'='*60}\n")
    
    evaluator.plot_confusion_matrices(results)
    evaluator.plot_confidence_distribution(results)
    
    # Save metrics
    evaluator.save_metrics(metrics)
    
    print(f"\n✅ Evaluation complete!")
    print(f"\nResults saved to: {EVAL_DIR}/")
    print(f"  - metrics.json")
    print(f"  - confusion_matrices.png")
    print(f"  - confidence_distribution.png")


if __name__ == '__main__':
    main()
