"""
Training Script for Shirt Size Prediction Model

This script trains a neural network to predict shirt size and fit type
from body measurement ratios extracted via computer vision.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm


class SizeDataset(Dataset):
    """PyTorch Dataset for shirt size data"""
    
    def __init__(self, df: pd.DataFrame, size_to_idx: Dict, fit_to_idx: Dict):
        """
        Initialize dataset
        
        Args:
            df: DataFrame with measurements and labels
            size_to_idx: Mapping from size label to index
            fit_to_idx: Mapping from fit label to index
        """
        # Features
        self.features = df[[
            'shoulder_ratio', 
            'chest_ratio', 
            'waist_ratio', 
            'torso_proportion'
        ]].values.astype(np.float32)
        
        # Labels
        self.size_labels = df['size'].map(size_to_idx).values
        self.fit_labels = df['fit_type'].map(fit_to_idx).values
        
        # Normalize features
        self.mean = self.features.mean(axis=0)
        self.std = self.features.std(axis=0)
        self.features = (self.features - self.mean) / (self.std + 1e-8)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.size_labels[idx], dtype=torch.long),
            torch.tensor(self.fit_labels[idx], dtype=torch.long)
        )


class SizeClassifier(nn.Module):
    """Neural network for shirt size classification"""
    
    def __init__(
        self, 
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_sizes: int = 6,
        num_fits: int = 3,
        dropout: float = 0.3
    ):
        super(SizeClassifier, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout / 2)
        )
        
        # Size classification head
        self.size_head = nn.Linear(hidden_dim // 2, num_sizes)
        
        # Fit type classification head
        self.fit_head = nn.Linear(hidden_dim // 2, num_fits)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        size_logits = self.size_head(features)
        fit_logits = self.fit_head(features)
        return size_logits, fit_logits


class ModelTrainer:
    """Trainer class for the size prediction model"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        size_classes: list,
        fit_classes: list,
        save_dir: str = 'models'
    ):
        self.model = model.to(device)
        self.device = device
        self.size_classes = size_classes
        self.fit_classes = fit_classes
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_size_acc': [],
            'val_size_acc': [],
            'train_fit_acc': [],
            'val_fit_acc': []
        }
    
    def train_epoch(
        self, 
        train_loader: DataLoader,
        criterion_size: nn.Module,
        criterion_fit: nn.Module,
        optimizer: optim.Optimizer
    ) -> Tuple[float, float, float]:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0
        size_correct = 0
        fit_correct = 0
        total_samples = 0
        
        for features, size_labels, fit_labels in train_loader:
            features = features.to(self.device)
            size_labels = size_labels.to(self.device)
            fit_labels = fit_labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            size_logits, fit_logits = self.model(features)
            
            # Calculate losses
            loss_size = criterion_size(size_logits, size_labels)
            loss_fit = criterion_fit(fit_logits, fit_labels)
            loss = loss_size + loss_fit
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * features.size(0)
            size_correct += (size_logits.argmax(1) == size_labels).sum().item()
            fit_correct += (fit_logits.argmax(1) == fit_labels).sum().item()
            total_samples += features.size(0)
        
        avg_loss = total_loss / total_samples
        size_acc = size_correct / total_samples
        fit_acc = fit_correct / total_samples
        
        return avg_loss, size_acc, fit_acc
    
    def validate(
        self,
        val_loader: DataLoader,
        criterion_size: nn.Module,
        criterion_fit: nn.Module
    ) -> Tuple[float, float, float]:
        """Validate the model"""
        
        self.model.eval()
        total_loss = 0
        size_correct = 0
        fit_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for features, size_labels, fit_labels in val_loader:
                features = features.to(self.device)
                size_labels = size_labels.to(self.device)
                fit_labels = fit_labels.to(self.device)
                
                size_logits, fit_logits = self.model(features)
                
                loss_size = criterion_size(size_logits, size_labels)
                loss_fit = criterion_fit(fit_logits, fit_labels)
                loss = loss_size + loss_fit
                
                total_loss += loss.item() * features.size(0)
                size_correct += (size_logits.argmax(1) == size_labels).sum().item()
                fit_correct += (fit_logits.argmax(1) == fit_labels).sum().item()
                total_samples += features.size(0)
        
        avg_loss = total_loss / total_samples
        size_acc = size_correct / total_samples
        fit_acc = fit_correct / total_samples
        
        return avg_loss, size_acc, fit_acc
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        patience: int = 15
    ):
        """Train the model with early stopping"""
        
        criterion_size = nn.CrossEntropyLoss()
        criterion_fit = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Early stopping patience: {patience}\n")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_size_acc, train_fit_acc = self.train_epoch(
                train_loader, criterion_size, criterion_fit, optimizer
            )
            
            # Validate
            val_loss, val_size_acc, val_fit_acc = self.validate(
                val_loader, criterion_size, criterion_fit
            )
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_size_acc'].append(train_size_acc)
            self.history['val_size_acc'].append(val_size_acc)
            self.history['train_fit_acc'].append(train_fit_acc)
            self.history['val_fit_acc'].append(val_fit_acc)
            
            # Print progress
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Size Acc: {train_size_acc:.4f} | Fit Acc: {train_fit_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Size Acc: {val_size_acc:.4f} | Fit Acc: {val_fit_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint('best_model.pth')
                print("  ✓ New best model saved!")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            print()
        
        print("Training complete!")
        self.plot_training_history()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'size_classes': self.size_classes,
            'fit_classes': self.fit_classes,
            'history': self.history
        }
        
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Size Accuracy
        axes[1].plot(self.history['train_size_acc'], label='Train')
        axes[1].plot(self.history['val_size_acc'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Size Classification Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # Fit Accuracy
        axes[2].plot(self.history['train_fit_acc'], label='Train')
        axes[2].plot(self.history['val_fit_acc'], label='Validation')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Accuracy')
        axes[2].set_title('Fit Type Classification Accuracy')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_history.png', dpi=300)
        print(f"Training history plot saved to {self.save_dir / 'training_history.png'}")


def main():
    """Main training pipeline"""
    
    # Configuration
    DATA_DIR = 'data/synthetic_sizes'
    MODEL_DIR = 'models'
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    HIDDEN_DIM = 64
    
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
    
    print(f"Size classes: {size_classes}")
    print(f"Fit classes: {fit_classes}\n")
    
    # Load data
    print("Loading datasets...")
    train_df = pd.read_csv(Path(DATA_DIR) / 'train.csv')
    val_df = pd.read_csv(Path(DATA_DIR) / 'val.csv')
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}\n")
    
    # Create datasets
    train_dataset = SizeDataset(train_df, size_to_idx, fit_to_idx)
    val_dataset = SizeDataset(val_df, size_to_idx, fit_to_idx)
    
    # Save normalization stats
    norm_stats = {
        'mean': train_dataset.mean.tolist(),
        'std': train_dataset.std.tolist()
    }
    with open(Path(MODEL_DIR) / 'normalization_stats.json', 'w') as f:
        json.dump(norm_stats, f, indent=2)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=2
    )
    
    # Create model
    model = SizeClassifier(
        input_dim=4,
        hidden_dim=HIDDEN_DIM,
        num_sizes=len(size_classes),
        num_fits=len(fit_classes),
        dropout=0.3
    )
    
    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}\n")
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        device=device,
        size_classes=size_classes,
        fit_classes=fit_classes,
        save_dir=MODEL_DIR
    )
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        patience=15
    )
    
    print("\n✅ Training complete!")
    print(f"\nModel saved to: {MODEL_DIR}/best_model.pth")
    print("\nNext steps:")
    print("1. Run evaluate_model.py to test the model")
    print("2. Update your pipeline to use the trained model")


if __name__ == '__main__':
    main()
