"""
Shirt size prediction model using PyTorch - UPDATED VERSION

This version properly loads trained models and handles normalization.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


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
        """
        Initialize the size classifier
        
        Args:
            input_dim: Number of input features (shoulder, chest, waist, torso ratios)
            hidden_dim: Hidden layer dimension
            num_sizes: Number of size classes (XS, S, M, L, XL, XXL)
            num_fits: Number of fit types (slim, regular, relaxed)
        """
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
        
        self.size_classes = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
        self.fit_classes = ['slim', 'regular', 'relaxed']
    
    def forward(self, x):
        """Forward pass"""
        features = self.feature_extractor(x)
        size_logits = self.size_head(features)
        fit_logits = self.fit_head(features)
        return size_logits, fit_logits


class SizePredictor:
    """Wrapper for size prediction with rule-based fallback"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize size predictor
        
        Args:
            model_path: Path to trained model weights (optional)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.use_neural_model = False
        self.normalization_stats = None
        
        # Try to load trained model
        if model_path and Path(model_path).exists():
            try:
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Initialize model with same architecture as training
                self.model = SizeClassifier(
                    input_dim=4,
                    hidden_dim=64,
                    num_sizes=6,
                    num_fits=3,
                    dropout=0.3
                ).to(self.device)
                
                # Load weights
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                # Load normalization statistics
                norm_stats_path = Path(model_path).parent / 'normalization_stats.json'
                if norm_stats_path.exists():
                    with open(norm_stats_path, 'r') as f:
                        self.normalization_stats = json.load(f)
                    logger.info(f"Loaded normalization stats from {norm_stats_path}")
                else:
                    logger.warning("Normalization stats not found, using default normalization")
                    self.normalization_stats = {
                        'mean': [0.25, 0.25, 0.25, 1.5],
                        'std': [0.1, 0.1, 0.1, 0.3]
                    }
                
                self.use_neural_model = True
                logger.info(f"✅ Loaded trained model from {model_path}")
                logger.info(f"   Device: {self.device}")
                logger.info(f"   Size classes: {self.model.size_classes}")
                logger.info(f"   Fit classes: {self.model.fit_classes}")
                
            except Exception as e:
                logger.error(f"Failed to load trained model: {e}")
                logger.info("Falling back to rule-based prediction")
                self.use_neural_model = False
        else:
            if model_path:
                logger.warning(f"Model path {model_path} does not exist")
            logger.info("Using rule-based size prediction")
            self.use_neural_model = False
        
        # Rule-based thresholds (fallback)
        self.size_thresholds = {
            'shoulder_ratio': {
                'XS': (0.0, 0.18),
                'S': (0.18, 0.21),
                'M': (0.21, 0.24),
                'L': (0.24, 0.27),
                'XL': (0.27, 0.30),
                'XXL': (0.30, 1.0)
            },
            'chest_ratio': {
                'XS': (0.0, 0.20),
                'S': (0.20, 0.23),
                'M': (0.23, 0.26),
                'L': (0.26, 0.29),
                'XL': (0.29, 0.32),
                'XXL': (0.32, 1.0)
            }
        }
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using training statistics"""
        if self.normalization_stats:
            mean = np.array(self.normalization_stats['mean'])
            std = np.array(self.normalization_stats['std'])
            return (features - mean) / (std + 1e-8)
        else:
            # Default normalization
            return (features - 0.25) / 0.1
    
    def predict_neural(
        self, 
        measurements: Dict[str, float]
    ) -> Tuple[str, str, float]:
        """
        Predict using neural network
        
        Returns:
            (size, fit_type, confidence)
        """
        try:
            # Prepare input
            features = np.array([
                measurements['shoulder_ratio'],
                measurements['chest_ratio'],
                measurements['waist_ratio'],
                measurements['torso_proportion']
            ], dtype=np.float32)
            
            # Normalize features
            features = self._normalize_features(features)
            
            # Convert to tensor
            x = torch.tensor(features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                size_logits, fit_logits = self.model(x)
                
                # Get predictions
                size_probs = torch.softmax(size_logits, dim=1)
                fit_probs = torch.softmax(fit_logits, dim=1)
                
                size_idx = torch.argmax(size_probs, dim=1).item()
                fit_idx = torch.argmax(fit_probs, dim=1).item()
                
                size_conf = size_probs[0, size_idx].item()
                fit_conf = fit_probs[0, fit_idx].item()
                
                overall_confidence = (size_conf + fit_conf) / 2
                
                size = self.model.size_classes[size_idx]
                fit_type = self.model.fit_classes[fit_idx]
            
            logger.debug(f"Neural prediction: size={size} (conf={size_conf:.3f}), fit={fit_type} (conf={fit_conf:.3f})")
            
            return size, fit_type, overall_confidence
        
        except Exception as e:
            logger.error(f"Neural prediction failed: {e}")
            return self.predict_rule_based(measurements)
    
    def predict_rule_based(
        self, 
        measurements: Dict[str, float]
    ) -> Tuple[str, str, float]:
        """
        Predict using rule-based system
        
        Returns:
            (size, fit_type, confidence)
        """
        shoulder_ratio = measurements['shoulder_ratio']
        chest_ratio = measurements['chest_ratio']
        waist_ratio = measurements['waist_ratio']
        torso_proportion = measurements['torso_proportion']
        
        # Predict size based on shoulder and chest ratios
        size_votes = []
        
        # Shoulder-based size
        for size, (low, high) in self.size_thresholds['shoulder_ratio'].items():
            if low <= shoulder_ratio < high:
                size_votes.append(size)
                break
        
        # Chest-based size
        for size, (low, high) in self.size_thresholds['chest_ratio'].items():
            if low <= chest_ratio < high:
                size_votes.append(size)
                break
        
        # Default to most common size if no match
        if not size_votes:
            estimated_size = 'M'
            confidence = 0.5
        else:
            # Use majority vote or average
            from collections import Counter
            size_counts = Counter(size_votes)
            estimated_size = size_counts.most_common(1)[0][0]
            confidence = 0.7 + (len(size_votes) * 0.1)  # Higher if multiple agree
        
        # Predict fit type based on waist-to-chest ratio
        waist_to_chest = waist_ratio / chest_ratio if chest_ratio > 0 else 1.0
        
        if waist_to_chest < 0.85:
            fit_type = 'slim'
            fit_confidence = 0.8
        elif waist_to_chest < 0.95:
            fit_type = 'regular'
            fit_confidence = 0.85
        else:
            fit_type = 'relaxed'
            fit_confidence = 0.8
        
        overall_confidence = min((confidence + fit_confidence) / 2, 1.0)
        
        logger.debug(f"Rule-based prediction: size={estimated_size}, fit={fit_type}, conf={overall_confidence:.3f}")
        
        return estimated_size, fit_type, overall_confidence
    
    def predict(
        self, 
        measurements: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Main prediction method
        
        Returns:
            Dictionary with size, fit_type, confidence, and reasoning
        """
        if self.use_neural_model:
            size, fit_type, confidence = self.predict_neural(measurements)
        else:
            size, fit_type, confidence = self.predict_rule_based(measurements)
        
        # Determine reasoning factors
        reasoning_factors = []
        
        if measurements['shoulder_ratio'] > 0.15:
            reasoning_factors.append('shoulder_ratio')
        
        if measurements['chest_ratio'] > 0.15:
            reasoning_factors.append('chest_ratio')
        
        if abs(measurements['waist_ratio'] - measurements['chest_ratio']) > 0.05:
            reasoning_factors.append('waist_ratio')
        
        if measurements['torso_proportion'] > 1.5:
            reasoning_factors.append('torso_proportion')
        
        if not reasoning_factors:
            reasoning_factors = ['shoulder_ratio', 'chest_ratio']
        
        return {
            'estimated_size': size,
            'fit_type': fit_type,
            'confidence': confidence,
            'reasoning_factors': reasoning_factors,
            'prediction_method': 'neural' if self.use_neural_model else 'rule-based'
        }
