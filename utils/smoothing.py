"""
Temporal smoothing and filtering for stable predictions
"""
import numpy as np
from collections import deque
from typing import Dict, Optional, List
from filterpy.kalman import KalmanFilter
import logging

logger = logging.getLogger(__name__)


class ExponentialMovingAverage:
    """EMA filter for smooth measurements"""
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize EMA filter
        
        Args:
            alpha: Smoothing factor (0-1). Lower = smoother, higher = more responsive
        """
        self.alpha = alpha
        self.value: Optional[float] = None
    
    def update(self, new_value: float) -> float:
        """Update with new value and return smoothed result"""
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        
        return self.value
    
    def reset(self):
        """Reset the filter"""
        self.value = None


class MeasurementSmoother:
    """Smooth body measurements over time"""
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize measurement smoother
        
        Args:
            alpha: EMA smoothing factor
        """
        self.filters = {
            'shoulder_ratio': ExponentialMovingAverage(alpha),
            'chest_ratio': ExponentialMovingAverage(alpha),
            'waist_ratio': ExponentialMovingAverage(alpha),
            'torso_proportion': ExponentialMovingAverage(alpha)
        }
    
    def smooth(self, measurements: Dict[str, float]) -> Dict[str, float]:
        """Apply smoothing to measurements"""
        smoothed = {}
        
        for key in ['shoulder_ratio', 'chest_ratio', 'waist_ratio', 'torso_proportion']:
            if key in measurements:
                smoothed[key] = self.filters[key].update(measurements[key])
            else:
                smoothed[key] = measurements.get(key, 0.0)
        
        # Pass through additional measurements unchanged
        for key, value in measurements.items():
            if key not in smoothed:
                smoothed[key] = value
        
        return smoothed
    
    def reset(self):
        """Reset all filters"""
        for filter in self.filters.values():
            filter.reset()


class PredictionStabilizer:
    """Stabilize size predictions using voting and confidence thresholds"""
    
    def __init__(
        self, 
        buffer_size: int = 10,
        min_confidence: float = 0.6,
        stability_threshold: float = 0.7
    ):
        """
        Initialize prediction stabilizer
        
        Args:
            buffer_size: Number of recent predictions to consider
            min_confidence: Minimum confidence to accept prediction
            stability_threshold: Fraction of buffer that must agree for stable prediction
        """
        self.buffer_size = buffer_size
        self.min_confidence = min_confidence
        self.stability_threshold = stability_threshold
        
        self.size_buffer = deque(maxlen=buffer_size)
        self.fit_buffer = deque(maxlen=buffer_size)
        self.confidence_buffer = deque(maxlen=buffer_size)
        
        self.current_size: Optional[str] = None
        self.current_fit: Optional[str] = None
        self.current_confidence: float = 0.0
    
    def update(
        self, 
        size: str, 
        fit_type: str, 
        confidence: float
    ) -> Dict[str, any]:
        """
        Update with new prediction and return stabilized result
        
        Returns:
            Dictionary with stabilized size, fit_type, confidence, and is_stable flag
        """
        # Add to buffers
        self.size_buffer.append(size)
        self.fit_buffer.append(fit_type)
        self.confidence_buffer.append(confidence)
        
        # Check if we have enough samples
        if len(self.size_buffer) < 3:
            return {
                'estimated_size': size,
                'fit_type': fit_type,
                'confidence': confidence,
                'is_stable': False
            }
        
        # Vote for most common size
        size_votes = {}
        for s in self.size_buffer:
            size_votes[s] = size_votes.get(s, 0) + 1
        
        most_common_size = max(size_votes, key=size_votes.get)
        size_agreement = size_votes[most_common_size] / len(self.size_buffer)
        
        # Vote for most common fit
        fit_votes = {}
        for f in self.fit_buffer:
            fit_votes[f] = fit_votes.get(f, 0) + 1
        
        most_common_fit = max(fit_votes, key=fit_votes.get)
        fit_agreement = fit_votes[most_common_fit] / len(self.fit_buffer)
        
        # Average confidence
        avg_confidence = np.mean(list(self.confidence_buffer))
        
        # Check stability
        is_stable = (
            size_agreement >= self.stability_threshold and
            fit_agreement >= self.stability_threshold and
            avg_confidence >= self.min_confidence
        )
        
        # Update current values if stable or if new prediction is much better
        if is_stable or confidence > avg_confidence + 0.2:
            self.current_size = most_common_size
            self.current_fit = most_common_fit
            self.current_confidence = avg_confidence
        
        return {
            'estimated_size': self.current_size or most_common_size,
            'fit_type': self.current_fit or most_common_fit,
            'confidence': self.current_confidence if self.current_size else avg_confidence,
            'is_stable': is_stable
        }
    
    def reset(self):
        """Reset the stabilizer"""
        self.size_buffer.clear()
        self.fit_buffer.clear()
        self.confidence_buffer.clear()
        self.current_size = None
        self.current_fit = None
        self.current_confidence = 0.0


class KalmanMeasurementFilter:
    """Kalman filter for measurement smoothing (advanced option)"""
    
    def __init__(self, measurement_noise: float = 0.01):
        """
        Initialize Kalman filter for 4D measurements
        
        Args:
            measurement_noise: Expected measurement noise variance
        """
        self.kf = KalmanFilter(dim_x=4, dim_z=4)
        
        # State transition matrix (identity - measurements don't change on their own)
        self.kf.F = np.eye(4)
        
        # Measurement matrix (identity - we measure what we track)
        self.kf.H = np.eye(4)
        
        # Process noise
        self.kf.Q = np.eye(4) * 0.001
        
        # Measurement noise
        self.kf.R = np.eye(4) * measurement_noise
        
        # Initial state covariance
        self.kf.P *= 100
        
        self.initialized = False
    
    def update(self, measurements: Dict[str, float]) -> Dict[str, float]:
        """
        Update Kalman filter with new measurements
        
        Args:
            measurements: Dict with shoulder_ratio, chest_ratio, waist_ratio, torso_proportion
        
        Returns:
            Filtered measurements
        """
        # Extract measurement vector
        z = np.array([
            measurements['shoulder_ratio'],
            measurements['chest_ratio'],
            measurements['waist_ratio'],
            measurements['torso_proportion']
        ])
        
        if not self.initialized:
            # Initialize state with first measurement
            self.kf.x = z
            self.initialized = True
            return measurements
        
        # Predict and update
        self.kf.predict()
        self.kf.update(z)
        
        # Extract filtered state
        filtered = {
            'shoulder_ratio': float(self.kf.x[0]),
            'chest_ratio': float(self.kf.x[1]),
            'waist_ratio': float(self.kf.x[2]),
            'torso_proportion': float(self.kf.x[3])
        }
        
        # Pass through other measurements
        for key, value in measurements.items():
            if key not in filtered:
                filtered[key] = value
        
        return filtered
    
    def reset(self):
        """Reset the filter"""
        self.kf.x = np.zeros(4)
        self.kf.P = np.eye(4) * 100
        self.initialized = False
