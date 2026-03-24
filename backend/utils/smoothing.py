"""
Smoothing and stabilization utilities for temporal consistency
"""
import numpy as np
from collections import deque
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class MeasurementSmoother:
    """
    Exponential Moving Average (EMA) smoother for body measurements
    
    Reduces jitter in measurements caused by:
    - Camera noise
    - Minor movements
    - Detection/pose estimation variations
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize smoother
        
        Args:
            alpha: Smoothing factor (0-1)
                  - 0 = no smoothing (all history)
                  - 1 = no smoothing (only current)
                  - 0.3 recommended for balance
        """
        self.alpha = alpha
        self.previous: Optional[Dict[str, float]] = None
    
    def smooth(self, measurements: Dict[str, float]) -> Dict[str, float]:
        """
        Apply exponential moving average to measurements
        
        Formula: smoothed = alpha * current + (1 - alpha) * previous
        
        Args:
            measurements: Current frame measurements
        
        Returns:
            Smoothed measurements
        """
        if self.previous is None:
            # First frame - no smoothing
            self.previous = measurements.copy()
            return measurements
        
        smoothed = {}
        
        for key in measurements:
            if key in self.previous:
                # Apply EMA
                smoothed[key] = (
                    self.alpha * measurements[key] + 
                    (1 - self.alpha) * self.previous[key]
                )
            else:
                # New measurement - no history
                smoothed[key] = measurements[key]
        
        # Update history
        self.previous = smoothed.copy()
        
        return smoothed
    
    def reset(self):
        """Reset smoother state"""
        self.previous = None


class PredictionStabilizer:
    """
    Stabilize size and fit predictions using a voting buffer
    
    Prevents prediction flickering by requiring consensus over multiple frames
    """
    
    def __init__(
        self, 
        buffer_size: int = 10,
        min_confidence: float = 0.6,
        stability_threshold: float = 0.7
    ):
        """
        Initialize stabilizer
        
        Args:
            buffer_size: Number of recent predictions to consider
            min_confidence: Minimum confidence to accept a prediction
            stability_threshold: Required agreement ratio (0-1) for stable prediction
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
    ) -> Dict[str, Any]:
        """
        Update buffer and compute stabilized prediction
        
        Args:
            size: Predicted size (XS, S, M, L, XL, XXL)
            fit_type: Predicted fit (slim, regular, relaxed)
            confidence: Prediction confidence (0-1)
        
        Returns:
            Dictionary with stabilized prediction and stability flag
        """
        # Add to buffers
        self.size_buffer.append(size)
        self.fit_buffer.append(fit_type)
        self.confidence_buffer.append(confidence)
        
        # Check if we have enough samples
        if len(self.size_buffer) < self.buffer_size * 0.5:
            # Not enough history - use current prediction
            self.current_size = size
            self.current_fit = fit_type
            self.current_confidence = confidence
            is_stable = False
        else:
            # Compute consensus
            size_consensus = self._get_consensus(self.size_buffer)
            fit_consensus = self._get_consensus(self.fit_buffer)
            avg_confidence = np.mean(list(self.confidence_buffer))
            
            # Check stability
            size_stable = size_consensus['ratio'] >= self.stability_threshold
            fit_stable = fit_consensus['ratio'] >= self.stability_threshold
            conf_stable = avg_confidence >= self.min_confidence
            
            is_stable = size_stable and fit_stable and conf_stable
            
            if is_stable:
                # Use consensus prediction
                self.current_size = size_consensus['value']
                self.current_fit = fit_consensus['value']
                self.current_confidence = avg_confidence
            else:
                # Not stable enough - use current but mark as unstable
                self.current_size = size
                self.current_fit = fit_type
                self.current_confidence = confidence
        
        return {
            'estimated_size': self.current_size,
            'fit_type': self.current_fit,
            'confidence': self.current_confidence,
            'is_stable': is_stable
        }
    
    def _get_consensus(self, buffer: deque) -> Dict[str, Any]:
        """
        Calculate consensus from buffer
        
        Returns most common value and its frequency ratio
        """
        if not buffer:
            return {'value': None, 'ratio': 0.0}
        
        # Count occurrences
        counts = {}
        for item in buffer:
            counts[item] = counts.get(item, 0) + 1
        
        # Find most common
        most_common = max(counts, key=counts.get)
        most_common_count = counts[most_common]
        
        # Calculate ratio
        ratio = most_common_count / len(buffer)
        
        return {
            'value': most_common,
            'ratio': ratio,
            'count': most_common_count
        }
    
    def reset(self):
        """Reset stabilizer state"""
        self.size_buffer.clear()
        self.fit_buffer.clear()
        self.confidence_buffer.clear()
        self.current_size = None
        self.current_fit = None
        self.current_confidence = 0.0


class KalmanFilter:
    """
    1D Kalman Filter for measurement smoothing (advanced alternative to EMA)
    
    More sophisticated than EMA - models both value and velocity
    Better for tracking smooth changes over time
    """
    
    def __init__(
        self, 
        process_variance: float = 0.01,
        measurement_variance: float = 0.1
    ):
        """
        Initialize Kalman filter
        
        Args:
            process_variance: How much we expect the value to change (smaller = smoother)
            measurement_variance: How noisy we expect measurements to be
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        
        # State
        self.x = None  # Estimated value
        self.p = 1.0   # Estimation error covariance
    
    def update(self, measurement: float) -> float:
        """
        Update filter with new measurement
        
        Args:
            measurement: New measured value
        
        Returns:
            Filtered estimate
        """
        if self.x is None:
            # Initialize
            self.x = measurement
            return measurement
        
        # Prediction step
        x_pred = self.x
        p_pred = self.p + self.process_variance
        
        # Update step
        K = p_pred / (p_pred + self.measurement_variance)  # Kalman gain
        self.x = x_pred + K * (measurement - x_pred)
        self.p = (1 - K) * p_pred
        
        return self.x
    
    def reset(self):
        """Reset filter state"""
        self.x = None
        self.p = 1.0


class MeasurementKalmanSmoother:
    """
    Multi-dimensional Kalman smoother for body measurements
    
    Alternative to MeasurementSmoother with better tracking
    """
    
    def __init__(
        self,
        process_variance: float = 0.01,
        measurement_variance: float = 0.1
    ):
        self.filters: Dict[str, KalmanFilter] = {}
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
    
    def smooth(self, measurements: Dict[str, float]) -> Dict[str, float]:
        """
        Apply Kalman filtering to all measurements
        
        Args:
            measurements: Current frame measurements
        
        Returns:
            Filtered measurements
        """
        smoothed = {}
        
        for key, value in measurements.items():
            # Create filter if needed
            if key not in self.filters:
                self.filters[key] = KalmanFilter(
                    self.process_variance,
                    self.measurement_variance
                )
            
            # Apply filter
            smoothed[key] = self.filters[key].update(value)
        
        return smoothed
    
    def reset(self):
        """Reset all filters"""
        self.filters.clear()


# Export main classes
__all__ = [
    'MeasurementSmoother',
    'PredictionStabilizer',
    'KalmanFilter',
    'MeasurementKalmanSmoother'
]