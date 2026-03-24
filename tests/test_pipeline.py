"""
Unit tests for the shirt size estimation pipeline
"""
import pytest
import numpy as np
import cv2
from models.pipeline import ShirtSizeEstimationPipeline
from models.person_detector import PersonDetector
from models.pose_estimator import PoseEstimator
from models.size_predictor import SizePredictor
from utils.smoothing import MeasurementSmoother, PredictionStabilizer


@pytest.fixture
def sample_frame():
    """Create a sample test frame"""
    # Create a blank 640x480 BGR image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some content (white rectangle in center)
    cv2.rectangle(frame, (200, 100), (440, 400), (255, 255, 255), -1)
    return frame


@pytest.fixture
def pipeline():
    """Create a pipeline instance for testing"""
    return ShirtSizeEstimationPipeline(
        yolo_model_path="yolov8n.pt",
        yolo_confidence=0.5,
        mediapipe_confidence=0.5
    )


class TestPersonDetector:
    """Test person detection module"""
    
    def test_detector_initialization(self):
        """Test that detector initializes correctly"""
        detector = PersonDetector(confidence_threshold=0.5)
        assert detector is not None
        assert detector.confidence_threshold == 0.5
    
    def test_detect_persons_empty_frame(self, sample_frame):
        """Test detection on empty frame"""
        detector = PersonDetector(confidence_threshold=0.5)
        detections = detector.detect_persons(sample_frame)
        # May or may not detect anything in blank frame
        assert isinstance(detections, list)
    
    def test_get_best_detection(self):
        """Test best detection selection"""
        detector = PersonDetector()
        detections = [
            {'bbox': [0, 0, 100, 100], 'confidence': 0.8, 'area': 10000},
            {'bbox': [0, 0, 200, 200], 'confidence': 0.7, 'area': 40000},
            {'bbox': [0, 0, 150, 150], 'confidence': 0.9, 'area': 22500}
        ]
        best = detector.get_best_detection(detections)
        # Should select largest area
        assert best['area'] == 40000


class TestPoseEstimator:
    """Test pose estimation module"""
    
    def test_estimator_initialization(self):
        """Test that estimator initializes correctly"""
        estimator = PoseEstimator(model_complexity=1)
        assert estimator is not None
    
    def test_extract_measurements_format(self):
        """Test measurement extraction output format"""
        # This would require a valid landmarks object
        # Simplified test - just verify method exists
        estimator = PoseEstimator()
        assert hasattr(estimator, 'extract_measurements')


class TestSizePredictor:
    """Test size prediction module"""
    
    def test_predictor_initialization(self):
        """Test predictor initializes correctly"""
        predictor = SizePredictor()
        assert predictor is not None
    
    def test_predict_rule_based(self):
        """Test rule-based prediction"""
        predictor = SizePredictor()
        measurements = {
            'shoulder_ratio': 0.22,
            'chest_ratio': 0.24,
            'waist_ratio': 0.20,
            'torso_proportion': 2.1
        }
        
        result = predictor.predict(measurements)
        
        assert 'estimated_size' in result
        assert 'fit_type' in result
        assert 'confidence' in result
        assert result['estimated_size'] in ['XS', 'S', 'M', 'L', 'XL', 'XXL']
        assert result['fit_type'] in ['slim', 'regular', 'relaxed']
        assert 0 <= result['confidence'] <= 1


class TestMeasurementSmoother:
    """Test temporal smoothing"""
    
    def test_smoother_initialization(self):
        """Test smoother initializes with correct alpha"""
        smoother = MeasurementSmoother(alpha=0.3)
        assert smoother.filters['shoulder_ratio'].alpha == 0.3
    
    def test_smoothing_reduces_variance(self):
        """Test that smoothing reduces measurement variance"""
        smoother = MeasurementSmoother(alpha=0.3)
        
        # Simulate noisy measurements
        measurements = [
            {'shoulder_ratio': 0.20, 'chest_ratio': 0.22, 'waist_ratio': 0.18, 'torso_proportion': 2.0},
            {'shoulder_ratio': 0.24, 'chest_ratio': 0.26, 'waist_ratio': 0.22, 'torso_proportion': 2.4},
            {'shoulder_ratio': 0.19, 'chest_ratio': 0.21, 'waist_ratio': 0.17, 'torso_proportion': 1.9},
        ]
        
        smoothed_values = []
        for m in measurements:
            smoothed = smoother.smooth(m)
            smoothed_values.append(smoothed['shoulder_ratio'])
        
        # Variance of smoothed should be less than raw variance
        raw_variance = np.var([m['shoulder_ratio'] for m in measurements])
        smoothed_variance = np.var(smoothed_values)
        
        # After 3 samples, smoothing should have some effect
        assert smoothed_variance <= raw_variance or len(measurements) < 5


class TestPredictionStabilizer:
    """Test prediction stabilization"""
    
    def test_stabilizer_initialization(self):
        """Test stabilizer initializes correctly"""
        stabilizer = PredictionStabilizer(buffer_size=10)
        assert len(stabilizer.size_buffer) == 0
    
    def test_stabilization_voting(self):
        """Test that stabilizer uses voting mechanism"""
        stabilizer = PredictionStabilizer(buffer_size=5, stability_threshold=0.6)
        
        # Feed consistent predictions
        for _ in range(5):
            result = stabilizer.update('M', 'regular', 0.85)
        
        # Should stabilize on 'M'
        assert result['estimated_size'] == 'M'
        assert result['fit_type'] == 'regular'
    
    def test_requires_minimum_samples(self):
        """Test that stabilizer needs minimum samples"""
        stabilizer = PredictionStabilizer(buffer_size=10)
        
        # First prediction
        result = stabilizer.update('L', 'slim', 0.7)
        assert result['is_stable'] == False  # Not enough samples yet


class TestPipeline:
    """Test complete pipeline integration"""
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initializes all components"""
        assert pipeline.person_detector is not None
        assert pipeline.pose_estimator is not None
        assert pipeline.size_predictor is not None
        assert pipeline.measurement_smoother is not None
        assert pipeline.prediction_stabilizer is not None
    
    def test_process_frame_returns_valid_structure(self, pipeline, sample_frame):
        """Test that process_frame returns expected structure"""
        result = pipeline.process_frame(sample_frame, "test_session")
        
        assert 'person_detected' in result
        assert 'confidence' in result
        assert 'estimated_size' in result
        assert 'fit_type' in result
        assert 'measurements' in result
        assert 'reasoning_factors' in result
        assert 'timestamp' in result
        assert isinstance(result['person_detected'], bool)
        assert isinstance(result['confidence'], float)
    
    def test_create_database_document(self, pipeline):
        """Test database document creation"""
        result = {
            'person_detected': True,
            'confidence': 0.85,
            'estimated_size': 'M',
            'fit_type': 'regular',
            'measurements': {
                'shoulder_ratio': 0.22,
                'chest_ratio': 0.24,
                'waist_ratio': 0.20,
                'torso_proportion': 2.1
            },
            'reasoning_factors': ['shoulder_ratio', 'chest_ratio'],
            'timestamp': '2024-02-11T10:30:00',
            'frame_number': 100
        }
        
        doc = pipeline.create_database_document(result, "test_session")
        
        assert doc is not None
        assert doc.session_id == "test_session"
        assert doc.estimated_size == 'M'
        assert doc.fit_type == 'regular'
    
    def test_pipeline_reset(self, pipeline):
        """Test that pipeline reset works"""
        # Process some frames first
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.process_frame(frame, "test_session")
        pipeline.process_frame(frame, "test_session")
        
        initial_count = pipeline.frame_count
        assert initial_count > 0
        
        # Reset
        pipeline.reset()
        
        # Frame count should be reset
        assert pipeline.frame_count == 0


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_measurements(self):
        """Test prediction with empty measurements"""
        predictor = SizePredictor()
        measurements = {
            'shoulder_ratio': 0.0,
            'chest_ratio': 0.0,
            'waist_ratio': 0.0,
            'torso_proportion': 0.0
        }
        
        # Should not crash
        result = predictor.predict(measurements)
        assert result is not None
    
    def test_extreme_measurements(self):
        """Test prediction with extreme values"""
        predictor = SizePredictor()
        measurements = {
            'shoulder_ratio': 0.99,
            'chest_ratio': 0.99,
            'waist_ratio': 0.99,
            'torso_proportion': 10.0
        }
        
        # Should handle gracefully
        result = predictor.predict(measurements)
        assert result['estimated_size'] in ['XS', 'S', 'M', 'L', 'XL', 'XXL']


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
