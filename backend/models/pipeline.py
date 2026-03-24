"""
Main video processing pipeline orchestrating all components
"""
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
import logging

from models.person_detector import PersonDetector
from models.pose_estimator import PoseEstimator
from models.size_predictor import SizePredictor
from utils.smoothing import MeasurementSmoother, PredictionStabilizer
from database.mongodb import SizePrediction, Measurements

logger = logging.getLogger(__name__)


class ShirtSizeEstimationPipeline:
    """Complete pipeline for real-time shirt size estimation"""
    
    def __init__(
        self,
        yolo_model_path: str = "yolov8n.pt",
        size_model_path: Optional[str] = None,
        yolo_confidence: float = 0.3,  # LOWERED for better detection
        mediapipe_confidence: float = 0.5,
        ema_alpha: float = 0.3,
        buffer_size: int = 10
    ):
        """
        Initialize the pipeline
        
        Args:
            yolo_model_path: Path to YOLO weights
            size_model_path: Path to trained size predictor (optional)
            yolo_confidence: YOLO detection confidence threshold
            mediapipe_confidence: MediaPipe pose confidence threshold
            ema_alpha: Exponential moving average factor
            buffer_size: Prediction buffer size for stabilization
        """
        # Initialize components
        self.person_detector = PersonDetector(
            model_path=yolo_model_path,
            confidence_threshold=yolo_confidence
        )
        
        self.pose_estimator = PoseEstimator(
            model_complexity=1,
            min_detection_confidence=mediapipe_confidence,
            min_tracking_confidence=mediapipe_confidence
        )
        
        self.size_predictor = SizePredictor(model_path=size_model_path)
        
        # Initialize smoothing and stabilization
        self.measurement_smoother = MeasurementSmoother(alpha=ema_alpha)
        self.prediction_stabilizer = PredictionStabilizer(
            buffer_size=buffer_size,
            min_confidence=0.6,
            stability_threshold=0.7
        )
        
        # Frame counter
        self.frame_count = 0
        
        logger.info("Shirt size estimation pipeline initialized")
        logger.info(f"YOLO confidence: {yolo_confidence}, MediaPipe confidence: {mediapipe_confidence}")
    
    def process_frame(
        self, 
        frame: np.ndarray,
        session_id: str,
        visualize: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single video frame
        
        Args:
            frame: Input BGR frame
            session_id: Current session identifier
            visualize: Whether to return annotated frame
        
        Returns:
            Dictionary containing prediction results and optional visualization
        """
        self.frame_count += 1
        
        # Initialize result
        result = {
            'person_detected': False,
            'confidence': 0.0,
            'estimated_size': None,
            'fit_type': None,
            'measurements': None,
            'reasoning_factors': [],
            'timestamp': datetime.utcnow().isoformat(),
            'frame_number': self.frame_count
        }
        
        try:
            # Validate frame
            if frame is None or frame.size == 0:
                logger.warning(f"Frame {self.frame_count}: Empty frame received")
                return result
            
            logger.debug(f"Frame {self.frame_count}: Processing frame of shape {frame.shape}")
            
            # Step 1: Detect persons
            detections = self.person_detector.detect_persons(frame)
            
            if not detections:
                logger.debug(f"Frame {self.frame_count}: No person detected")
                return result
            
            # Get best detection (largest person)
            best_detection = self.person_detector.get_best_detection(detections)
            
            if not best_detection:
                logger.debug(f"Frame {self.frame_count}: No valid detection selected")
                return result
            
            result['person_detected'] = True
            logger.info(f"Frame {self.frame_count}: Person detected with confidence {best_detection['confidence']:.3f}")
            
            # Step 2: Crop person region
            person_crop, adjusted_bbox = self.person_detector.crop_person(
                frame, 
                best_detection['bbox']
            )
            
            if person_crop.size == 0:
                logger.warning(f"Frame {self.frame_count}: Empty crop")
                return result
            
            # Step 3: Pose estimation on cropped region
            landmarks = self.pose_estimator.process_frame(person_crop)
            
            if landmarks is None:
                logger.debug(f"Frame {self.frame_count}: No pose detected")
                result['confidence'] = 0.3
                return result
            
            logger.debug(f"Frame {self.frame_count}: Pose landmarks detected")
            
            # Step 4: Check pose quality
            if not self.pose_estimator.is_good_pose(landmarks):
                logger.debug(f"Frame {self.frame_count}: Poor pose quality")
                result['confidence'] = 0.4
                return result
            
            # Step 5: Extract measurements
            measurements = self.pose_estimator.extract_measurements(
                landmarks, 
                person_crop.shape
            )
            
            if measurements is None:
                logger.debug(f"Frame {self.frame_count}: Failed to extract measurements")
                result['confidence'] = 0.4
                return result
            
            logger.debug(f"Frame {self.frame_count}: Measurements extracted: {measurements}")
            
            # Step 6: Smooth measurements
            smoothed_measurements = self.measurement_smoother.smooth(measurements)
            
            # Step 7: Predict size
            prediction = self.size_predictor.predict(smoothed_measurements)
            
            logger.info(f"Frame {self.frame_count}: Predicted size={prediction['estimated_size']}, fit={prediction['fit_type']}, conf={prediction['confidence']:.3f}")
            
            # Step 8: Stabilize prediction
            stabilized = self.prediction_stabilizer.update(
                prediction['estimated_size'],
                prediction['fit_type'],
                prediction['confidence']
            )
            
            # Step 9: Calculate pose confidence
            pose_confidence = self.pose_estimator.calculate_pose_confidence(landmarks)
            
            # Combine confidences
            combined_confidence = (
                best_detection['confidence'] * 0.3 +
                pose_confidence * 0.3 +
                stabilized['confidence'] * 0.4
            )
            
            # Update result
            result.update({
                'confidence': float(combined_confidence),
                'estimated_size': stabilized['estimated_size'],
                'fit_type': stabilized['fit_type'],
                'measurements': {
                    'shoulder_ratio': smoothed_measurements['shoulder_ratio'],
                    'chest_ratio': smoothed_measurements['chest_ratio'],
                    'waist_ratio': smoothed_measurements['waist_ratio'],
                    'torso_proportion': smoothed_measurements['torso_proportion']
                },
                'reasoning_factors': prediction['reasoning_factors'],
                'is_stable': stabilized.get('is_stable', False)
            })
            
            logger.info(
                f"Frame {self.frame_count}: FINAL RESULT - Size={result['estimated_size']}, "
                f"Fit={result['fit_type']}, Confidence={result['confidence']:.2f}, Stable={result['is_stable']}"
            )
            
            # Optional visualization
            if visualize:
                annotated_frame = self._create_visualization(
                    frame,
                    best_detection,
                    landmarks,
                    result,
                    person_crop,
                    adjusted_bbox
                )
                result['annotated_frame'] = annotated_frame
        
        except Exception as e:
            logger.error(f"Error processing frame {self.frame_count}: {e}", exc_info=True)
            result['confidence'] = 0.0
        
        return result
    
    def _create_visualization(
        self,
        frame: np.ndarray,
        detection: Dict,
        landmarks: Any,
        result: Dict,
        person_crop: np.ndarray,
        adjusted_bbox: list
    ) -> np.ndarray:
        """Create annotated visualization frame"""
        # Draw detection box
        annotated = self.person_detector.draw_detections(frame, [detection])
        
        # Draw pose on original frame (map from crop to full frame)
        x1, y1, x2, y2 = adjusted_bbox
        
        # Scale landmarks to full frame
        for landmark in landmarks.landmark:
            # Convert from crop coordinates to full frame
            px = int(x1 + landmark.x * (x2 - x1))
            py = int(y1 + landmark.y * (y2 - y1))
            
            cv2.circle(annotated, (px, py), 3, (0, 255, 0), -1)
        
        # Add text overlay with results
        if result['estimated_size']:
            text_lines = [
                f"Size: {result['estimated_size']}",
                f"Fit: {result['fit_type']}",
                f"Confidence: {result['confidence']:.2f}",
                f"Stable: {'Yes' if result.get('is_stable') else 'No'}"
            ]
            
            y_offset = 30
            for line in text_lines:
                cv2.putText(
                    annotated,
                    line,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                y_offset += 30
        
        return annotated
    
    def create_database_document(
        self, 
        result: Dict[str, Any], 
        session_id: str
    ) -> Optional[SizePrediction]:
        """
        Convert pipeline result to database document
        
        Args:
            result: Pipeline processing result
            session_id: Session identifier
        
        Returns:
            SizePrediction document ready for MongoDB insertion
        """
        try:
            if not result['person_detected']:
                return None
            
            measurements = None
            if result['measurements']:
                measurements = Measurements(**result['measurements'])
            
            prediction = SizePrediction(
                session_id=session_id,
                timestamp=datetime.fromisoformat(result['timestamp']),
                person_detected=result['person_detected'],
                confidence=result['confidence'],
                estimated_size=result['estimated_size'],
                fit_type=result['fit_type'],
                measurements=measurements,
                reasoning_factors=result['reasoning_factors'],
                frame_number=result.get('frame_number')
            )
            
            return prediction
        
        except Exception as e:
            logger.error(f"Error creating database document: {e}")
            return None
    
    def reset(self):
        """Reset the pipeline state"""
        self.measurement_smoother.reset()
        self.prediction_stabilizer.reset()
        self.frame_count = 0
        logger.info("Pipeline reset")
    
    def release(self):
        """Release all resources"""
        self.pose_estimator.release()
        logger.info("Pipeline resources released")