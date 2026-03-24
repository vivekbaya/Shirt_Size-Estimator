"""
Pose estimation and body measurement extraction using MediaPipe
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Dict, Tuple, List,Any

import logging

logger = logging.getLogger(__name__)


class PoseEstimator:
    """MediaPipe-based pose estimation and measurement extraction"""
    
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_segmentation=False
        )
        
        # MediaPipe landmark indices
        self.NOSE = 0
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
    
    def process_frame(self, frame: np.ndarray) -> Optional[Any]:
        """Process a single frame and extract pose landmarks"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                return results.pose_landmarks
            
            return None
        
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None
    
    def extract_measurements(
        self, 
        landmarks: Any,
        image_shape: Tuple[int, int]
    ) -> Optional[Dict[str, float]]:
        """
        Extract normalized body measurements from landmarks
        
        Returns ratios relative to image dimensions for calibration-free estimation
        """
        try:
            height, width = image_shape[:2]
            
            # Extract key landmark positions
            nose = landmarks.landmark[self.NOSE]
            left_shoulder = landmarks.landmark[self.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.RIGHT_SHOULDER]
            left_hip = landmarks.landmark[self.LEFT_HIP]
            right_hip = landmarks.landmark[self.RIGHT_HIP]
            
            # Check visibility of critical landmarks
            if (left_shoulder.visibility < 0.5 or 
                right_shoulder.visibility < 0.5 or
                left_hip.visibility < 0.5 or 
                right_hip.visibility < 0.5):
                return None
            
            # Convert normalized coordinates to pixels
            def to_pixel(landmark):
                return np.array([landmark.x * width, landmark.y * height])
            
            nose_px = to_pixel(nose)
            left_shoulder_px = to_pixel(left_shoulder)
            right_shoulder_px = to_pixel(right_shoulder)
            left_hip_px = to_pixel(left_hip)
            right_hip_px = to_pixel(right_hip)
            
            # Calculate measurements
            shoulder_width = np.linalg.norm(left_shoulder_px - right_shoulder_px)
            chest_width = shoulder_width  # Approximation at shoulder level
            waist_width = np.linalg.norm(left_hip_px - right_hip_px)
            
            # Torso length from nose to hip midpoint
            hip_midpoint = (left_hip_px + right_hip_px) / 2
            torso_length = np.linalg.norm(nose_px - hip_midpoint)
            
            # Normalize by image diagonal for scale invariance
            image_diagonal = np.sqrt(width**2 + height**2)
            
            shoulder_ratio = shoulder_width / image_diagonal
            chest_ratio = chest_width / image_diagonal
            waist_ratio = waist_width / image_diagonal
            
            # Torso proportion (ratio of torso length to shoulder width)
            torso_proportion = torso_length / shoulder_width if shoulder_width > 0 else 0
            
            measurements = {
                'shoulder_ratio': float(shoulder_ratio),
                'chest_ratio': float(chest_ratio),
                'waist_ratio': float(waist_ratio),
                'torso_proportion': float(torso_proportion),
                'shoulder_width_px': float(shoulder_width),
                'chest_width_px': float(chest_width),
                'waist_width_px': float(waist_width),
                'torso_length_px': float(torso_length)
            }
            
            return measurements
        
        except Exception as e:
            logger.error(f"Error extracting measurements: {e}")
            return None
    
    def calculate_pose_confidence(
        self, 
        landmarks: Any
    ) -> float:
        """Calculate overall pose detection confidence"""
        try:
            key_landmarks = [
                self.NOSE,
                self.LEFT_SHOULDER,
                self.RIGHT_SHOULDER,
                self.LEFT_HIP,
                self.RIGHT_HIP
            ]
            
            visibilities = [
                landmarks.landmark[idx].visibility 
                for idx in key_landmarks
            ]
            
            return float(np.mean(visibilities))
        
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    def draw_landmarks(
        self, 
        frame: np.ndarray, 
        landmarks: Any
    ) -> np.ndarray:
        """Draw pose landmarks on the frame"""
        try:
            annotated_frame = frame.copy()
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), 
                    thickness=2, 
                    circle_radius=3
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 255, 255), 
                    thickness=2
                )
            )
            return annotated_frame
        
        except Exception as e:
            logger.error(f"Error drawing landmarks: {e}")
            return frame
    
    def is_good_pose(self, landmarks: Any) -> bool:
        """
        Check if the pose is suitable for measurement
        (frontal, standing, minimal occlusion)
        """
        try:
            left_shoulder = landmarks.landmark[self.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.RIGHT_SHOULDER]
            
            # Check if person is roughly facing camera (shoulders should be horizontal)
            shoulder_y_diff = abs(left_shoulder.y - right_shoulder.y)
            
            # Check visibility
            min_visibility = min(left_shoulder.visibility, right_shoulder.visibility)
            
            # Good pose criteria
            is_frontal = shoulder_y_diff < 0.1  # Shoulders roughly level
            is_visible = min_visibility > 0.5
            
            return is_frontal and is_visible
        
        except Exception as e:
            logger.error(f"Error checking pose quality: {e}")
            return False
    
    def release(self):
        """Release MediaPipe resources"""
        if self.pose:
            self.pose.close()
