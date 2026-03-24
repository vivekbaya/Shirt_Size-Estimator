"""
YOLO-based person detection for the shirt size estimation system
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional, List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class PersonDetector:
    """YOLOv8-based person detection"""
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.3,  # LOWERED from 0.5 for better detection
        iou_threshold: float = 0.45
    ):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IOU threshold for NMS
        """
        try:
            self.model = YOLO(model_path)
            self.confidence_threshold = confidence_threshold
            self.iou_threshold = iou_threshold
            self.person_class_id = 0  # COCO dataset person class
            logger.info(f"Loaded YOLO model from {model_path}")
            logger.info(f"Detection confidence threshold: {confidence_threshold}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect_persons(
        self, 
        frame: np.ndarray
    ) -> List[Dict[str, any]]:
        """
        Detect persons in the frame
        
        Returns:
            List of detection dictionaries containing:
            - bbox: [x1, y1, x2, y2]
            - confidence: detection confidence
            - area: bounding box area
        """
        try:
            # Check if frame is valid
            if frame is None or frame.size == 0:
                logger.warning("Received empty frame")
                return []
            
            # Log frame info for debugging
            logger.debug(f"Processing frame: shape={frame.shape}, dtype={frame.dtype}")
            
            # Convert BGR to RGB for YOLO
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(
                rgb_frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=[self.person_class_id],
                verbose=False
            )
            
            detections = []
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                logger.debug(f"YOLO detected {len(boxes)} person(s)")
                
                for box in boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Calculate area
                    area = (x2 - x1) * (y2 - y1)
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'area': float(area)
                    }
                    
                    detections.append(detection)
                    logger.debug(f"Detection: bbox={detection['bbox']}, conf={confidence:.3f}, area={area:.0f}")
            else:
                logger.debug("No person detected by YOLO")
            
            return detections
        
        except Exception as e:
            logger.error(f"Error during person detection: {e}", exc_info=True)
            return []
    
    def get_best_detection(
        self, 
        detections: List[Dict[str, any]]
    ) -> Optional[Dict[str, any]]:
        """
        Get the best detection (largest person in frame)
        
        Prioritizes central, large detections
        """
        if not detections:
            return None
        
        # Sort by area (largest first)
        sorted_detections = sorted(
            detections, 
            key=lambda d: d['area'], 
            reverse=True
        )
        
        best = sorted_detections[0]
        logger.debug(f"Selected best detection: area={best['area']:.0f}, conf={best['confidence']:.3f}")
        
        return best
    
    def crop_person(
        self, 
        frame: np.ndarray, 
        bbox: List[int],
        padding: float = 0.1
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Crop the person from the frame with padding
        
        Args:
            frame: Input frame
            bbox: [x1, y1, x2, y2]
            padding: Relative padding to add around bbox
        
        Returns:
            Cropped frame and adjusted bbox
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Calculate padding
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        pad_w = int(bbox_w * padding)
        pad_h = int(bbox_h * padding)
        
        # Apply padding with bounds checking
        x1_padded = max(0, x1 - pad_w)
        y1_padded = max(0, y1 - pad_h)
        x2_padded = min(w, x2 + pad_w)
        y2_padded = min(h, y2 + pad_h)
        
        # Crop
        cropped = frame[y1_padded:y2_padded, x1_padded:x2_padded]
        
        logger.debug(f"Cropped person: original_bbox={bbox}, padded_bbox=[{x1_padded},{y1_padded},{x2_padded},{y2_padded}], crop_shape={cropped.shape}")
        
        return cropped, [x1_padded, y1_padded, x2_padded, y2_padded]
    
    def draw_detections(
        self, 
        frame: np.ndarray, 
        detections: List[Dict[str, any]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """Draw bounding boxes on the frame"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Draw rectangle
            cv2.rectangle(
                annotated_frame, 
                (x1, y1), 
                (x2, y2), 
                color, 
                thickness
            )
            
            # Draw confidence label
            label = f"Person: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(
                label, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                1
            )
            
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        return annotated_frame