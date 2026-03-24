"""
Example usage script for the shirt size estimation system
"""
import cv2
import asyncio
from models.pipeline import ShirtSizeEstimationPipeline
from database.mongodb import init_db_manager
from config.settings import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_webcam_estimation():
    """
    Run real-time estimation from webcam
    Simple standalone example without WebSocket
    """
    
    # Initialize database
    db_manager = init_db_manager(
        settings.MONGODB_URL,
        settings.MONGODB_DB_NAME,
        settings.MONGODB_COLLECTION
    )
    await db_manager.connect()
    
    # Create pipeline
    pipeline = ShirtSizeEstimationPipeline(
        yolo_model_path=settings.YOLO_MODEL_PATH,
        yolo_confidence=settings.YOLO_CONFIDENCE,
        mediapipe_confidence=settings.MEDIAPIPE_MIN_DETECTION_CONFIDENCE
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        logger.error("Failed to open webcam")
        return
    
    logger.info("Webcam opened successfully")
    logger.info("Press 'q' to quit, 'r' to reset, 's' to save prediction")
    
    session_id = "webcam_session_001"
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                logger.error("Failed to read frame")
                break
            
            # Process frame with visualization
            result = pipeline.process_frame(
                frame, 
                session_id=session_id,
                visualize=True
            )
            
            # Display annotated frame
            if 'annotated_frame' in result:
                display_frame = result['annotated_frame']
            else:
                display_frame = frame
            
            # Add additional info overlay
            if result['person_detected']:
                info_text = [
                    f"Size: {result['estimated_size']} | Fit: {result['fit_type']}",
                    f"Confidence: {result['confidence']:.2%}",
                    f"Stable: {'Yes' if result.get('is_stable') else 'No'}"
                ]
                
                y_offset = 30
                for text in info_text:
                    cv2.putText(
                        display_frame,
                        text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                    y_offset += 25
            else:
                cv2.putText(
                    display_frame,
                    "No person detected - stand in front of camera",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
            
            # Show instructions
            cv2.putText(
                display_frame,
                "Q: Quit | R: Reset | S: Save",
                (10, display_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            cv2.imshow('Shirt Size Estimation', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("Quit requested")
                break
            elif key == ord('r'):
                logger.info("Resetting pipeline")
                pipeline.reset()
            elif key == ord('s'):
                if result['person_detected']:
                    doc = pipeline.create_database_document(result, session_id)
                    if doc:
                        doc_id = await db_manager.insert_prediction(doc)
                        logger.info(f"Saved prediction: {doc_id}")
                else:
                    logger.warning("Cannot save - no person detected")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        pipeline.release()
        await db_manager.disconnect()
        logger.info("Cleanup complete")


async def test_static_image(image_path: str):
    """
    Test estimation on a static image
    
    Args:
        image_path: Path to test image
    """
    
    # Create pipeline
    pipeline = ShirtSizeEstimationPipeline()
    
    # Load image
    frame = cv2.imread(image_path)
    
    if frame is None:
        logger.error(f"Failed to load image: {image_path}")
        return
    
    logger.info(f"Processing image: {image_path}")
    
    # Process
    result = pipeline.process_frame(
        frame,
        session_id="static_test",
        visualize=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("ESTIMATION RESULTS")
    print("="*60)
    print(f"Person Detected: {result['person_detected']}")
    
    if result['person_detected']:
        print(f"Estimated Size: {result['estimated_size']}")
        print(f"Fit Type: {result['fit_type']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        if result['measurements']:
            print("\nMeasurements:")
            for key, value in result['measurements'].items():
                print(f"  {key}: {value:.4f}")
        
        print(f"\nReasoning Factors: {', '.join(result['reasoning_factors'])}")
        
        # Show annotated image
        if 'annotated_frame' in result:
            cv2.imshow('Result', result['annotated_frame'])
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("No person detected in image")
    
    print("="*60 + "\n")
    
    pipeline.release()


def print_system_info():
    """Print system configuration"""
    print("\n" + "="*60)
    print("SHIRT SIZE ESTIMATION SYSTEM")
    print("="*60)
    print(f"YOLO Model: {settings.YOLO_MODEL_PATH}")
    print(f"YOLO Confidence: {settings.YOLO_CONFIDENCE}")
    print(f"MediaPipe Confidence: {settings.MEDIAPIPE_MIN_DETECTION_CONFIDENCE}")
    print(f"EMA Alpha: {settings.EMA_ALPHA}")
    print(f"Frame Buffer Size: {settings.FRAME_BUFFER_SIZE}")
    print(f"MongoDB URL: {settings.MONGODB_URL}")
    print(f"MongoDB Database: {settings.MONGODB_DB_NAME}")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    
    print_system_info()
    
    if len(sys.argv) > 1:
        # Test on static image
        image_path = sys.argv[1]
        asyncio.run(test_static_image(image_path))
    else:
        # Run webcam estimation
        print("Starting webcam estimation...")
        print("Make sure MongoDB is running!")
        asyncio.run(run_webcam_estimation())
