"""
Application Settings Configuration
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

class Settings:
    """Application configuration settings"""
    
    # API Settings
    API_TITLE = "Shirt Size Estimation API"
    API_VERSION = "1.0.0"
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # MongoDB Settings
    MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
    MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "shirt_size_db")
    MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "predictions")
    
    # YOLO Model Settings
    YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
    YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE", "0.3"))
    
    # Trained Size Prediction Model
    SIZE_MODEL_PATH = str(BASE_DIR / "trained_models" / "best_model.pth")
    USE_TRAINED_MODEL = os.path.exists(SIZE_MODEL_PATH)
    
    # MediaPipe Settings
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE = float(os.getenv("MEDIAPIPE_DETECTION_CONF", "0.5"))
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE = float(os.getenv("MEDIAPIPE_TRACKING_CONF", "0.5"))
    
    # Processing Settings
    MAX_FRAME_WIDTH = int(os.getenv("MAX_FRAME_WIDTH", "1280"))
    MAX_FRAME_HEIGHT = int(os.getenv("MAX_FRAME_HEIGHT", "720"))
    FRAME_BUFFER_SIZE = int(os.getenv("FRAME_BUFFER_SIZE", "10"))
    
    # Smoothing Settings
    EMA_ALPHA = float(os.getenv("EMA_ALPHA", "0.3"))  # Exponential moving average factor
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print("APPLICATION CONFIGURATION")
        print("=" * 60)
        print(f"API Host: {cls.API_HOST}:{cls.API_PORT}")
        print(f"MongoDB URL: {cls.MONGODB_URL}")
        print(f"MongoDB Database: {cls.MONGODB_DB_NAME}")
        print(f"YOLO Model: {cls.YOLO_MODEL_PATH}")
        print(f"Trained Model: {cls.SIZE_MODEL_PATH}")
        print(f"Use Trained Model: {cls.USE_TRAINED_MODEL}")
        print(f"Max Frame Size: {cls.MAX_FRAME_WIDTH}x{cls.MAX_FRAME_HEIGHT}")
        print("=" * 60)


# Create settings instance
settings = Settings()