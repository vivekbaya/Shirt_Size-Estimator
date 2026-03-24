"""
FastAPI backend server for shirt size estimation
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import cv2
import numpy as np
import base64
import json
import logging
from typing import Dict, List
import uuid

from config.settings import settings
from database.mongodb import init_db_manager, get_db_manager
from models.pipeline import ShirtSizeEstimationPipeline

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for troubleshooting
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global pipeline instances (one per WebSocket connection)
active_pipelines: Dict[str, ShirtSizeEstimationPipeline] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    logger.info("Starting up application...")
    
    # Initialize database
    db_manager = init_db_manager(
        settings.MONGODB_URL,
        settings.MONGODB_DB_NAME,
        settings.MONGODB_COLLECTION
    )
    await db_manager.connect()
    
    logger.info("Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    await db_manager.disconnect()
    
    # Release all pipelines
    for pipeline in active_pipelines.values():
        pipeline.release()
    
    logger.info("Application shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Shirt Size Estimation API",
        "version": settings.API_VERSION,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db = get_db_manager()
        await db.collection.find_one({})
        
        return {
            "status": "healthy",
            "database": "connected",
            "active_connections": len(active_pipelines)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.post("/session/create")
async def create_session():
    """Create a new estimation session"""
    session_id = str(uuid.uuid4())
    
    logger.info(f"Created new session: {session_id}")
    
    return {
        "session_id": session_id,
        "message": "Session created successfully"
    }


@app.get("/session/{session_id}/predictions")
async def get_session_predictions(session_id: str, limit: int = 100):
    """Retrieve predictions for a session"""
    try:
        db = get_db_manager()
        predictions = await db.get_session_predictions(session_id, limit)
        
        return {
            "session_id": session_id,
            "count": len(predictions),
            "predictions": predictions
        }
    except Exception as e:
        logger.error(f"Error retrieving predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/statistics")
async def get_session_statistics(session_id: str):
    """Get aggregated statistics for a session"""
    try:
        db = get_db_manager()
        stats = await db.get_session_statistics(session_id)
        
        return stats
    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its predictions"""
    try:
        db = get_db_manager()
        deleted_count = await db.delete_session(session_id)
        
        return {
            "session_id": session_id,
            "deleted_count": deleted_count,
            "message": "Session deleted successfully"
        }
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/estimate/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time size estimation
    
    Expected message format:
    {
        "type": "frame",
        "data": "base64_encoded_image",
        "visualize": true/false
    }
    
    Response format:
    {
        "type": "prediction",
        "person_detected": true/false,
        "confidence": 0.0-1.0,
        "estimated_size": "XS|S|M|L|XL|XXL",
        "fit_type": "slim|regular|relaxed",
        "measurements": {...},
        "reasoning_factors": [...],
        "timestamp": "ISO-8601",
        "annotated_frame": "base64_encoded_image" (optional)
    }
    """
    await websocket.accept()
    logger.info(f"WebSocket connection established: {session_id}")
    
    # Create pipeline for this connection
    pipeline = ShirtSizeEstimationPipeline(
        yolo_model_path=settings.YOLO_MODEL_PATH,
        size_model_path=settings.SIZE_MODEL_PATH if settings.USE_TRAINED_MODEL else None,  # ADD THIS
        yolo_confidence=0.3,  # LOWERED for better detection
        mediapipe_confidence=settings.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        ema_alpha=settings.EMA_ALPHA,
        buffer_size=settings.FRAME_BUFFER_SIZE
    )
    
    active_pipelines[session_id] = pipeline
    db = get_db_manager()
    
    frame_count = 0
    
    try:
        while True:
            # Receive message
            message = await websocket.receive_json()
            
            if message.get('type') == 'frame':
                frame_count += 1
                
                try:
                    # Decode base64 frame
                    if 'data' not in message:
                        logger.error("Frame message missing 'data' field")
                        await websocket.send_json({
                            'type': 'error',
                            'message': 'Frame data missing'
                        })
                        continue
                    
                    frame_data = base64.b64decode(message['data'])
                    nparr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        logger.error(f"Frame {frame_count}: Failed to decode frame")
                        await websocket.send_json({
                            'type': 'error',
                            'message': 'Failed to decode frame'
                        })
                        continue
                    
                    logger.debug(f"Frame {frame_count}: Decoded successfully, shape={frame.shape}")
                    
                    # Resize frame if necessary
                    h, w = frame.shape[:2]
                    if w > settings.MAX_FRAME_WIDTH or h > settings.MAX_FRAME_HEIGHT:
                        scale = min(
                            settings.MAX_FRAME_WIDTH / w,
                            settings.MAX_FRAME_HEIGHT / h
                        )
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        frame = cv2.resize(frame, (new_w, new_h))
                        logger.debug(f"Frame {frame_count}: Resized to {frame.shape}")
                    
                    # Process frame
                    visualize = message.get('visualize', False)
                    result = pipeline.process_frame(frame, session_id, visualize)
                    
                    logger.debug(f"Frame {frame_count}: Processing result - person_detected={result['person_detected']}, confidence={result['confidence']:.3f}")
                    
                    # Store in database if person detected
                    if result['person_detected']:
                        doc = pipeline.create_database_document(result, session_id)
                        if doc:
                            await db.insert_prediction(doc)
                            logger.debug(f"Frame {frame_count}: Stored prediction in database")
                    
                    # Prepare response
                    response = {
                        'type': 'prediction',
                        'person_detected': result['person_detected'],
                        'confidence': result['confidence'],
                        'estimated_size': result['estimated_size'],
                        'fit_type': result['fit_type'],
                        'measurements': result['measurements'],
                        'reasoning_factors': result['reasoning_factors'],
                        'timestamp': result['timestamp']
                    }
                    
                    # Add annotated frame if requested
                    if visualize and 'annotated_frame' in result:
                        _, buffer = cv2.imencode('.jpg', result['annotated_frame'])
                        annotated_b64 = base64.b64encode(buffer).decode('utf-8')
                        response['annotated_frame'] = annotated_b64
                    
                    # Send response
                    await websocket.send_json(response)
                    
                    if frame_count % 30 == 0:  # Log every 30 frames
                        logger.info(f"Processed {frame_count} frames for session {session_id}")
                
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {e}", exc_info=True)
                    await websocket.send_json({
                        'type': 'error',
                        'message': f'Processing error: {str(e)}'
                    })
            
            elif message.get('type') == 'reset':
                # Reset pipeline state
                pipeline.reset()
                logger.info(f"Pipeline reset for session {session_id}")
                await websocket.send_json({
                    'type': 'info',
                    'message': 'Pipeline reset successfully'
                })
            
            else:
                logger.warning(f"Unknown message type: {message.get('type')}")
                await websocket.send_json({
                    'type': 'error',
                    'message': f'Unknown message type: {message.get("type")}'
                })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    
    finally:
        # Cleanup
        if session_id in active_pipelines:
            active_pipelines[session_id].release()
            del active_pipelines[session_id]
        
        logger.info(f"WebSocket connection closed: {session_id}. Processed {frame_count} total frames.")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="debug"  # Changed to debug
    )