"""
MongoDB database models and connection management
"""
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class Measurements(BaseModel):
    """Measurement ratios for body proportions"""
    shoulder_ratio: float = Field(..., ge=0, le=1)
    chest_ratio: float = Field(..., ge=0, le=1)
    waist_ratio: float = Field(..., ge=0, le=1)
    torso_proportion: float = Field(..., ge=0, le=5)


class SizePrediction(BaseModel):
    """Size prediction document model"""
    session_id: str
    timestamp: datetime
    person_detected: bool
    confidence: float = Field(..., ge=0, le=1)
    estimated_size: Optional[str] = None
    fit_type: Optional[str] = None
    measurements: Optional[Measurements] = None
    reasoning_factors: List[str] = []
    frame_number: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_123",
                "timestamp": "2024-02-11T10:30:00Z",
                "person_detected": True,
                "confidence": 0.85,
                "estimated_size": "M",
                "fit_type": "regular",
                "measurements": {
                    "shoulder_ratio": 0.22,
                    "chest_ratio": 0.24,
                    "waist_ratio": 0.20,
                    "torso_proportion": 2.1
                },
                "reasoning_factors": ["shoulder_ratio", "chest_ratio"],
                "frame_number": 150
            }
        }


class DatabaseManager:
    """MongoDB async database manager"""
    
    def __init__(self, mongodb_url: str, db_name: str, collection_name: str):
        self.mongodb_url = mongodb_url
        self.db_name = db_name
        self.collection_name = collection_name
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.collection = None
    
    async def connect(self):
        """Establish database connection"""
        try:
            self.client = AsyncIOMotorClient(self.mongodb_url)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            
            # Create indexes
            await self.collection.create_index("session_id")
            await self.collection.create_index("timestamp")
            await self.collection.create_index([("session_id", 1), ("timestamp", -1)])
            
            logger.info(f"Connected to MongoDB: {self.db_name}.{self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def disconnect(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def insert_prediction(self, prediction: SizePrediction) -> str:
        """Insert a size prediction document"""
        try:
            prediction_dict = prediction.model_dump(mode='json')
            result = await self.collection.insert_one(prediction_dict)
            logger.debug(f"Inserted prediction: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to insert prediction: {e}")
            raise
    
    async def get_session_predictions(
        self, 
        session_id: str, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve predictions for a session"""
        try:
            cursor = self.collection.find(
                {"session_id": session_id}
            ).sort("timestamp", -1).limit(limit)
            
            predictions = await cursor.to_list(length=limit)
            return predictions
        except Exception as e:
            logger.error(f"Failed to retrieve predictions: {e}")
            raise
    
    async def get_latest_prediction(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent prediction for a session"""
        try:
            prediction = await self.collection.find_one(
                {"session_id": session_id},
                sort=[("timestamp", -1)]
            )
            return prediction
        except Exception as e:
            logger.error(f"Failed to retrieve latest prediction: {e}")
            raise
    
    async def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get aggregated statistics for a session"""
        try:
            pipeline = [
                {"$match": {"session_id": session_id, "person_detected": True}},
                {
                    "$group": {
                        "_id": "$estimated_size",
                        "count": {"$sum": 1},
                        "avg_confidence": {"$avg": "$confidence"}
                    }
                },
                {"$sort": {"count": -1}}
            ]
            
            results = await self.collection.aggregate(pipeline).to_list(None)
            
            total_predictions = sum(r['count'] for r in results)
            most_common_size = results[0]['_id'] if results else None
            
            return {
                "session_id": session_id,
                "total_predictions": total_predictions,
                "most_common_size": most_common_size,
                "size_distribution": results
            }
        except Exception as e:
            logger.error(f"Failed to retrieve session statistics: {e}")
            raise
    
    async def delete_session(self, session_id: str) -> int:
        """Delete all predictions for a session"""
        try:
            result = await self.collection.delete_many({"session_id": session_id})
            logger.info(f"Deleted {result.deleted_count} predictions for session {session_id}")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            raise


# Singleton instance
db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get the database manager instance"""
    global db_manager
    if db_manager is None:
        raise RuntimeError("Database manager not initialized")
    return db_manager


def init_db_manager(mongodb_url: str, db_name: str, collection_name: str):
    """Initialize the database manager"""
    global db_manager
    db_manager = DatabaseManager(mongodb_url, db_name, collection_name)
    return db_manager
