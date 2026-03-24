# Shirt Size Estimation System - Technical Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Details](#architecture-details)
3. [Algorithm Pipeline](#algorithm-pipeline)
4. [API Reference](#api-reference)
5. [Database Schema](#database-schema)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)

## System Overview

### Purpose
Real-time estimation of shirt sizes from live video input without requiring physical measurements or calibration objects.

### Key Technologies
- **Computer Vision**: YOLO v8, MediaPipe Pose
- **Backend**: FastAPI, Python 3.9+
- **Frontend**: React 18, WebSocket
- **Database**: MongoDB 4.4+
- **ML Framework**: PyTorch

### System Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: Optional (CUDA-compatible for faster inference)
- **Camera**: 640x480 minimum resolution
- **Network**: WebSocket support

## Architecture Details

### Component Breakdown

#### 1. Person Detection Layer (YOLO v8)
**Purpose**: Locate and isolate persons in video frames

**Model**: YOLOv8n (nano variant)
- Input: 640x640 RGB image
- Output: Bounding boxes [x1, y1, x2, y2], confidence scores
- Class filter: Person class only (COCO class 0)

**Configuration**:
```python
confidence_threshold = 0.5  # Minimum detection confidence
iou_threshold = 0.45        # Non-maximum suppression
```

**Output**: List of detections, sorted by area (largest first)

#### 2. Pose Estimation Layer (MediaPipe)
**Purpose**: Extract body landmarks for measurement calculation

**Model**: MediaPipe Pose
- Landmarks: 33 points including shoulders, hips, chest
- Tracking: Maintains identity across frames
- Visibility: Per-landmark confidence scores

**Key Landmarks Used**:
```python
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
```

#### 3. Measurement Extraction
**Purpose**: Convert landmarks to normalized body ratios

**Calculated Metrics**:

1. **Shoulder Ratio**
   ```
   shoulder_width = distance(left_shoulder, right_shoulder)
   image_diagonal = sqrt(width² + height²)
   shoulder_ratio = shoulder_width / image_diagonal
   ```

2. **Chest Ratio**
   ```
   chest_width ≈ shoulder_width (approximation at shoulder level)
   chest_ratio = chest_width / image_diagonal
   ```

3. **Waist Ratio**
   ```
   waist_width = distance(left_hip, right_hip)
   waist_ratio = waist_width / image_diagonal
   ```

4. **Torso Proportion**
   ```
   torso_length = distance(nose, hip_midpoint)
   torso_proportion = torso_length / shoulder_width
   ```

**Normalization**: All ratios are scale-invariant (independent of camera distance)

#### 4. Temporal Smoothing
**Purpose**: Reduce frame-to-frame jitter

**Methods**:

1. **Exponential Moving Average (EMA)**
   ```python
   smoothed_value = α × new_value + (1 - α) × previous_value
   α = 0.3  # Smoothing factor
   ```

2. **Kalman Filter (Optional)**
   - State: [shoulder, chest, waist, torso]
   - Process noise: Q = 0.001 × I
   - Measurement noise: R = 0.01 × I

#### 5. Size Prediction
**Purpose**: Classify size and fit type

**Rule-Based Classification**:

Size thresholds (shoulder_ratio):
```python
XS:  0.00 - 0.18
S:   0.18 - 0.21
M:   0.21 - 0.24
L:   0.24 - 0.27
XL:  0.27 - 0.30
XXL: 0.30 - 1.00
```

Fit type (waist_to_chest ratio):
```python
slim:    ratio < 0.85
regular: 0.85 ≤ ratio < 0.95
relaxed: ratio ≥ 0.95
```

**Neural Network (Optional)**:
- Architecture: 4 → 32 → (6, 3)
- Multi-task: Size classification + Fit classification
- Loss: CrossEntropy for each task
- Optimizer: Adam with learning rate 0.001

#### 6. Prediction Stabilization
**Purpose**: Ensure consistent outputs

**Voting Mechanism**:
- Buffer size: 10 frames
- Stability threshold: 70% agreement
- Confidence threshold: 60% minimum

**Update Logic**:
```python
if agreement >= 0.7 and confidence >= 0.6:
    output = most_common_prediction
else:
    output = current_stable_prediction
```

## Algorithm Pipeline

### Frame Processing Flow

```python
def process_frame(frame):
    # Step 1: Person Detection
    detections = yolo.detect(frame)
    if not detections:
        return NO_PERSON_RESULT
    
    best_person = select_largest(detections)
    person_crop = crop_with_padding(frame, best_person.bbox)
    
    # Step 2: Pose Estimation
    landmarks = mediapipe.estimate_pose(person_crop)
    if not landmarks or not is_good_pose(landmarks):
        return LOW_CONFIDENCE_RESULT
    
    # Step 3: Measurement Extraction
    raw_measurements = extract_measurements(landmarks, person_crop.shape)
    
    # Step 4: Temporal Smoothing
    smoothed = ema_filter.update(raw_measurements)
    
    # Step 5: Size Prediction
    size, fit, confidence = predictor.predict(smoothed)
    
    # Step 6: Stabilization
    stable_result = stabilizer.update(size, fit, confidence)
    
    # Step 7: Database Storage
    save_to_mongodb(stable_result)
    
    return stable_result
```

### Timing Breakdown (Typical CPU)

| Stage | Time (ms) | % Total |
|-------|-----------|---------|
| YOLO Detection | 30-40 | 40% |
| Pose Estimation | 25-35 | 35% |
| Measurement | 1-2 | 2% |
| Smoothing | <1 | 1% |
| Prediction | 1-2 | 2% |
| Stabilization | <1 | 1% |
| Visualization | 10-15 | 15% |
| **Total** | **70-100** | **100%** |

## API Reference

### REST Endpoints

#### Create Session
```http
POST /session/create
```

**Response**:
```json
{
  "session_id": "uuid-v4-string",
  "message": "Session created successfully"
}
```

#### Get Predictions
```http
GET /session/{session_id}/predictions?limit=100
```

**Response**:
```json
{
  "session_id": "uuid",
  "count": 50,
  "predictions": [
    {
      "_id": "mongodb-object-id",
      "session_id": "uuid",
      "timestamp": "2024-02-11T10:30:00Z",
      "person_detected": true,
      "confidence": 0.85,
      "estimated_size": "M",
      "fit_type": "regular",
      "measurements": {
        "shoulder_ratio": 0.22,
        "chest_ratio": 0.24,
        "waist_ratio": 0.20,
        "torso_proportion": 2.1
      },
      "reasoning_factors": ["shoulder_ratio", "chest_ratio"]
    }
  ]
}
```

#### Get Statistics
```http
GET /session/{session_id}/statistics
```

**Response**:
```json
{
  "session_id": "uuid",
  "total_predictions": 150,
  "most_common_size": "M",
  "size_distribution": [
    {
      "_id": "M",
      "count": 80,
      "avg_confidence": 0.87
    },
    {
      "_id": "L",
      "count": 50,
      "avg_confidence": 0.82
    }
  ]
}
```

### WebSocket Protocol

#### Connection
```
ws://localhost:8000/ws/estimate/{session_id}
```

#### Client → Server Message
```json
{
  "type": "frame",
  "data": "base64-encoded-jpeg",
  "visualize": true
}
```

#### Server → Client Response
```json
{
  "type": "prediction",
  "person_detected": true,
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
  "timestamp": "2024-02-11T10:30:00.123Z",
  "annotated_frame": "base64-encoded-jpeg"  // if visualize=true
}
```

## Database Schema

### MongoDB Collection: `size_predictions`

```javascript
{
  _id: ObjectId,
  session_id: String,          // UUID v4
  timestamp: ISODate,           // UTC timestamp
  person_detected: Boolean,
  confidence: Double,           // 0.0 - 1.0
  estimated_size: String,       // "XS"|"S"|"M"|"L"|"XL"|"XXL"
  fit_type: String,            // "slim"|"regular"|"relaxed"
  measurements: {
    shoulder_ratio: Double,
    chest_ratio: Double,
    waist_ratio: Double,
    torso_proportion: Double
  },
  reasoning_factors: [String], // Array of measurement keys
  frame_number: Int32          // Sequential frame number
}
```

### Indexes

```javascript
db.size_predictions.createIndex({ session_id: 1 })
db.size_predictions.createIndex({ timestamp: -1 })
db.size_predictions.createIndex({ session_id: 1, timestamp: -1 })
```

## Performance Optimization

### Techniques Implemented

1. **Frame Skipping**
   ```python
   PROCESS_EVERY_N_FRAMES = 1  # Process all frames
   # For lower-end hardware, set to 2 or 3
   ```

2. **Resolution Limiting**
   ```python
   MAX_FRAME_WIDTH = 640
   MAX_FRAME_HEIGHT = 480
   # Reduces computation while maintaining accuracy
   ```

3. **Model Selection**
   - YOLOv8n (nano): Fastest variant
   - MediaPipe complexity=1: Balanced speed/accuracy

4. **Async Processing**
   - WebSocket for non-blocking I/O
   - Database operations are async

5. **Caching**
   - YOLO model loaded once per session
   - MediaPipe maintains tracking state

### GPU Acceleration

Enable CUDA for PyTorch:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

Expected speedup: 3-5x on consumer GPUs

## Troubleshooting

### Common Issues

#### 1. Low Accuracy
**Symptoms**: Wrong size predictions, low confidence

**Solutions**:
- Improve lighting (even, front-facing)
- Ensure frontal pose (face camera directly)
- Stand at recommended distance (6-8 feet)
- Wear fitted clothing
- Clear background

#### 2. Jittery Predictions
**Symptoms**: Size jumps between frames

**Solutions**:
- Increase EMA alpha (more smoothing)
- Increase buffer size for stabilization
- Reduce frame rate

#### 3. High Latency
**Symptoms**: Delay between camera and predictions

**Solutions**:
- Reduce frame size
- Enable GPU acceleration
- Skip frames (process every 2nd or 3rd)
- Use YOLOv8n (fastest model)

#### 4. Database Connection Failed
**Symptoms**: Cannot connect to MongoDB

**Solutions**:
```bash
# Check MongoDB status
sudo systemctl status mongodb

# Restart MongoDB
sudo systemctl restart mongodb

# Check connection string
echo $MONGODB_URL
```

#### 5. Camera Not Detected
**Symptoms**: Black screen, no video

**Solutions**:
```bash
# Linux: List cameras
ls -l /dev/video*

# Check permissions
sudo usermod -a -G video $USER

# Test with OpenCV
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### Debug Mode

Enable verbose logging:
```python
# config/settings.py
LOG_LEVEL = "DEBUG"
```

View detailed logs:
```bash
tail -f app.log
```

## Best Practices

### For Developers

1. **Error Handling**: Always wrap CV operations in try-except
2. **Resource Management**: Use context managers for cameras
3. **Testing**: Write unit tests for each component
4. **Monitoring**: Log performance metrics
5. **Validation**: Validate all user inputs

### For End Users

1. **Setup**: Good lighting, clear background
2. **Position**: 6-8 feet from camera, centered
3. **Pose**: Stand straight, face camera, arms at sides
4. **Clothing**: Fitted clothing for accurate measurements
5. **Stability**: Hold pose for 2-3 seconds

---

**Last Updated**: February 2024  
**Version**: 1.0.0
