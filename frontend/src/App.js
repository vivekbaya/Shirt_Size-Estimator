import React, { useState, useRef, useEffect, useCallback } from "react";
import "./App.css";

const API_URL = "http://localhost:8000";
const WS_URL = "ws://localhost:8000";

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [showVisualization, setShowVisualization] = useState(true);
  const [statistics, setStatistics] = useState(null);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const animationRef = useRef(null);
  const isStreamingRef = useRef(false); // Track streaming state for RAF callback

  // Create session on mount
  useEffect(() => {
    createSession();
    return () => {
      stopStreaming();
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const createSession = async () => {
    try {
      const response = await fetch(`${API_URL}/session/create`, {
        method: "POST",
      });
      const data = await response.json();
      setSessionId(data.session_id);
      console.log("Session created:", data.session_id);
    } catch (error) {
      console.error("Error creating session:", error);
    }
  };

  const connectWebSocket = useCallback(() => {
    if (!sessionId) return;

    const ws = new WebSocket(`${WS_URL}/ws/estimate/${sessionId}`);

    ws.onopen = () => {
      console.log("WebSocket connected");
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("Received message:", data.type);

      if (data.type === "prediction") {
        setPrediction(data);

        // Display annotated frame if available
        if (data.annotated_frame && canvasRef.current) {
          const img = new Image();
          img.onload = () => {
            const ctx = canvasRef.current.getContext("2d");
            canvasRef.current.width = img.width;
            canvasRef.current.height = img.height;
            ctx.drawImage(img, 0, 0);
          };
          img.src = `data:image/jpeg;base64,${data.annotated_frame}`;
        }
      } else if (data.type === "error") {
        console.error("Server error:", data.message);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      setIsConnected(false);
    };

    ws.onclose = () => {
      console.log("WebSocket disconnected");
      setIsConnected(false);
    };

    wsRef.current = ws;
  }, [sessionId]);

  const startStreaming = async () => {
    try {
      // Get camera stream
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user",
        },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;

        videoRef.current.onloadeddata = () => {
          console.log("VIDEO READY – STARTING WEBSOCKET");
          
          // Connect WebSocket first
          connectWebSocket();
          
          // Set streaming state
          setIsStreaming(true);
          isStreamingRef.current = true;
          
          // Wait for WebSocket to be ready before starting frame capture
          setTimeout(() => {
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
              console.log("Starting frame capture loop");
              captureAndSendFrame();
            } else {
              console.warn("WebSocket not ready, retrying...");
              const checkWs = setInterval(() => {
                if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                  clearInterval(checkWs);
                  console.log("WebSocket ready, starting frame capture");
                  captureAndSendFrame();
                }
              }, 100);
            }
          }, 500);
        };
      }
    } catch (error) {
      console.error("Error accessing camera:", error);
      alert("Failed to access camera. Please grant camera permissions.");
    }
  };

  const stopStreaming = () => {
    setIsStreaming(false);
    isStreamingRef.current = false;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  };

  const captureAndSendFrame = () => {
    // Use ref instead of state to avoid stale closure
    if (!isStreamingRef.current) {
      console.log("Streaming stopped, ending frame capture");
      return;
    }

    try {
      if (
        videoRef.current &&
        wsRef.current &&
        wsRef.current.readyState === WebSocket.OPEN &&
        videoRef.current.readyState === videoRef.current.HAVE_ENOUGH_DATA
      ) {
        const canvas = document.createElement("canvas");
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;

        const ctx = canvas.getContext("2d");
        ctx.drawImage(videoRef.current, 0, 0);

        canvas.toBlob(
          (blob) => {
            if (!blob) {
              console.error("Failed to create blob from canvas");
              return;
            }

            const reader = new FileReader();
            reader.onloadend = () => {
              const base64data = reader.result.split(",")[1];

              console.log("Sending frame, size:", base64data.length);

              wsRef.current.send(
                JSON.stringify({
                  type: "frame",
                  data: base64data,
                  visualize: showVisualization,
                })
              );
            };
            reader.readAsDataURL(blob);
          },
          "image/jpeg",
          0.8
        );
      } else {
        console.log("Waiting for video/WebSocket to be ready...", {
          hasVideo: !!videoRef.current,
          hasWs: !!wsRef.current,
          wsState: wsRef.current?.readyState,
          videoState: videoRef.current?.readyState
        });
      }
    } catch (error) {
      console.error("Error capturing frame:", error);
    }

    // Schedule next frame
    animationRef.current = requestAnimationFrame(captureAndSendFrame);
  };

  const fetchStatistics = async () => {
    if (!sessionId) return;

    try {
      const response = await fetch(
        `${API_URL}/session/${sessionId}/statistics`,
      );
      const data = await response.json();
      setStatistics(data);
    } catch (error) {
      console.error("Error fetching statistics:", error);
    }
  };

  const resetPipeline = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "reset" }));
      setPrediction(null);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>👕 Real-Time Shirt Size Estimator</h1>
        <p className="subtitle">AI-Powered Computer Vision System</p>
      </header>

      <div className="container">
        <div className="video-section">
          <div className="video-container">
            {/* Always show camera video */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="video-display"
            />

            {/* Canvas overlay (only visible when visualization ON) */}
            <canvas
              ref={canvasRef}
              className="video-display overlay-canvas"
              style={{ display: showVisualization ? "block" : "none" }}
            />

            {!isStreaming && (
              <div className="overlay">
                <p>Click "Start Camera" to begin</p>
              </div>
            )}
          </div>

          <div className="controls">
            {!isStreaming ? (
              <button onClick={startStreaming} className="btn btn-primary">
                🔹 Start Camera
              </button>
            ) : (
              <button onClick={stopStreaming} className="btn btn-danger">
                ⏹️ Stop Camera
              </button>
            )}

            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={showVisualization}
                onChange={(e) => setShowVisualization(e.target.checked)}
              />
              Show Visualization
            </label>

            <button onClick={resetPipeline} className="btn btn-secondary">
              🔄 Reset
            </button>

            <button onClick={fetchStatistics} className="btn btn-secondary">
              📊 Statistics
            </button>
          </div>

          <div className="status-bar">
            <span
              className={`status-indicator ${isConnected ? "connected" : "disconnected"}`}
            >
              {isConnected ? "🟢 Connected" : "🔴 Disconnected"}
            </span>
            <span className="session-id">
              Session: {sessionId?.substring(0, 8)}...
            </span>
          </div>
        </div>

        <div className="results-section">
          <h2>Estimation Results</h2>

          {prediction && prediction.person_detected ? (
            <div className="prediction-card">
              <div className="prediction-main">
                <div className="size-display">
                  <span className="label">Size:</span>
                  <span className="value size-value">
                    {prediction.estimated_size}
                  </span>
                </div>
                <div className="fit-display">
                  <span className="label">Fit:</span>
                  <span className="value">{prediction.fit_type}</span>
                </div>
              </div>

              <div className="confidence-bar">
                <span className="label">Confidence:</span>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${prediction.confidence * 100}%` }}
                  />
                </div>
                <span className="confidence-value">
                  {(prediction.confidence * 100).toFixed(1)}%
                </span>
              </div>

              {prediction.measurements && (
                <div className="measurements">
                  <h3>Body Measurements (Normalized)</h3>
                  <div className="measurement-grid">
                    <div className="measurement-item">
                      <span className="measurement-label">Shoulder Ratio:</span>
                      <span className="measurement-value">
                        {prediction.measurements.shoulder_ratio.toFixed(3)}
                      </span>
                    </div>
                    <div className="measurement-item">
                      <span className="measurement-label">Chest Ratio:</span>
                      <span className="measurement-value">
                        {prediction.measurements.chest_ratio.toFixed(3)}
                      </span>
                    </div>
                    <div className="measurement-item">
                      <span className="measurement-label">Waist Ratio:</span>
                      <span className="measurement-value">
                        {prediction.measurements.waist_ratio.toFixed(3)}
                      </span>
                    </div>
                    <div className="measurement-item">
                      <span className="measurement-label">
                        Torso Proportion:
                      </span>
                      <span className="measurement-value">
                        {prediction.measurements.torso_proportion.toFixed(2)}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {prediction.reasoning_factors &&
                prediction.reasoning_factors.length > 0 && (
                  <div className="reasoning">
                    <h3>Key Factors:</h3>
                    <div className="tags">
                      {prediction.reasoning_factors.map((factor, idx) => (
                        <span key={idx} className="tag">
                          {factor}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

              <div className="timestamp">
                Last updated:{" "}
                {new Date(prediction.timestamp).toLocaleTimeString()}
              </div>
            </div>
          ) : (
            <div className="no-detection">
              <p>👤 No person detected</p>
              <p className="hint">
                Stand in front of the camera in a frontal pose for best results
              </p>
            </div>
          )}

          {statistics && (
            <div className="statistics-card">
              <h3>Session Statistics</h3>
              <p>Total Predictions: {statistics.total_predictions}</p>
              <p>Most Common Size: {statistics.most_common_size}</p>

              {statistics.size_distribution && (
                <div className="size-distribution">
                  <h4>Size Distribution:</h4>
                  {statistics.size_distribution.map((item, idx) => (
                    <div key={idx} className="distribution-item">
                      <span>{item._id}: </span>
                      <span>
                        {item.count} ({(item.avg_confidence * 100).toFixed(1)}%
                        avg confidence)
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <footer className="App-footer">
        <p>Built with React, FastAPI, MediaPipe, YOLO & MongoDB</p>
      </footer>
    </div>
  );
}

export default App;