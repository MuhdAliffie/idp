#!/usr/bin/env python3
"""
Live Web Stream Viewer for Raspberry Pi Camera
Streams processed camera feed to web browser
Avoids Qt/display issues by using HTTP MJPEG stream
"""

import cv2
import numpy as np
import time
import logging
import subprocess
import threading
import tempfile
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict
import math
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
from urllib.parse import urlparse, parse_qs

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLOv8 available")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLOv8 not available - obstacle detection disabled")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ZoneType(Enum):
    """Zone classification types"""
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"

@dataclass
class GridCell:
    """Represents a single cell in the 8x8 grid"""
    row: int
    col: int
    x1: int
    y1: int
    x2: int
    y2: int
    zone_type: ZoneType
    has_obstacle: bool = False
    obstacle_confidence: float = 0.0
    obstacle_distance: float = float('inf')

@dataclass
class ObstacleInfo:
    """Obstacle information from detection"""
    distance: float
    direction: str  # 'front', 'left', 'right'
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    grid_cells: List[Tuple[int, int]] = field(default_factory=list)

class GridZoneManager:
    """Manages the 8x8 grid overlay and zone classification"""
    
    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.grid_rows = 8
        self.grid_cols = 8
        self.cell_width = frame_width // self.grid_cols
        self.cell_height = frame_height // self.grid_rows
        
        # Initialize grid
        self.grid = self._initialize_grid()
        
        # Define zone mapping
        self.zone_map = self._create_zone_map()
        
    def _initialize_grid(self) -> List[List[GridCell]]:
        """Initialize the 8x8 grid with cells"""
        grid = []
        for row in range(self.grid_rows):
            grid_row = []
            for col in range(self.grid_cols):
                x1 = col * self.cell_width
                y1 = row * self.cell_height
                x2 = x1 + self.cell_width
                y2 = y1 + self.cell_height
                
                cell = GridCell(
                    row=row,
                    col=col,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    zone_type=ZoneType.SAFE
                )
                grid_row.append(cell)
            grid.append(grid_row)
        return grid
    
    def _create_zone_map(self) -> Dict[Tuple[int, int], ZoneType]:
        """Create the zone mapping for the grid"""
        zone_map = {}
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Critical zone - center of view (direct path)
                if 3 <= row <= 4 and 3 <= col <= 4:
                    zone_map[(row, col)] = ZoneType.CRITICAL
                # Danger zone - around center
                elif 2 <= row <= 5 and 2 <= col <= 5:
                    zone_map[(row, col)] = ZoneType.DANGER
                # Warning zone - sides
                elif 1 <= row <= 6 and 1 <= col <= 6:
                    zone_map[(row, col)] = ZoneType.WARNING
                # Safe zone - edges
                else:
                    zone_map[(row, col)] = ZoneType.SAFE
                    
        return zone_map
    
    def update_zone_types(self):
        """Update zone types for all cells based on zone map"""
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                self.grid[row][col].zone_type = self.zone_map.get((row, col), ZoneType.SAFE)
    
    def reset_obstacles(self):
        """Reset obstacle detection in all cells"""
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                self.grid[row][col].has_obstacle = False
                self.grid[row][col].obstacle_confidence = 0.0
                self.grid[row][col].obstacle_distance = float('inf')
    
    def get_cells_for_bbox(self, bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        """Get grid cells that overlap with a bounding box"""
        x1, y1, x2, y2 = bbox
        cells = []
        
        start_col = max(0, x1 // self.cell_width)
        end_col = min(self.grid_cols - 1, x2 // self.cell_width)
        start_row = max(0, y1 // self.cell_height)
        end_row = min(self.grid_rows - 1, y2 // self.cell_height)
        
        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):
                cells.append((row, col))
                
        return cells
    
    def update_obstacle_in_cells(self, cells: List[Tuple[int, int]], confidence: float, distance: float):
        """Update obstacle information in specified cells"""
        for row, col in cells:
            if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                self.grid[row][col].has_obstacle = True
                self.grid[row][col].obstacle_confidence = max(
                    self.grid[row][col].obstacle_confidence, 
                    confidence
                )
                self.grid[row][col].obstacle_distance = min(
                    self.grid[row][col].obstacle_distance,
                    distance
                )
    
    def get_danger_cells(self) -> List[GridCell]:
        """Get all cells with obstacles in danger or critical zones"""
        danger_cells = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                cell = self.grid[row][col]
                if cell.has_obstacle and cell.zone_type in [ZoneType.DANGER, ZoneType.CRITICAL]:
                    danger_cells.append(cell)
        return danger_cells
    
    def draw_grid_overlay(self, frame: np.ndarray, show_zones: bool = True) -> np.ndarray:
        """Draw grid overlay on frame with zone coloring"""
        overlay = frame.copy()
        
        # Define colors for different zone types
        zone_colors = {
            ZoneType.SAFE: (0, 255, 0),      # Green
            ZoneType.WARNING: (0, 255, 255),  # Yellow
            ZoneType.DANGER: (0, 165, 255),   # Orange
            ZoneType.CRITICAL: (0, 0, 255)    # Red
        }
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                cell = self.grid[row][col]
                
                # Draw cell background if it has an obstacle or if showing zones
                if show_zones or cell.has_obstacle:
                    color = zone_colors[cell.zone_type]
                    alpha = 0.3 if not cell.has_obstacle else 0.6
                    
                    # Create semi-transparent overlay
                    cv2.rectangle(overlay, (cell.x1, cell.y1), (cell.x2, cell.y2), color, -1)
                
                # Draw grid lines
                cv2.rectangle(frame, (cell.x1, cell.y1), (cell.x2, cell.y2), (255, 255, 255), 1)
                
                # Label cells with obstacles
                if cell.has_obstacle:
                    text = f"D:{cell.obstacle_distance:.1f}m"
                    cv2.putText(frame, text, (cell.x1 + 5, cell.y1 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Blend overlay with original frame
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add legend
        legend_y = 20
        for zone_type, color in zone_colors.items():
            cv2.rectangle(result, (10, legend_y), (30, legend_y + 15), color, -1)
            cv2.putText(result, zone_type.value, (35, legend_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            legend_y += 20
        
        return result

class VisionDetector:
    """YOLOv8n-based obstacle detection with grid mapping"""
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        if not YOLO_AVAILABLE:
            self.model = None
            return
            
        try:
            self.model = YOLO(model_path)
            self.confidence_threshold = confidence_threshold
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load YOLO model: {e}. Vision detection disabled.")
            self.model = None
        
    def detect_obstacles(self, frame: np.ndarray, grid_manager: GridZoneManager) -> List[ObstacleInfo]:
        """Detect obstacles and map them to grid cells"""
        if self.model is None:
            return []
            
        try:
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            obstacles = []
            
            # Reset grid obstacles
            grid_manager.reset_obstacles()
            
            if results and len(results) > 0:
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = float(box.conf[0])
                            bbox = (x1, y1, x2, y2)
                            
                            # Determine obstacle direction
                            frame_center = frame.shape[1] // 2
                            obstacle_center = (x1 + x2) // 2
                            
                            if obstacle_center < frame_center - 50:
                                direction = 'left'
                            elif obstacle_center > frame_center + 50:
                                direction = 'right'
                            else:
                                direction = 'front'
                            
                            # Estimate distance based on bounding box size
                            bbox_area = (x2 - x1) * (y2 - y1)
                            frame_area = frame.shape[0] * frame.shape[1]
                            area_ratio = bbox_area / frame_area
                            
                            if area_ratio > 0.5:
                                estimated_distance = 0.5
                            elif area_ratio > 0.3:
                                estimated_distance = 1.0
                            elif area_ratio > 0.1:
                                estimated_distance = 2.0
                            elif area_ratio > 0.05:
                                estimated_distance = 3.0
                            else:
                                estimated_distance = 5.0
                            
                            # Get grid cells for this obstacle
                            grid_cells = grid_manager.get_cells_for_bbox(bbox)
                            
                            # Update grid with obstacle information
                            grid_manager.update_obstacle_in_cells(grid_cells, confidence, estimated_distance)
                            
                            obstacles.append(ObstacleInfo(
                                distance=estimated_distance,
                                direction=direction,
                                confidence=confidence,
                                bbox=bbox,
                                grid_cells=grid_cells
                            ))
            
            return obstacles
            
        except Exception as e:
            logger.error(f"Vision detection error: {e}")
            return []

class CameraStreamer:
    """Camera streamer that captures frames and provides them for web streaming"""
    
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize components
        self.grid_manager = GridZoneManager(width, height)
        self.vision_detector = VisionDetector()
        self.grid_manager.update_zone_types()
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        
    def capture_frame(self):
        """Capture a single frame using libcamera-still"""
        try:
            frame_file = os.path.join(self.temp_dir, "stream_frame.jpg")
            
            cmd = [
                'libcamera-still',
                '--width', str(self.width),
                '--height', str(self.height),
                '--output', frame_file,
                '--timeout', '100',
                '--nopreview'
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=2)
            
            if result.returncode == 0 and os.path.exists(frame_file):
                # Read the frame
                frame = cv2.imread(frame_file)
                
                # Clean up temp file
                try:
                    os.remove(frame_file)
                except:
                    pass
                    
                return frame
            
            return None
            
        except Exception as e:
            print(f"Capture error: {e}")
            return None
    
    def process_frame(self, frame):
        """Process frame with grid overlay and obstacle detection"""
        if frame is None:
            return None
            
        # Detect obstacles
        obstacles = self.vision_detector.detect_obstacles(frame, self.grid_manager)
        
        # Draw obstacles
        processed_frame = frame.copy()
        for obs in obstacles:
            if obs.bbox:
                x1, y1, x2, y2 = obs.bbox
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{obs.direction} {obs.distance:.1f}m"
                cv2.putText(processed_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw grid overlay
        final_frame = self.grid_manager.draw_grid_overlay(processed_frame, show_zones=True)
        
        # Add status info
        elapsed = time.time() - self.start_time
        timestamp = time.strftime("%H:%M:%S")
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        danger_cells = self.grid_manager.get_danger_cells()
        
        cv2.putText(final_frame, f"Time: {timestamp}", (10, 460),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(final_frame, f"FPS: {fps:.1f}", (150, 460),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(final_frame, f"Obstacles: {len(obstacles)}", (250, 460),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(final_frame, f"Danger: {len(danger_cells)}", (400, 460),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return final_frame
    
    def streaming_loop(self):
        """Main streaming loop"""
        print("🎥 Starting camera streaming loop...")
        
        while self.running:
            try:
                # Capture frame
                frame = self.capture_frame()
                
                if frame is not None:
                    # Process frame
                    processed_frame = self.process_frame(frame)
                    
                    if processed_frame is not None:
                        # Update current frame
                        with self.frame_lock:
                            self.current_frame = processed_frame.copy()
                        
                        self.frame_count += 1
                        
                        # Print stats periodically
                        if self.frame_count % 30 == 0:
                            elapsed = time.time() - self.start_time
                            fps = self.frame_count / elapsed
                            print(f"📊 Frames: {self.frame_count}, FPS: {fps:.1f}")
                
                time.sleep(0.1)  # ~10 FPS
                
            except Exception as e:
                print(f"Streaming error: {e}")
                time.sleep(1)
    
    def get_current_frame(self):
        """Get the current frame for web streaming"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def start(self):
        """Start the camera streaming"""
        self.running = True
        self.start_time = time.time()
        self.streaming_thread = threading.Thread(target=self.streaming_loop)
        self.streaming_thread.daemon = True
        self.streaming_thread.start()
    
    def stop(self):
        """Stop the camera streaming"""
        self.running = False
        if hasattr(self, 'streaming_thread'):
            self.streaming_thread.join()
        
        # Clean up temp directory
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except:
            pass

# Global camera streamer instance
camera_streamer = None

class StreamingHandler(BaseHTTPRequestHandler):
    """HTTP handler for streaming video"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
<!DOCTYPE html>
<html>
<head>
    <title>🚁 Drone Camera Live Stream</title>
    <style>
        body { 
            background: #000; 
            color: #fff; 
            font-family: Arial, sans-serif; 
            text-align: center;
            margin: 0;
            padding: 20px;
        }
        h1 { color: #00ff00; }
        img { 
            max-width: 90vw; 
            max-height: 70vh; 
            border: 2px solid #00ff00;
            border-radius: 10px;
        }
        .info {
            margin: 20px;
            padding: 10px;
            background: rgba(0,255,0,0.1);
            border-radius: 5px;
            display: inline-block;
        }
        .controls {
            margin: 20px;
        }
        button {
            background: #00ff00;
            color: #000;
            border: none;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
        }
        .legend {
            text-align: left;
            display: inline-block;
            margin: 20px;
        }
        .zone { 
            display: inline-block; 
            width: 20px; 
            height: 20px; 
            margin-right: 10px;
            vertical-align: middle;
        }
        .safe { background: #00ff00; }
        .warning { background: #ffff00; }
        .danger { background: #ffa500; }
        .critical { background: #ff0000; }
    </style>
    <script>
        function refreshStream() {
            document.getElementById('stream').src = '/stream?' + new Date().getTime();
        }
        setInterval(function() {
            document.getElementById('timestamp').innerHTML = new Date().toLocaleTimeString();
        }, 1000);
    </script>
</head>
<body>
    <h1>🚁 Drone Camera Live Stream</h1>
    <div class="info">
        <strong>Live Feed with Grid-Based Obstacle Detection</strong><br>
        Time: <span id="timestamp"></span>
    </div>
    
    <br>
    <img id="stream" src="/stream" alt="Camera Stream">
    
    <div class="controls">
        <button onclick="refreshStream()">🔄 Refresh Stream</button>
        <button onclick="location.reload()">🔄 Reload Page</button>
    </div>
    
    <div class="legend">
        <h3>Grid Zone Legend:</h3>
        <div><span class="zone safe"></span> Safe Zone</div>
        <div><span class="zone warning"></span> Warning Zone</div>
        <div><span class="zone danger"></span> Danger Zone</div>
        <div><span class="zone critical"></span> Critical Zone</div>
    </div>
    
    <div class="info">
        <strong>🎯 Features:</strong><br>
        • Real-time obstacle detection<br>
        • 8x8 grid zone analysis<br>
        • Distance estimation<br>
        • Danger zone highlighting
    </div>
</body>
</html>
            """
            
            self.wfile.write(html.encode())
            
        elif self.path.startswith('/stream'):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            
            try:
                while True:
                    frame = camera_streamer.get_current_frame()
                    if frame is not None:
                        # Encode frame as JPEG
                        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ret:
                            self.wfile.write(b'--frame\r\n')
                            self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                            self.wfile.write(jpeg.tobytes())
                            self.wfile.write(b'\r\n')
                    
                    time.sleep(0.1)  # ~10 FPS
                    
            except Exception as e:
                print(f"Streaming error: {e}")
    
    def log_message(self, format, *args):
        """Suppress log messages"""
        pass

def main():
    """Main function for web streaming"""
    global camera_streamer
    
    print("🌐 Drone Camera Live Web Stream")
    print("=" * 50)
    print("Streams live camera feed to web browser")
    print("Avoids Qt/display issues completely")
    print("=" * 50)
    
    # Get settings
    try:
        width = int(input("Camera width (default 640): ") or "640")
        height = int(input("Camera height (default 480): ") or "480")
        port = int(input("Web server port (default 8080): ") or "8080")
    except ValueError:
        width, height, port = 640, 480, 8080
    
    # Check libcamera availability
    try:
        result = subprocess.run(['libcamera-still', '--help'], capture_output=True, timeout=5)
        if result.returncode != 0:
            print("❌ libcamera-still not available")
            return
    except:
        print("❌ libcamera commands not found")
        print("Install with: sudo apt install libcamera-apps")
        return
    
    # Start camera streamer
    print(f"\n🎥 Starting camera streamer ({width}x{height})...")
    camera_streamer = CameraStreamer(width, height)
    camera_streamer.start()
    
    # Give camera time to initialize
    time.sleep(2)
    
    # Start web server
    try:
        print(f"🌐 Starting web server on port {port}...")
        server = HTTPServer(('0.0.0.0', port), StreamingHandler)
        
        print(f"\n✅ Live stream ready!")
        print(f"🌐 Open your browser and go to:")
        print(f"   http://localhost:{port}")
        print(f"   http://192.168.159.156:{port} (if on local network)")
        print(f"\n📱 Works on phone/tablet too!")
        print(f"🎮 Press Ctrl+C to stop")
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Stopping stream...")
    except Exception as e:
        print(f"❌ Server error: {e}")
    finally:
        if camera_streamer:
            camera_streamer.stop()

if __name__ == "__main__":
    main()