#!/usr/bin/env python3
"""
Raspberry Pi Camera Test using libcamera commands
Uses libcamera-raw/libcamera-vid for frame capture
Compatible with new Raspberry Pi camera system
"""

import cv2
import numpy as np
import time
import logging
import subprocess
import threading
import queue
import tempfile
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict
import math
from enum import Enum

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

class LibCameraCapture:
    """Camera capture using libcamera commands"""
    
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.capture_process = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.running = False
        self.temp_dir = tempfile.mkdtemp()
        
    def check_libcamera_availability(self):
        """Check if libcamera commands are available"""
        commands_to_check = ['libcamera-raw', 'libcamera-vid', 'libcamera-hello']
        available_commands = []
        
        for cmd in commands_to_check:
            try:
                result = subprocess.run([cmd, '--help'], capture_output=True, timeout=5)
                if result.returncode == 0:
                    available_commands.append(cmd)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        return available_commands
    
    def start_capture_method1(self):
        """Start capture using libcamera-vid with JPEG output"""
        try:
            cmd = [
                'libcamera-vid',
                '--width', str(self.width),
                '--height', str(self.height),
                '--framerate', str(self.fps),
                '--timeout', '0',  # Continuous
                '--inline',  # Include headers in stream
                '--output', '-',  # Output to stdout
                '--codec', 'mjpeg',  # MJPEG format
                '--quality', '80'
            ]
            
            print(f"Starting libcamera-vid: {' '.join(cmd)}")
            
            self.capture_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            return True
            
        except Exception as e:
            print(f"Failed to start libcamera-vid: {e}")
            return False
    
    def start_capture_method2(self):
        """Start capture using libcamera-raw with file output"""
        try:
            # Use libcamera-raw to continuously capture frames to files
            self.frame_file = os.path.join(self.temp_dir, "frame.jpg")
            
            cmd = [
                'libcamera-still',
                '--width', str(self.width),
                '--height', str(self.height),
                '--timeout', '0',
                '--output', self.frame_file,
                '--timelapse', '100',  # Capture every 100ms
                '--nopreview'
            ]
            
            print(f"Starting libcamera-still: {' '.join(cmd)}")
            
            self.capture_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            return True
            
        except Exception as e:
            print(f"Failed to start libcamera-still: {e}")
            return False
    
    def start_capture_method3(self):
        """Start capture using rapid libcamera-still calls"""
        # This method calls libcamera-still repeatedly
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        return True
    
    def _capture_loop(self):
        """Continuous capture loop using libcamera-still"""
        frame_counter = 0
        while self.running:
            try:
                frame_file = os.path.join(self.temp_dir, f"frame_{frame_counter}.jpg")
                
                cmd = [
                    'libcamera-still',
                    '--width', str(self.width),
                    '--height', str(self.height),
                    '--output', frame_file,
                    '--timeout', '1',
                    '--nopreview'
                ]
                
                result = subprocess.run(cmd, capture_output=True, timeout=2)
                
                if result.returncode == 0 and os.path.exists(frame_file):
                    # Read the frame
                    frame = cv2.imread(frame_file)
                    if frame is not None:
                        # Add to queue (non-blocking)
                        try:
                            self.frame_queue.put(frame, block=False)
                        except queue.Full:
                            # Remove oldest frame and add new one
                            try:
                                self.frame_queue.get_nowait()
                                self.frame_queue.put(frame, block=False)
                            except queue.Empty:
                                pass
                    
                    # Clean up frame file
                    try:
                        os.remove(frame_file)
                    except:
                        pass
                
                frame_counter = (frame_counter + 1) % 1000
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(0.5)
    
    def start_capture(self):
        """Start camera capture using the best available method"""
        available_commands = self.check_libcamera_availability()
        
        if not available_commands:
            print("❌ No libcamera commands found!")
            return False
        
        print(f"✅ Available libcamera commands: {available_commands}")
        
        # Try different capture methods
        methods = [
            ("libcamera-still (rapid)", self.start_capture_method3),
            ("libcamera-vid (MJPEG)", self.start_capture_method1),
            ("libcamera-still (timelapse)", self.start_capture_method2),
        ]
        
        for method_name, method_func in methods:
            print(f"🔄 Trying {method_name}...")
            if method_func():
                print(f"✅ {method_name} started successfully")
                self.running = True
                return True
            else:
                print(f"❌ {method_name} failed")
        
        return False
    
    def get_frame(self):
        """Get the latest frame"""
        try:
            # Get the most recent frame
            frame = None
            while not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
            return frame
        except queue.Empty:
            return None
    
    def stop_capture(self):
        """Stop camera capture"""
        self.running = False
        
        if self.capture_process:
            self.capture_process.terminate()
            self.capture_process.wait()
        
        # Clean up temp directory
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except:
            pass

class LibCameraTester:
    """Camera tester using libcamera system"""
    
    def __init__(self):
        self.camera_capture = LibCameraCapture()
        self.grid_manager = GridZoneManager()
        self.vision_detector = VisionDetector()
    
    def run_diagnostics(self):
        """Run camera diagnostics"""
        print("\n" + "="*60)
        print("LIBCAMERA SYSTEM DIAGNOSTICS")
        print("="*60)
        
        # Check system info
        import platform
        print(f"System: {platform.system()} {platform.release()}")
        print(f"Python: {platform.python_version()}")
        print(f"OpenCV: {cv2.__version__}")
        
        # Check libcamera commands
        print("\n📷 libcamera Commands:")
        available_commands = self.camera_capture.check_libcamera_availability()
        if available_commands:
            print(f"✅ Available: {available_commands}")
        else:
            print("❌ No libcamera commands found")
            print("Install with: sudo apt install libcamera-apps")
        
        # Test basic camera
        print("\n🔍 Camera Hardware Test:")
        try:
            result = subprocess.run(['libcamera-hello', '--list-cameras'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ Camera hardware detected:")
                print(result.stdout)
            else:
                print("❌ Camera hardware test failed")
                print(result.stderr)
        except Exception as e:
            print(f"❌ Camera test error: {e}")
        
        print("="*60)
        return len(available_commands) > 0
    
    def test_camera(self):
        """Test camera with grid overlay and obstacle detection"""
        if not self.run_diagnostics():
            print("❌ Diagnostics failed - cannot proceed")
            return
        
        print("\n🚀 Starting libcamera test...")
        
        if not self.camera_capture.start_capture():
            print("❌ Failed to start camera capture")
            return
        
        try:
            self.grid_manager.update_zone_types()
            
            print("\n🎥 libcamera Test Running")
            print("Controls: 'q'=quit, 's'=screenshot, 'h'=help, 'r'=restart capture")
            
            frame_count = 0
            fps_start = time.time()
            fps = 0
            
            # Wait for first frame
            print("Waiting for first frame...")
            start_wait = time.time()
            while time.time() - start_wait < 10:  # Wait up to 10 seconds
                frame = self.camera_capture.get_frame()
                if frame is not None:
                    print("✅ First frame received!")
                    break
                time.sleep(0.1)
            else:
                print("❌ No frames received after 10 seconds")
                return
            
            while True:
                frame = self.camera_capture.get_frame()
                
                if frame is not None:
                    # Process frame
                    obstacles = self.vision_detector.detect_obstacles(frame, self.grid_manager)
                    
                    # Draw obstacles
                    for obs in obstacles:
                        if obs.bbox:
                            x1, y1, x2, y2 = obs.bbox
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            label = f"{obs.direction} {obs.distance:.1f}m"
                            cv2.putText(frame, label, (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Draw grid overlay
                    processed_frame = self.grid_manager.draw_grid_overlay(frame, show_zones=True)
                    
                    # Add status info
                    cv2.putText(processed_frame, "LIBCAMERA TEST", (10, 460),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Calculate FPS
                    frame_count += 1
                    if frame_count % 10 == 0:
                        fps = 10 / (time.time() - fps_start)
                        fps_start = time.time()
                    
                    cv2.putText(processed_frame, f"FPS: {fps:.1f}", (500, 460),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    cv2.putText(processed_frame, f"Obstacles: {len(obstacles)}", (300, 460),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    cv2.imshow("libcamera Test", processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('s'):
                    if frame is not None:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"libcamera_test_{timestamp}.jpg"
                        cv2.imwrite(filename, processed_frame)
                        print(f"Screenshot saved: {filename}")
                elif key == ord('h'):
                    print("Controls: q=quit, s=screenshot, h=help, r=restart")
                elif key == ord('r'):
                    print("Restarting capture...")
                    self.camera_capture.stop_capture()
                    time.sleep(1)
                    if not self.camera_capture.start_capture():
                        print("Failed to restart capture")
                        break
                
                time.sleep(0.01)  # Small delay
        
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        except Exception as e:
            print(f"Test error: {e}")
        finally:
            self.camera_capture.stop_capture()
            cv2.destroyAllWindows()
            print("Camera test completed")

def main():
    """Main function for libcamera testing"""
    print("📷 libcamera Raspberry Pi Camera Test")
    print("="*60)
    print("Using libcamera system for frame capture")
    print("Compatible with new Raspberry Pi OS camera system")
    print("="*60)
    
    tester = LibCameraTester()
    tester.test_camera()

if __name__ == "__main__":
    main()