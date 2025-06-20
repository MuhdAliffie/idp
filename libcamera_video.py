#!/usr/bin/env python3
"""
Raspberry Pi Camera Video Recording using libcamera
Records video while processing frames for obstacle detection
Saves both raw video and processed video with grid overlay
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
import signal
import sys
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

class LibCameraVideoRecorder:
    """Video recorder using libcamera with real-time processing"""
    
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.recording = False
        self.output_dir = f"video_output_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Video recording processes
        self.raw_video_process = None
        self.processed_video_writer = None
        
        # Frame processing
        self.grid_manager = GridZoneManager(width, height)
        self.vision_detector = VisionDetector()
        self.grid_manager.update_zone_types()
        
        # Statistics
        self.frame_count = 0
        self.obstacle_detections = []
        self.start_time = None
        
    def check_libcamera_availability(self):
        """Check if libcamera commands are available"""
        commands_to_check = ['libcamera-vid', 'libcamera-still', 'libcamera-hello']
        available_commands = []
        
        for cmd in commands_to_check:
            try:
                result = subprocess.run([cmd, '--help'], capture_output=True, timeout=5)
                if result.returncode == 0:
                    available_commands.append(cmd)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        return available_commands
    
    def start_raw_video_recording(self, duration_minutes=10):
        """Start recording raw video using libcamera-vid"""
        try:
            raw_video_file = os.path.join(self.output_dir, "raw_video.h264")
            
            cmd = [
                'libcamera-vid',
                '--width', str(self.width),
                '--height', str(self.height),
                '--framerate', str(self.fps),
                '--timeout', str(duration_minutes * 60 * 1000),  # Convert to milliseconds
                '--output', raw_video_file,
                '--nopreview'
            ]
            
            print(f"🎬 Starting raw video recording: {raw_video_file}")
            print(f"Command: {' '.join(cmd)}")
            
            self.raw_video_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            return True
            
        except Exception as e:
            print(f"Failed to start raw video recording: {e}")
            return False
    
    def setup_processed_video_writer(self):
        """Setup OpenCV video writer for processed frames"""
        try:
            processed_video_file = os.path.join(self.output_dir, "processed_video.avi")
            
            # Use XVID codec (widely compatible)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            
            self.processed_video_writer = cv2.VideoWriter(
                processed_video_file,
                fourcc,
                self.fps,
                (self.width, self.height)
            )
            
            if self.processed_video_writer.isOpened():
                print(f"📹 Processed video writer ready: {processed_video_file}")
                return True
            else:
                print("❌ Failed to open processed video writer")
                return False
                
        except Exception as e:
            print(f"Failed to setup processed video writer: {e}")
            return False
    
    def capture_and_process_frame(self):
        """Capture a single frame and process it"""
        try:
            # Capture frame using libcamera-still (quick capture)
            temp_frame_file = os.path.join(self.output_dir, "temp_frame.jpg")
            
            cmd = [
                'libcamera-still',
                '--width', str(self.width),
                '--height', str(self.height),
                '--output', temp_frame_file,
                '--timeout', '100',  # Very quick capture
                '--nopreview'
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=2)
            
            if result.returncode == 0 and os.path.exists(temp_frame_file):
                # Read the frame
                frame = cv2.imread(temp_frame_file)
                
                if frame is not None:
                    # Process frame
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
                    
                    # Add timestamp and stats
                    timestamp = time.strftime("%H:%M:%S")
                    cv2.putText(final_frame, f"Time: {timestamp}", (10, 460),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(final_frame, f"Frame: {self.frame_count}", (200, 460),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(final_frame, f"Obstacles: {len(obstacles)}", (350, 460),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_frame_file)
                    except:
                        pass
                    
                    # Update statistics
                    self.frame_count += 1
                    if obstacles:
                        self.obstacle_detections.append({
                            'frame': self.frame_count,
                            'timestamp': timestamp,
                            'obstacles': len(obstacles),
                            'details': [{'direction': obs.direction, 'distance': obs.distance, 'confidence': obs.confidence} for obs in obstacles]
                        })
                    
                    return final_frame, obstacles
                
                # Clean up temp file even if reading failed
                try:
                    os.remove(temp_frame_file)
                except:
                    pass
            
            return None, []
            
        except Exception as e:
            print(f"Frame capture error: {e}")
            return None, []
    
    def record_video(self, duration_minutes=5, process_interval=0.5):
        """Record video with real-time processing"""
        
        print(f"\n🎬 Starting video recording for {duration_minutes} minutes")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"⚙️ Settings: {self.width}x{self.height} @ {self.fps}fps")
        print(f"🔄 Processing interval: {process_interval}s")
        
        # Check libcamera availability
        available_commands = self.check_libcamera_availability()
        if 'libcamera-vid' not in available_commands:
            print("❌ libcamera-vid not available")
            return False
        
        # Start raw video recording
        if not self.start_raw_video_recording(duration_minutes):
            print("❌ Failed to start raw video recording")
            return False
        
        # Setup processed video writer
        if not self.setup_processed_video_writer():
            print("❌ Failed to setup processed video writer")
            return False
        
        self.recording = True
        self.start_time = time.time()
        
        print(f"\n🔴 Recording started! Press Ctrl+C to stop early")
        print(f"📊 Real-time stats will be displayed...")
        
        try:
            last_process_time = 0
            
            while self.recording:
                current_time = time.time()
                elapsed = current_time - self.start_time
                
                # Check if recording duration exceeded
                if elapsed >= duration_minutes * 60:
                    print(f"\n⏰ Recording duration ({duration_minutes} minutes) reached")
                    break
                
                # Process frame at specified interval
                if current_time - last_process_time >= process_interval:
                    processed_frame, obstacles = self.capture_and_process_frame()
                    
                    if processed_frame is not None:
                        # Write to processed video
                        self.processed_video_writer.write(processed_frame)
                        
                        # Print real-time stats
                        if self.frame_count % 10 == 0:  # Every 10th frame
                            danger_cells = self.grid_manager.get_danger_cells()
                            print(f"⏱️ {elapsed:.1f}s | Frame {self.frame_count} | "
                                  f"Obstacles: {len(obstacles)} | Danger zones: {len(danger_cells)}")
                    
                    last_process_time = current_time
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print(f"\n⏹️ Recording stopped by user after {time.time() - self.start_time:.1f}s")
        
        finally:
            self.stop_recording()
        
        return True
    
    def stop_recording(self):
        """Stop recording and cleanup"""
        self.recording = False
        
        print(f"\n🛑 Stopping recording...")
        
        # Stop raw video process
        if self.raw_video_process:
            self.raw_video_process.terminate()
            self.raw_video_process.wait()
            print("✅ Raw video recording stopped")
        
        # Close processed video writer
        if self.processed_video_writer:
            self.processed_video_writer.release()
            print("✅ Processed video saved")
        
        # Convert raw H.264 to MP4 for better compatibility
        self.convert_raw_video()
        
        # Save statistics
        self.save_statistics()
        
        print(f"\n📊 Recording Summary:")
        print(f"⏱️ Duration: {time.time() - self.start_time:.1f}s")
        print(f"📸 Frames processed: {self.frame_count}")
        print(f"🚨 Obstacle detections: {len(self.obstacle_detections)}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📹 Files: raw_video.mp4, processed_video.avi, statistics.txt")
    
    def convert_raw_video(self):
        """Convert raw H.264 to MP4 using ffmpeg"""
        try:
            raw_file = os.path.join(self.output_dir, "raw_video.h264")
            mp4_file = os.path.join(self.output_dir, "raw_video.mp4")
            
            if os.path.exists(raw_file):
                print("🔄 Converting H.264 to MP4...")
                cmd = ['ffmpeg', '-i', raw_file, '-c', 'copy', mp4_file, '-y']
                result = subprocess.run(cmd, capture_output=True)
                
                if result.returncode == 0:
                    print("✅ Raw video converted to MP4")
                    # Remove original H.264 file
                    os.remove(raw_file)
                else:
                    print("⚠️ ffmpeg conversion failed - keeping H.264 file")
        except Exception as e:
            print(f"⚠️ Video conversion error: {e}")
    
    def save_statistics(self):
        """Save recording statistics to file"""
        try:
            stats_file = os.path.join(self.output_dir, "statistics.txt")
            
            with open(stats_file, 'w') as f:
                f.write("DRONE CAMERA RECORDING STATISTICS\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Recording Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}\n")
                f.write(f"Duration: {time.time() - self.start_time:.1f} seconds\n")
                f.write(f"Resolution: {self.width}x{self.height}\n")
                f.write(f"Target FPS: {self.fps}\n")
                f.write(f"Frames Processed: {self.frame_count}\n")
                f.write(f"Obstacle Detections: {len(self.obstacle_detections)}\n\n")
                
                if self.obstacle_detections:
                    f.write("OBSTACLE DETECTION LOG:\n")
                    f.write("-" * 30 + "\n")
                    for detection in self.obstacle_detections:
                        f.write(f"Frame {detection['frame']} ({detection['timestamp']}): {detection['obstacles']} obstacles\n")
                        for obs in detection['details']:
                            f.write(f"  - {obs['direction']} at {obs['distance']:.1f}m (conf: {obs['confidence']:.2f})\n")
                        f.write("\n")
                
            print(f"📊 Statistics saved to: {stats_file}")
            
        except Exception as e:
            print(f"⚠️ Failed to save statistics: {e}")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n⏹️ Stopping recording...')
    sys.exit(0)

def main():
    """Main function for video recording"""
    print("🎬 libcamera Video Recording with Obstacle Detection")
    print("=" * 60)
    print("Records both raw video and processed video with grid overlay")
    print("=" * 60)
    
    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Get recording parameters
    try:
        width = int(input("Video width (default 640): ") or "640")
        height = int(input("Video height (default 480): ") or "480")
        fps = int(input("Frames per second (default 30): ") or "30")
        duration = float(input("Recording duration in minutes (default 5): ") or "5")
        process_interval = float(input("Frame processing interval in seconds (default 0.5): ") or "0.5")
    except ValueError:
        print("Using default values...")
        width, height, fps, duration, process_interval = 640, 480, 30, 5, 0.5
    
    # Create recorder
    recorder = LibCameraVideoRecorder(width, height, fps)
    
    # Check camera
    print("\n🔍 Checking camera availability...")
    available_commands = recorder.check_libcamera_availability()
    
    if not available_commands:
        print("❌ No libcamera commands found!")
        print("Install with: sudo apt install libcamera-apps")
        return
    
    print(f"✅ Available commands: {available_commands}")
    
    # Test camera hardware
    try:
        result = subprocess.run(['libcamera-hello', '--list-cameras'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Camera hardware detected")
        else:
            print("❌ Camera hardware test failed")
            return
    except Exception as e:
        print(f"❌ Camera test error: {e}")
        return
    
    # Start recording
    print(f"\n🚀 Ready to start recording!")
    input("Press Enter to start recording...")
    
    success = recorder.record_video(duration, process_interval)
    
    if success:
        print(f"\n✅ Recording completed successfully!")
        print(f"📁 Check output directory: {recorder.output_dir}")
    else:
        print(f"\n❌ Recording failed")

if __name__ == "__main__":
    main()