#!/usr/bin/env python3
"""
Headless Raspberry Pi Camera Test using libcamera
No GUI - saves processed images to files
Avoids Qt/Wayland display issues
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

class LibCameraHeadless:
    """Headless camera capture using libcamera - saves images instead of showing"""
    
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.running = False
        self.output_dir = f"camera_output_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def check_libcamera_availability(self):
        """Check if libcamera commands are available"""
        commands_to_check = ['libcamera-still', 'libcamera-vid', 'libcamera-hello']
        available_commands = []
        
        for cmd in commands_to_check:
            try:
                result = subprocess.run([cmd, '--help'], capture_output=True, timeout=5)
                if result.returncode == 0:
                    available_commands.append(cmd)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        return available_commands
    
    def capture_single_frame(self):
        """Capture a single frame using libcamera-still"""
        try:
            frame_file = os.path.join(self.output_dir, "temp_frame.jpg")
            
            cmd = [
                'libcamera-still',
                '--width', str(self.width),
                '--height', str(self.height),
                '--output', frame_file,
                '--timeout', '1000',  # 1 second timeout
                '--nopreview'
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=5)
            
            if result.returncode == 0 and os.path.exists(frame_file):
                # Read the frame
                frame = cv2.imread(frame_file)
                # Clean up temp file
                try:
                    os.remove(frame_file)
                except:
                    pass
                return frame
            else:
                print(f"Capture failed: {result.stderr.decode()}")
                return None
                
        except Exception as e:
            print(f"Capture error: {e}")
            return None
    
    def run_headless_test(self, num_frames=50, interval=2.0):
        """Run headless camera test - capture and process frames"""
        
        print(f"\n🎥 Starting headless camera test")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📸 Capturing {num_frames} frames, {interval}s apart")
        print(f"💾 Processed images will be saved to: {self.output_dir}/")
        print("\nPress Ctrl+C to stop early")
        
        grid_manager = GridZoneManager(self.width, self.height)
        vision_detector = VisionDetector()
        grid_manager.update_zone_types()
        
        successful_captures = 0
        
        try:
            for frame_num in range(num_frames):
                print(f"\n📸 Capturing frame {frame_num + 1}/{num_frames}...")
                
                # Capture frame
                frame = self.capture_single_frame()
                
                if frame is not None:
                    print(f"✅ Frame captured: {frame.shape}")
                    
                    # Process frame
                    obstacles = vision_detector.detect_obstacles(frame, grid_manager)
                    
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
                    final_frame = grid_manager.draw_grid_overlay(processed_frame, show_zones=True)
                    
                    # Add status info
                    cv2.putText(final_frame, f"Frame {frame_num + 1}/{num_frames}", (10, 460),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(final_frame, f"Obstacles: {len(obstacles)}", (300, 460),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Save processed frame
                    timestamp = time.strftime("%H%M%S")
                    filename = f"frame_{frame_num+1:03d}_{timestamp}.jpg"
                    filepath = os.path.join(self.output_dir, filename)
                    cv2.imwrite(filepath, final_frame)
                    
                    # Save original frame too
                    orig_filename = f"original_{frame_num+1:03d}_{timestamp}.jpg"
                    orig_filepath = os.path.join(self.output_dir, orig_filename)
                    cv2.imwrite(orig_filepath, frame)
                    
                    print(f"💾 Saved: {filename}")
                    
                    # Print obstacle info
                    if obstacles:
                        print(f"🚨 Detected {len(obstacles)} obstacles:")
                        for i, obs in enumerate(obstacles):
                            print(f"   {i+1}. {obs.direction} at {obs.distance:.1f}m (confidence: {obs.confidence:.2f})")
                    else:
                        print("✅ No obstacles detected")
                    
                    # Get danger zones info
                    danger_cells = grid_manager.get_danger_cells()
                    if danger_cells:
                        print(f"⚠️ {len(danger_cells)} danger zones occupied")
                    else:
                        print("✅ No danger zones occupied")
                    
                    successful_captures += 1
                    
                else:
                    print("❌ Failed to capture frame")
                
                # Wait before next capture
                if frame_num < num_frames - 1:  # Don't wait after last frame
                    print(f"⏳ Waiting {interval}s...")
                    time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Test stopped by user after {successful_captures} frames")
        
        print(f"\n📊 Test Summary:")
        print(f"✅ Successfully captured: {successful_captures}/{num_frames} frames")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📸 Files saved: original_*.jpg (raw frames), frame_*.jpg (processed)")
        
        return successful_captures > 0

def main():
    """Main function for headless libcamera testing"""
    print("📷 Headless libcamera Raspberry Pi Camera Test")
    print("="*60)
    print("No GUI - saves processed images to files")
    print("Avoids Qt/Wayland display issues")
    print("="*60)
    
    # Check libcamera availability
    tester = LibCameraHeadless()
    available_commands = tester.check_libcamera_availability()
    
    if not available_commands:
        print("❌ No libcamera commands found!")
        print("Install with: sudo apt install libcamera-apps")
        return
    
    print(f"✅ Available libcamera commands: {available_commands}")
    
    # Test camera hardware
    print("\n🔍 Testing camera hardware...")
    try:
        result = subprocess.run(['libcamera-hello', '--list-cameras'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Camera hardware detected:")
            print(result.stdout)
        else:
            print("❌ Camera hardware test failed")
            print(result.stderr)
            return
    except Exception as e:
        print(f"❌ Camera test error: {e}")
        return
    
    # Get user preferences
    try:
        num_frames = int(input("\nHow many frames to capture? (default: 10): ") or "10")
        interval = float(input("Interval between frames in seconds? (default: 2.0): ") or "2.0")
    except ValueError:
        num_frames = 10
        interval = 2.0
    
    # Run the test
    success = tester.run_headless_test(num_frames, interval)
    
    if success:
        print(f"\n✅ Test completed successfully!")
        print(f"📁 Check the output directory for processed images")
    else:
        print(f"\n❌ Test failed - no frames captured")

if __name__ == "__main__":
    main()