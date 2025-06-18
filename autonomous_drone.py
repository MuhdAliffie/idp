#!/usr/bin/env python3
"""
Autonomous Drone with Grid-Based Obstacle Detection and Avoidance System
- 8x8 grid overlay on camera feed for zone-based detection
- Safe and danger zone classification
- Enhanced obstacle avoidance with return-to-path mechanism
- YOLOv8n for computer vision obstacle detection
- Ultrasonic sensor for proximity detection
- SpeedyBee F405 flight controller communication

RASPBERRY PI 5 INSTALLATION:
---------------------------
# Install required GPIO libraries for Pi 5:
sudo apt update
sudo apt install python3-lgpio python3-gpiozero
pip3 install gpiozero lgpio

# Install other dependencies:
pip3 install opencv-python ultralytics pymavlink numpy

# If you prefer RPi.GPIO compatibility:
pip3 install rpi-lgpio
# Then replace the gpiozero imports with: import rpi_lgpio.GPIO as GPIO
"""

import cv2
import numpy as np
import time
import threading
import queue
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict
import math
import serial
import struct
from ultralytics import YOLO
# For Raspberry Pi 5 compatibility, use one of these:
# Option 1: gpiozero (Recommended)
from gpiozero import DistanceSensor, Device
from gpiozero.pins.lgpio import LGPIOFactory
# Option 2: rpi-lgpio (drop-in replacement)
# import rpi_lgpio.GPIO as GPIO
# Option 3: lgpio directly
# import lgpio
from pymavlink import mavutil
from enum import Enum

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
class Position:
    """GPS position representation"""
    lat: float
    lon: float
    alt: float
    
@dataclass
class Waypoint:
    """Waypoint with position and acceptance radius"""
    position: Position
    acceptance_radius: float = 5.0  # meters

@dataclass
class ObstacleInfo:
    """Obstacle information from detection"""
    distance: float
    direction: str  # 'front', 'left', 'right'
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    grid_cells: List[Tuple[int, int]] = field(default_factory=list)  # List of (row, col) tuples

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
        
        # Define zone mapping (can be customized)
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
        
        # Define zones based on position in grid
        # Center cells (3,3), (3,4), (4,3), (4,4) are critical for forward movement
        # Surrounding cells are danger zones
        # Outer cells are warning/safe zones
        
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
        
        # Find all cells that overlap with the bounding box
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
    
    def get_safe_direction(self) -> str:
        """Determine the safest direction to move based on obstacle distribution"""
        # Count obstacles in different regions
        left_obstacles = 0
        right_obstacles = 0
        center_obstacles = 0
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                if self.grid[row][col].has_obstacle:
                    if col < 3:
                        left_obstacles += 1
                    elif col > 4:
                        right_obstacles += 1
                    else:
                        center_obstacles += 1
        
        # Determine safest direction
        if center_obstacles == 0:
            return "forward"
        elif left_obstacles < right_obstacles:
            return "left"
        elif right_obstacles < left_obstacles:
            return "right"
        else:
            return "up"  # If both sides blocked, go up
    
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

class UltrasonicSensor:
    """Ultrasonic sensor interface for HC-SR04 - Raspberry Pi 5 compatible"""
    
    def __init__(self, trigger_pin: int, echo_pin: int, max_distance: float = 4.0):
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self.max_distance = max_distance
        
        # Initialize the distance sensor using gpiozero (Pi 5 compatible)
        self.sensor = DistanceSensor(
            echo=echo_pin, 
            trigger=trigger_pin,
            max_distance=max_distance,
            threshold_distance=0.1
        )
        
        logger.info(f"Ultrasonic sensor initialized on pins trigger={trigger_pin}, echo={echo_pin}")
        
    def get_distance(self) -> float:
        """Get distance measurement in meters"""
        try:
            # gpiozero returns distance in meters
            distance = self.sensor.distance
            
            if distance is None:
                return self.max_distance
                
            return min(distance, self.max_distance)
            
        except Exception as e:
            logger.error(f"Ultrasonic sensor error: {e}")
            return self.max_distance
    
    def cleanup(self):
        """Clean up sensor resources"""
        self.sensor.close()

class VisionDetector:
    """YOLOv8n-based obstacle detection with grid mapping"""
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
    def detect_obstacles(self, frame: np.ndarray, grid_manager: GridZoneManager) -> List[ObstacleInfo]:
        """Detect obstacles and map them to grid cells"""
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
                            
                            # Determine obstacle direction based on position in frame
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
                            
                            # More sophisticated distance estimation
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

class FlightController:
    """Interface for SpeedyBee F405 flight controller via MAVLink"""
    
    def __init__(self, connection_string: str = "/dev/ttyACM0", baud_rate: int = 57600):
        self.connection_string = connection_string
        self.baud_rate = baud_rate
        self.master = None
        self.current_position = Position(0, 0, 0)
        self.current_heading = 0
        self.armed = False
        self.mode = "STABILIZE"
        
    def connect(self) -> bool:
        """Connect to flight controller"""
        try:
            self.master = mavutil.mavlink_connection(
                self.connection_string, 
                baud=self.baud_rate
            )
            
            # Wait for heartbeat
            self.master.wait_heartbeat()
            logger.info("Connected to flight controller")
            
            # Request data streams
            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                4, 1
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to flight controller: {e}")
            return False
    
    def update_telemetry(self):
        """Update telemetry data from flight controller"""
        try:
            msg = self.master.recv_match(blocking=False)
            if msg:
                if msg.get_type() == 'GLOBAL_POSITION_INT':
                    self.current_position = Position(
                        lat=msg.lat / 1e7,
                        lon=msg.lon / 1e7,
                        alt=msg.alt / 1000.0
                    )
                elif msg.get_type() == 'ATTITUDE':
                    self.current_heading = math.degrees(msg.yaw)
                elif msg.get_type() == 'HEARTBEAT':
                    self.armed = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                    
        except Exception as e:
            logger.error(f"Telemetry update error: {e}")
    
    def arm_drone(self) -> bool:
        """Arm the drone"""
        try:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, 0, 0, 0, 0, 0, 0
            )
            time.sleep(2)
            return True
        except Exception as e:
            logger.error(f"Arming failed: {e}")
            return False
    
    def set_mode(self, mode: str) -> bool:
        """Set flight mode"""
        try:
            mode_mapping = {
                'STABILIZE': 0,
                'ALT_HOLD': 2,
                'AUTO': 3,
                'GUIDED': 4,
                'LOITER': 5,
                'RTL': 6,
                'LAND': 9,
                'POSHOLD': 16,
                'BRAKE': 17
            }
            
            if mode in mode_mapping:
                self.master.mav.set_mode_send(
                    self.master.target_system,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    mode_mapping[mode]
                )
                self.mode = mode
                return True
            return False
            
        except Exception as e:
            logger.error(f"Mode change failed: {e}")
            return False
    
    def takeoff(self, altitude: float) -> bool:
        """Takeoff to specified altitude"""
        try:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0, 0, 0, 0, 0, 0, altitude
            )
            return True
        except Exception as e:
            logger.error(f"Takeoff failed: {e}")
            return False
    
    def goto_position(self, lat: float, lon: float, alt: float) -> bool:
        """Navigate to GPS position"""
        try:
            self.master.mav.set_position_target_global_int_send(
                0,
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                0b0000111111111000,
                int(lat * 1e7),
                int(lon * 1e7),
                alt,
                0, 0, 0, 0, 0, 0, 0, 0
            )
            return True
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return False
    
    def set_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float = 0) -> bool:
        """Set velocity in body frame"""
        try:
            self.master.mav.set_position_target_local_ned_send(
                0,
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                0b0000111111000111,
                0, 0, 0,
                vx, vy, vz,
                0, 0, 0,
                0, yaw_rate
            )
            return True
        except Exception as e:
            logger.error(f"Velocity command failed: {e}")
            return False
    
    def return_to_launch(self) -> bool:
        """Return to launch position"""
        try:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
                0, 0, 0, 0, 0, 0, 0, 0
            )
            return True
        except Exception as e:
            logger.error(f"RTL failed: {e}")
            return False

class AutonomousDrone:
    """Main autonomous drone controller with grid-based obstacle avoidance"""
    
    def __init__(self):
        # Initialize components
        self.flight_controller = FlightController()
        self.vision_detector = VisionDetector()
        self.ultrasonic_sensor = UltrasonicSensor(trigger_pin=18, echo_pin=24)
        self.grid_manager = GridZoneManager()
        
        # Navigation parameters
        self.waypoints: List[Waypoint] = []
        self.current_waypoint_index = 0
        self.mission_active = False
        self.avoidance_active = False
        self.return_to_path_active = False
        self.avoidance_start_position: Optional[Position] = None
        self.pre_avoidance_waypoint_index = 0
        
        # Obstacle detection parameters
        self.obstacle_clear_time = 0
        self.obstacle_clear_duration = 2.0  # seconds to confirm clear path
        self.avoidance_speed = 2.0  # m/s
        self.cruise_speed = 3.0  # m/s
        self.cruise_altitude = 10.0  # meters
        
        # Threading
        self.running = False
        self.detection_thread = None
        self.telemetry_thread = None
        self.video_writer = None
        
        # Camera
        self.camera = None
        
        # Avoidance state
        self.avoidance_direction = None
        self.avoidance_distance = 0
        
    def initialize(self) -> bool:
        """Initialize all systems"""
        logger.info("Initializing autonomous drone system...")
        
        # Connect to flight controller
        if not self.flight_controller.connect():
            logger.error("Failed to connect to flight controller")
            return False
        
        # Initialize camera
        try:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.camera.isOpened():
                logger.error("Failed to open camera")
                return False
            
            # Initialize grid zone types
            self.grid_manager.update_zone_types()
                
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
        
        logger.info("System initialized successfully")
        return True
    
    def add_waypoint(self, lat: float, lon: float, alt: float, acceptance_radius: float = 5.0):
        """Add waypoint to mission"""
        waypoint = Waypoint(
            position=Position(lat, lon, alt),
            acceptance_radius=acceptance_radius
        )
        self.waypoints.append(waypoint)
        logger.info(f"Added waypoint: {lat}, {lon}, {alt}")
    
    def calculate_distance(self, pos1: Position, pos2: Position) -> float:
        """Calculate distance between two GPS positions"""
        R = 6371000  # Earth radius in meters
        
        lat1_rad = math.radians(pos1.lat)
        lat2_rad = math.radians(pos2.lat)
        delta_lat = math.radians(pos2.lat - pos1.lat)
        delta_lon = math.radians(pos2.lon - pos1.lon)
        
        a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon/2) * math.sin(delta_lon/2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def calculate_bearing(self, pos1: Position, pos2: Position) -> float:
        """Calculate bearing from pos1 to pos2"""
        lat1_rad = math.radians(pos1.lat)
        lat2_rad = math.radians(pos2.lat)
        delta_lon = math.radians(pos2.lon - pos1.lon)
        
        y = math.sin(delta_lon) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
        
        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing
    
    def detect_obstacles(self) -> Tuple[List[ObstacleInfo], np.ndarray]:
        """Detect obstacles using vision and ultrasonic sensors, return obstacles and processed frame"""
        obstacles = []
        frame = None
        
        # Vision-based detection
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                vision_obstacles = self.vision_detector.detect_obstacles(frame, self.grid_manager)
                obstacles.extend(vision_obstacles)
                
                # Draw bounding boxes on frame
                for obs in vision_obstacles:
                    if obs.bbox:
                        x1, y1, x2, y2 = obs.bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f"{obs.direction} {obs.distance:.1f}m"
                        cv2.putText(frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Ultrasonic sensor detection
        ultrasonic_distance = self.ultrasonic_sensor.get_distance()
        if ultrasonic_distance < 3.0:  # Within 3 meters
            # Map ultrasonic reading to center cells
            center_cells = [(3, 3), (3, 4), (4, 3), (4, 4)]
            self.grid_manager.update_obstacle_in_cells(center_cells, 1.0, ultrasonic_distance)
            
            obstacles.append(ObstacleInfo(
                distance=ultrasonic_distance,
                direction='front',
                confidence=1.0,
                grid_cells=center_cells
            ))
        
        return obstacles, frame
    
    def plan_avoidance_maneuver(self) -> Tuple[float, float, float]:
        """Plan avoidance maneuver based on grid analysis"""
        danger_cells = self.grid_manager.get_danger_cells()
        
        if not danger_cells:
            # No obstacles in danger zones
            return self.cruise_speed, 0, 0
        
        # Get safe direction from grid analysis
        safe_direction = self.grid_manager.get_safe_direction()
        
        # Convert direction to velocity commands
        vx, vy, vz = 0, 0, 0
        
        if safe_direction == "forward":
            vx = self.cruise_speed
        elif safe_direction == "left":
            vy = -self.avoidance_speed
            self.avoidance_direction = "left"
        elif safe_direction == "right":
            vy = self.avoidance_speed
            self.avoidance_direction = "right"
        elif safe_direction == "up":
            vz = -self.avoidance_speed  # Negative is up in NED frame
            self.avoidance_direction = "up"
        
        # Check for critical obstacles (very close)
        critical_cells = [cell for cell in danger_cells if cell.zone_type == ZoneType.CRITICAL]
        if critical_cells:
            # Emergency avoidance - stop forward movement
            vx = 0
            # Increase lateral/vertical speed
            if vy != 0:
                vy = vy * 1.5
            if vz != 0:
                vz = vz * 1.5
        
        logger.info(f"Avoidance maneuver: direction={safe_direction}, vx={vx}, vy={vy}, vz={vz}")
        
        return vx, vy, vz
    
    def check_path_clear(self) -> bool:
        """Check if path to current waypoint is clear of obstacles"""
        danger_cells = self.grid_manager.get_danger_cells()
        return len(danger_cells) == 0
    
    def initiate_return_to_path(self):
        """Initiate return to original path after obstacle avoidance"""
        self.return_to_path_active = True
        self.obstacle_clear_time = time.time()
        logger.info("Initiating return to path...")
    
    def navigation_loop(self):
        """Main navigation loop with grid-based avoidance"""
        while self.running and self.mission_active:
            try:
                # Update telemetry
                self.flight_controller.update_telemetry()
                
                # Check if we have waypoints
                if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
                    logger.info("Mission completed")
                    self.mission_active = False
                    break
                
                # Get current waypoint
                current_waypoint = self.waypoints[self.current_waypoint_index]
                current_pos = self.flight_controller.current_position
                
                # Calculate distance to waypoint
                distance_to_waypoint = self.calculate_distance(current_pos, current_waypoint.position)
                
                # Check if waypoint reached
                if distance_to_waypoint < current_waypoint.acceptance_radius:
                    logger.info(f"Waypoint {self.current_waypoint_index} reached")
                    self.current_waypoint_index += 1
                    continue
                
                # Detect obstacles and get processed frame
                obstacles, frame = self.detect_obstacles()
                
                # Process frame with grid overlay
                if frame is not None:
                    # Draw grid overlay
                    processed_frame = self.grid_manager.draw_grid_overlay(frame, show_zones=True)
                    
                    # Add status information
                    status = "AVOIDING" if self.avoidance_active else "NAVIGATING"
                    cv2.putText(processed_frame, f"Status: {status}", (10, 460),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(processed_frame, f"Waypoint: {self.current_waypoint_index}/{len(self.waypoints)}", 
                               (400, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Save processed frame if recording
                    if self.video_writer:
                        self.video_writer.write(processed_frame)
                    
                    # Display frame (optional - remove in production)
                    # cv2.imshow("Drone Vision", processed_frame)
                    # cv2.waitKey(1)
                
                # Get danger cells from grid
                danger_cells = self.grid_manager.get_danger_cells()
                
                if danger_cells and not self.avoidance_active:
                    # Obstacles detected in danger zones - enter avoidance mode
                    self.avoidance_active = True
                    self.avoidance_start_position = current_pos
                    self.pre_avoidance_waypoint_index = self.current_waypoint_index
                    self.avoidance_distance = 0
                    logger.info(f"Obstacles detected in {len(danger_cells)} danger cells - entering avoidance mode")
                    
                elif self.avoidance_active and not danger_cells:
                    # No obstacles in danger zones - check if we can return to path
                    if not self.return_to_path_active:
                        self.initiate_return_to_path()
                    
                    # Check if path has been clear for sufficient time
                    if time.time() - self.obstacle_clear_time > self.obstacle_clear_duration:
                        self.avoidance_active = False
                        self.return_to_path_active = False
                        logger.info("Path clear - resuming normal navigation")
                        
                        # Navigate back to current waypoint
                        self.flight_controller.goto_position(
                            current_waypoint.position.lat,
                            current_waypoint.position.lon,
                            current_waypoint.position.alt
                        )
                
                if self.avoidance_active:
                    # Execute avoidance maneuver
                    vx, vy, vz = self.plan_avoidance_maneuver()
                    self.flight_controller.set_velocity(vx, vy, vz)
                    
                    # Track avoidance distance
                    self.avoidance_distance += math.sqrt(vx**2 + vy**2 + vz**2) * 0.1
                    
                else:
                    # Normal navigation to waypoint
                    self.flight_controller.goto_position(
                        current_waypoint.position.lat,
                        current_waypoint.position.lon,
                        current_waypoint.position.alt
                    )
                
                time.sleep(0.1)  # 10Hz update rate
                
            except Exception as e:
                logger.error(f"Navigation loop error: {e}")
                time.sleep(1)
    
    def telemetry_loop(self):
        """Telemetry monitoring loop"""
        while self.running:
            try:
                self.flight_controller.update_telemetry()
                time.sleep(0.05)  # 20Hz update rate
            except Exception as e:
                logger.error(f"Telemetry loop error: {e}")
                time.sleep(1)
    
    def start_mission(self, record_video: bool = False) -> bool:
        """Start autonomous mission"""
        if not self.waypoints:
            logger.error("No waypoints defined")
            return False
        
        logger.info("Starting autonomous mission with grid-based obstacle avoidance...")
        
        # Set up video recording if requested
        if record_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.video_writer = cv2.VideoWriter(f'drone_mission_{timestamp}.avi', 
                                              fourcc, 10.0, (640, 480))
        
        # Set guided mode
        if not self.flight_controller.set_mode("GUIDED"):
            logger.error("Failed to set guided mode")
            return False
        
        # Arm drone
        if not self.flight_controller.armed:
            if not self.flight_controller.arm_drone():
                logger.error("Failed to arm drone")
                return False
        
        # Takeoff
        if not self.flight_controller.takeoff(self.cruise_altitude):
            logger.error("Failed to initiate takeoff")
            return False
        
        # Wait for takeoff to complete
        time.sleep(10)
        
        # Start mission
        self.mission_active = True
        self.current_waypoint_index = 0
        self.running = True
        
        # Start threads
        self.telemetry_thread = threading.Thread(target=self.telemetry_loop)
        self.telemetry_thread.daemon = True
        self.telemetry_thread.start()
        
        # Start navigation loop in main thread
        self.navigation_loop()
        
        return True
    
    def stop_mission(self):
        """Stop mission and return to launch"""
        logger.info("Stopping mission...")
        self.running = False
        self.mission_active = False
        
        # Return to launch
        self.flight_controller.return_to_launch()
        
        # Wait for threads to finish
        if self.telemetry_thread:
            self.telemetry_thread.join()
        
        # Close video writer
        if self.video_writer:
            self.video_writer.release()
    
    def emergency_stop(self):
        """Emergency stop - immediate landing"""
        logger.warning("Emergency stop initiated!")
        self.running = False
        self.mission_active = False
        
        # Set land mode
        self.flight_controller.set_mode("LAND")
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        
        self.running = False
        self.mission_active = False
        
        if self.camera:
            self.camera.release()
        
        if self.video_writer:
            self.video_writer.release()
        
        cv2.destroyAllWindows()
        
        # Clean up ultrasonic sensor
        self.ultrasonic_sensor.cleanup()
        
        logger.info("Cleanup completed")

# Example usage
def main():
    """Main function demonstrating the autonomous drone system with grid-based detection"""
    drone = AutonomousDrone()
    
    try:
        # Initialize system
        if not drone.initialize():
            logger.error("System initialization failed")
            return
        
        # Define mission waypoints (example coordinates)
        # Replace with your actual GPS coordinates
        drone.add_waypoint(lat=-35.363261, lon=149.165230, alt=10.0)  # Waypoint 1
        drone.add_waypoint(lat=-35.363300, lon=149.165280, alt=10.0)  # Waypoint 2
        drone.add_waypoint(lat=-35.363350, lon=149.165320, alt=10.0)  # Waypoint 3
        drone.add_waypoint(lat=-35.363400, lon=149.165350, alt=10.0)  # Waypoint 4
        drone.add_waypoint(lat=-35.363261, lon=149.165230, alt=10.0)  # Return to start
        
        # Start mission with video recording
        logger.info("Starting autonomous mission with grid-based obstacle detection...")
        drone.start_mission(record_video=True)
        
    except KeyboardInterrupt:
        logger.info("Mission interrupted by user")
        drone.stop_mission()
        
    except Exception as e:
        logger.error(f"Mission failed: {e}")
        drone.emergency_stop()
        
    finally:
        drone.cleanup()

if __name__ == "__main__":
    main()

# Save this file as: autonomous_drone_grid.py
# Run with: python3 autonomous_drone_grid.py