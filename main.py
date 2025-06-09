#!/usr/bin/env python3
"""
Complete Autonomous Drone System
Features:
- Mission Planning and Execution
- GPS Waypoint Navigation
- Obstacle Detection and Avoidance
- Return to Launch (RTL)
- Battery Monitoring
- Autonomous Takeoff and Landing
- Real-time Decision Making
"""

import cv2
import numpy as np
import time
import threading
import json
import math
from datetime import datetime
from pymavlink import mavutil
from ultralytics import YOLO
import logging
from .safety_systems import *
from .flight_modes import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_drone.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MissionPlanner:
    def __init__(self):
        """Enhanced mission planning with flight modes"""
        self.waypoints = []
        self.current_waypoint_index = 0
        self.mission_active = False
        self.home_position = None
        self.mission_type = None
        self.flight_modes = FlightModes()
        
    def load_mission_from_file(self, filename):
        """Load mission waypoints from JSON file"""
        try:
            with open(filename, 'r') as f:
                mission_data = json.load(f)
                self.waypoints = mission_data.get('waypoints', [])
                self.mission_type = mission_data.get('mission_type', 'custom')
                logger.info(f"Loaded {len(self.waypoints)} waypoints from {filename}")
                return True
        except Exception as e:
            logger.error(f"Failed to load mission: {e}")
            return False
    
    def create_mission_by_type(self, mission_type, **kwargs):
        """Create mission using predefined flight modes"""
        if mission_type == 'survey':
            self.waypoints = self.flight_modes.create_survey_mission(
                kwargs['start_lat'], kwargs['start_lon'],
                kwargs['width'], kwargs['height'],
                kwargs['altitude'], kwargs.get('spacing', 20)
            )
        elif mission_type == 'patrol':
            self.waypoints = self.flight_modes.create_patrol_mission(
                kwargs['perimeter_points'], kwargs['altitude'],
                kwargs.get('loops', 5), kwargs.get('patrol_delay', 10)
            )
        elif mission_type == 'search':
            self.waypoints = self.flight_modes.create_search_mission(
                kwargs['center_lat'], kwargs['center_lon'],
                kwargs['search_radius'], kwargs['altitude'],
                kwargs.get('pattern', 'spiral')
            )
        
        self.mission_type = mission_type
        logger.info(f"Created {mission_type} mission with {len(self.waypoints)} waypoints")
            
    def create_simple_mission(self, home_lat, home_lon, home_alt):
        """Create a simple rectangular mission pattern"""
        self.home_position = {'lat': home_lat, 'lon': home_lon, 'alt': home_alt}
        
        # Define waypoints in a rectangular pattern (adjust coordinates as needed)
        offset = 0.0001  # Approximately 10 meters
        altitude = home_alt + 20  # 20 meters above home
        
        self.waypoints = [
            {'lat': home_lat, 'lon': home_lon, 'alt': altitude, 'action': 'takeoff'},
            {'lat': home_lat + offset, 'lon': home_lon, 'alt': altitude, 'action': 'waypoint'},
            {'lat': home_lat + offset, 'lon': home_lon + offset, 'alt': altitude, 'action': 'waypoint'},
            {'lat': home_lat, 'lon': home_lon + offset, 'alt': altitude, 'action': 'waypoint'},
            {'lat': home_lat, 'lon': home_lon, 'alt': altitude, 'action': 'waypoint'},
            {'lat': home_lat, 'lon': home_lon, 'alt': home_alt, 'action': 'land'}
        ]
        
        logger.info(f"Created simple mission with {len(self.waypoints)} waypoints")
        
    def get_current_waypoint(self):
        """Get current waypoint"""
        if self.current_waypoint_index < len(self.waypoints):
            return self.waypoints[self.current_waypoint_index]
        return None
        
    def advance_waypoint(self):
        """Move to next waypoint"""
        if self.current_waypoint_index < len(self.waypoints) - 1:
            self.current_waypoint_index += 1
            logger.info(f"Advanced to waypoint {self.current_waypoint_index}")
            return True
        else:
            logger.info("Mission completed - all waypoints reached")
            self.mission_active = False
            return False
            
    def reset_mission(self):
        """Reset mission to beginning"""
        self.current_waypoint_index = 0
        self.mission_active = True
        logger.info("Mission reset to beginning")

class AutonomousFlightController:
    def __init__(self, connection_string='/dev/ttyUSB0', baudrate=57600):
        """Enhanced flight controller for autonomous operations"""
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.master = None
        
        # Flight state
        self.current_position = None
        self.current_attitude = None
        self.battery_voltage = 0.0
        self.battery_remaining = 100
        self.is_armed = False
        self.current_mode = 'UNKNOWN'
        
        # Autonomous control parameters
        self.waypoint_radius = 2.0  # meters
        self.cruise_speed = 5.0     # m/s
        self.takeoff_altitude = 10.0 # meters
        self.min_battery_voltage = 11.0  # volts
        self.rtl_triggered = False
        
        # Navigation state
        self.navigation_active = False
        self.obstacle_avoidance_active = False
        
        # Connect to flight controller
        self.connect()
        
    def connect(self):
        """Establish MAVLink connection"""
        try:
            self.master = mavutil.mavlink_connection(
                self.connection_string, 
                baud=self.baudrate
            )
            self.master.wait_heartbeat()
            logger.info("MAVLink connection established")
            
            # Request data streams
            self.request_data_streams()
            
            # Start telemetry monitoring thread
            self.telemetry_thread = threading.Thread(target=self.monitor_telemetry)
            self.telemetry_thread.daemon = True
            self.telemetry_thread.start()
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise
            
    def request_data_streams(self):
        """Request telemetry data streams"""
        streams = [
            (mavutil.mavlink.MAV_DATA_STREAM_POSITION, 10),
            (mavutil.mavlink.MAV_DATA_STREAM_ATTITUDE, 10),
            (mavutil.mavlink.MAV_DATA_STREAM_VFR_HUD, 5),
            (mavutil.mavlink.MAV_DATA_STREAM_SYSTEM_STATUS, 2)
        ]
        
        for stream_id, rate in streams:
            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                stream_id, rate, 1
            )
            
    def monitor_telemetry(self):
        """Monitor incoming telemetry data"""
        while True:
            try:
                msg = self.master.recv_match(blocking=False, timeout=0.1)
                if msg:
                    self.process_telemetry_message(msg)
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"Telemetry monitoring error: {e}")
                time.sleep(1.0)
                
    def process_telemetry_message(self, msg):
        """Process incoming telemetry messages"""
        msg_type = msg.get_type()
        
        if msg_type == 'GLOBAL_POSITION_INT':
            self.current_position = {
                'lat': msg.lat / 1e7,
                'lon': msg.lon / 1e7,
                'alt': msg.alt / 1000.0,
                'relative_alt': msg.relative_alt / 1000.0
            }
            
        elif msg_type == 'ATTITUDE':
            self.current_attitude = {
                'roll': msg.roll,
                'pitch': msg.pitch,
                'yaw': msg.yaw
            }
            
        elif msg_type == 'VFR_HUD':
            self.battery_voltage = msg.battery_remaining
            
        elif msg_type == 'SYS_STATUS':
            self.battery_remaining = msg.battery_remaining
            
        elif msg_type == 'HEARTBEAT':
            self.is_armed = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
            
    def wait_for_position_estimate(self, timeout=30):
        """Wait for valid GPS position"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.current_position and self.current_position['lat'] != 0:
                logger.info("GPS position acquired")
                return True
            time.sleep(0.5)
        logger.error("Failed to acquire GPS position")
        return False
        
    def set_mode(self, mode):
        """Change flight mode"""
        mode_mapping = {
            'STABILIZE': 0, 'ACRO': 1, 'ALT_HOLD': 2, 'AUTO': 3,
            'GUIDED': 4, 'LOITER': 5, 'RTL': 6, 'CIRCLE': 7,
            'LAND': 9, 'DRIFT': 11, 'SPORT': 13, 'FLIP': 14,
            'AUTOTUNE': 15, 'POSHOLD': 16, 'BRAKE': 17
        }
        
        if mode in mode_mapping:
            self.master.mav.set_mode_send(
                self.master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_mapping[mode]
            )
            logger.info(f"Mode changed to {mode}")
            self.current_mode = mode
            
    def arm_motors(self):
        """Arm the motors"""
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )
        logger.info("Arming motors...")
        
    def disarm_motors(self):
        """Disarm the motors"""
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        logger.info("Disarming motors...")
        
    def takeoff(self, altitude):
        """Autonomous takeoff"""
        logger.info(f"Starting autonomous takeoff to {altitude}m")
        
        # Set GUIDED mode
        self.set_mode('GUIDED')
        time.sleep(2)
        
        # Arm motors
        self.arm_motors()
        time.sleep(3)
        
        # Send takeoff command
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude
        )
        
        # Wait for takeoff completion
        return self.wait_for_altitude(altitude, timeout=60)
        
    def wait_for_altitude(self, target_altitude, tolerance=1.0, timeout=60):
        """Wait for drone to reach target altitude"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.current_position:
                current_alt = self.current_position['relative_alt']
                if abs(current_alt - target_altitude) < tolerance:
                    logger.info(f"Reached target altitude: {current_alt:.1f}m")
                    return True
                logger.info(f"Current altitude: {current_alt:.1f}m, target: {target_altitude}m")
            time.sleep(1.0)
        logger.error("Timeout waiting for altitude")
        return False
        
    def navigate_to_waypoint(self, waypoint):
        """Navigate to specific GPS waypoint"""
        logger.info(f"Navigating to waypoint: {waypoint}")
        
        self.master.mav.set_position_target_global_int_send(
            0,  # time_boot_ms
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000111111111000,  # type_mask (ignore velocity and acceleration)
            int(waypoint['lat'] * 1e7),
            int(waypoint['lon'] * 1e7),
            waypoint['alt'],
            0, 0, 0,  # vx, vy, vz
            0, 0, 0,  # afx, afy, afz
            0, 0      # yaw, yaw_rate
        )
        
    def wait_for_waypoint_reached(self, waypoint, timeout=120):
        """Wait for waypoint to be reached"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.current_position:
                distance = self.calculate_distance(
                    self.current_position['lat'], self.current_position['lon'],
                    waypoint['lat'], waypoint['lon']
                )
                
                if distance < self.waypoint_radius:
                    logger.info(f"Waypoint reached (distance: {distance:.1f}m)")
                    return True
                    
                logger.info(f"Distance to waypoint: {distance:.1f}m")
            time.sleep(2.0)
            
        logger.error("Timeout waiting for waypoint")
        return False
        
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two GPS coordinates"""
        R = 6371000  # Earth's radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon/2) * math.sin(delta_lon/2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance
        
    def land_at_current_position(self):
        """Land at current position"""
        logger.info("Landing at current position")
        self.set_mode('LAND')
        
    def return_to_launch(self):
        """Return to launch position"""
        logger.info("Returning to launch")
        self.set_mode('RTL')
        self.rtl_triggered = True
        
    def emergency_stop(self):
        """Emergency stop - hover in place"""
        logger.warning("EMERGENCY STOP")
        self.set_mode('LOITER')
        self.obstacle_avoidance_active = True
        
    def check_battery_level(self):
        """Check battery level and trigger RTL if low"""
        if self.battery_voltage > 0 and self.battery_voltage < self.min_battery_voltage:
            if not self.rtl_triggered:
                logger.warning(f"Low battery detected: {self.battery_voltage:.1f}V")
                self.return_to_launch()
                return False
        return True

class AutonomousDroneSystem:
    def __init__(self, camera_index=0, model_path='yolov8n.pt'):
        """Complete autonomous drone system with integrated safety"""
        self.flight_controller = AutonomousFlightController()
        self.mission_planner = MissionPlanner()
        self.obstacle_detector = ObstacleDetector(model_path)
        self.safety_system = SafetySystem(self.flight_controller)
        
        # Camera setup
        self.camera = cv2.VideoCapture(camera_index)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # System state
        self.running = False
        self.autonomous_mode = False
        self.mission_thread = None
        self.detection_thread = None
        self.safety_thread = None
        
        # Configuration
        self.config = {
            'geofence': {
                'center_lat': 0, 'center_lon': 0, 'radius': 500
            },
            'min_battery_voltage': 14.0,
            'critical_battery_voltage': 13.5,
            'max_flight_time': 1800
        }
        
    def load_configuration(self, config_file):
        """Load system configuration from file"""
        try:
            with open(config_file, 'r') as f:
                self.config.update(json.load(f))
            logger.info(f"Configuration loaded from {config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def initialize_system(self):
        """Initialize all subsystems with safety systems"""
        logger.info("Initializing autonomous drone system...")
        
        # Wait for GPS lock
        if not self.flight_controller.wait_for_position_estimate():
            logger.error("Failed to get GPS position")
            return False
        
        # Initialize safety systems
        if self.flight_controller.current_position:
            pos = self.flight_controller.current_position
            # Set geofence center to current position if not configured
            if self.config['geofence']['center_lat'] == 0:
                self.config['geofence']['center_lat'] = pos['lat']
                self.config['geofence']['center_lon'] = pos['lon']
        
        self.safety_system.initialize_safety_systems(self.config)
        
        # Create default mission if none loaded
        if not self.mission_planner.waypoints:
            pos = self.flight_controller.current_position
            self.mission_planner.create_simple_mission(
                pos['lat'], pos['lon'], pos['alt']
            )
            
        logger.info("System initialization complete")
        return True
    
    def create_mission(self, mission_type, **params):
        """Create mission using flight modes"""
        self.mission_planner.create_mission_by_type(mission_type, **params)
        logger.info(f"Created {mission_type} mission")
    
    def start_autonomous_flight(self):
        """Start autonomous flight with integrated safety monitoring"""
        if not self.initialize_system():
            return False
            
        logger.info("Starting autonomous flight with safety systems...")
        self.running = True
        self.autonomous_mode = True
        
        # Start safety monitoring first
        self.safety_system.flight_time_monitor.start_flight_timer()
        self.safety_thread = threading.Thread(target=self.safety_monitoring_loop)
        self.safety_thread.daemon = True
        self.safety_thread.start()
        
        # Start mission execution thread
        self.mission_thread = threading.Thread(target=self.execute_mission)
        self.mission_thread.daemon = True
        self.mission_thread.start()
        
        # Start obstacle detection thread
        self.detection_thread = threading.Thread(target=self.obstacle_detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        return True
    
    def safety_monitoring_loop(self):
        """Continuous safety monitoring"""
        while self.running and self.autonomous_mode:
            try:
                safety_status = self.safety_system.monitor_safety()
                
                # If any safety system fails, abort mission
                if not all(safety_status.values()):
                    logger.critical("Safety system failure detected")
                    self.autonomous_mode = False
                    break
                
                # Log safety status periodically
                if int(time.time()) % 30 == 0:  # Every 30 seconds
                    logger.info(f"Safety Status: {safety_status}")
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Safety monitoring error: {e}")
                time.sleep(5.0)
        
    def execute_mission(self):
        """Main mission execution loop"""
        self.mission_planner.reset_mission()
        
        while self.running and self.autonomous_mode:
            try:
                current_waypoint = self.mission_planner.get_current_waypoint()
                if not current_waypoint:
                    logger.info("Mission completed")
                    break
                    
                # Execute waypoint action
                if current_waypoint['action'] == 'takeoff':
                    if self.flight_controller.takeoff(current_waypoint['alt']):
                        self.mission_planner.advance_waypoint()
                    else:
                        logger.error("Takeoff failed")
                        break
                        
                elif current_waypoint['action'] == 'waypoint':
                    self.flight_controller.navigate_to_waypoint(current_waypoint)
                    if self.flight_controller.wait_for_waypoint_reached(current_waypoint):
                        self.mission_planner.advance_waypoint()
                    else:
                        logger.error("Failed to reach waypoint")
                        break
                        
                elif current_waypoint['action'] == 'land':
                    self.flight_controller.land_at_current_position()
                    self.mission_planner.advance_waypoint()
                    break
                    
                # Small delay between waypoints
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Mission execution error: {e}")
                break
                
        logger.info("Mission execution completed")
        self.autonomous_mode = False
        
    def obstacle_detection_loop(self):
        """Obstacle detection and avoidance loop"""
        consecutive_detections = 0
        detection_threshold = 3
        
        while self.running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                    
                # Only process if in autonomous mode and not already avoiding
                if (self.autonomous_mode and 
                    not self.flight_controller.obstacle_avoidance_active):
                    
                    obstacles = self.obstacle_detector.detect_obstacles(frame)
                    significant_obstacles = [
                        obs for obs in obstacles 
                        if obs['area'] > 5000  # Minimum area threshold
                    ]
                    
                    if significant_obstacles:
                        consecutive_detections += 1
                        if consecutive_detections >= detection_threshold:
                            self.handle_obstacle_detection(significant_obstacles, frame.shape)
                            consecutive_detections = 0
                    else:
                        consecutive_detections = 0
                        
                    # Resume mission if obstacle avoidance is complete
                    if (self.flight_controller.obstacle_avoidance_active and
                        not significant_obstacles):
                        self.resume_mission_after_avoidance()
                        
                time.sleep(0.1)  # 10Hz detection rate
                
            except Exception as e:
                logger.error(f"Obstacle detection error: {e}")
                time.sleep(1.0)
                
    def handle_obstacle_detection(self, obstacles, frame_shape):
        """Handle obstacle detection"""
        logger.warning("Obstacle detected - initiating avoidance")
        
        # Emergency stop
        self.flight_controller.emergency_stop()
        
        # Determine avoidance direction
        direction = self.obstacle_detector.analyze_obstacle_direction(
            obstacles, frame_shape
        )
        
        # Execute avoidance maneuver in separate thread
        avoidance_thread = threading.Thread(
            target=self.execute_avoidance_maneuver,
            args=(direction,)
        )
        avoidance_thread.start()
        
    def execute_avoidance_maneuver(self, direction):
        """Execute obstacle avoidance maneuver"""
        try:
            logger.info(f"Executing avoidance maneuver: {direction}")
            
            # Wait for stabilization
            time.sleep(2.0)
            
            # Set guided mode for manual control
            self.flight_controller.set_mode('GUIDED')
            time.sleep(1.0)
            
            # Execute avoidance based on direction
            avoidance_distance = 10.0  # meters
            
            if direction == 'front':
                # Move backward and up
                self.move_relative(0, -avoidance_distance, -3)
            elif direction == 'left':
                # Move right
                self.move_relative(avoidance_distance, 0, 0)
            elif direction == 'right':
                # Move left
                self.move_relative(-avoidance_distance, 0, 0)
            elif direction == 'top':
                # Move down and forward
                self.move_relative(0, avoidance_distance, 2)
            elif direction == 'bottom':
                # Move up and forward
                self.move_relative(0, avoidance_distance, -3)
                
            # Wait for avoidance completion
            time.sleep(5.0)
            
            logger.info("Avoidance maneuver completed")
            
        except Exception as e:
            logger.error(f"Avoidance maneuver failed: {e}")
            self.flight_controller.return_to_launch()
            
    def move_relative(self, dx, dy, dz):
        """Move relative to current position"""
        if not self.flight_controller.current_position:
            return
            
        # Calculate new position
        current_pos = self.flight_controller.current_position
        
        # Simple relative movement (for more accuracy, use proper coordinate transformation)
        new_lat = current_pos['lat'] + (dy / 111320.0)  # Rough conversion
        new_lon = current_pos['lon'] + (dx / (111320.0 * math.cos(math.radians(current_pos['lat']))))
        new_alt = current_pos['relative_alt'] + dz
        
        # Navigate to new position
        waypoint = {'lat': new_lat, 'lon': new_lon, 'alt': new_alt}
        self.flight_controller.navigate_to_waypoint(waypoint)
        self.flight_controller.wait_for_waypoint_reached(waypoint, timeout=30)
        
    def resume_mission_after_avoidance(self):
        """Resume mission after obstacle avoidance"""
        logger.info("Resuming mission after obstacle avoidance")
        self.flight_controller.obstacle_avoidance_active = False
        
    def safety_monitor(self):
        """Monitor safety parameters"""
        while self.running:
            try:
                # Check battery level
                if not self.flight_controller.check_battery_level():
                    logger.warning("Low battery - mission aborted")
                    self.autonomous_mode = False
                    break
                    
                # Check flight time
                if (self.flight_start_time and 
                    time.time() - self.flight_start_time > self.max_flight_time):
                    logger.warning("Maximum flight time reached")
                    self.flight_controller.return_to_launch()
                    self.autonomous_mode = False
                    break
                    
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Safety monitor error: {e}")
                time.sleep(5.0)
                
    def stop_autonomous_flight(self):
        """Stop autonomous flight operations"""
        logger.info("Stopping autonomous flight...")
        self.running = False
        self.autonomous_mode = False
        
        # Land immediately
        self.flight_controller.land_at_current_position()
        
        # Clean up
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()

class ObstacleDetector:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """YOLOv8 obstacle detector"""
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.obstacle_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe'
        ]
        
    def detect_obstacles(self, frame):
        """Detect obstacles in frame"""
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        obstacles = []
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                if class_name in self.obstacle_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    obstacle_info = {
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'center': [center_x, center_y],
                        'area': (x2 - x1) * (y2 - y1)
                    }
                    obstacles.append(obstacle_info)
                    
        return obstacles
        
    def analyze_obstacle_direction(self, obstacles, frame_shape):
        """Analyze obstacle direction"""
        if not obstacles:
            return None
            
        height, width = frame_shape[:2]
        frame_center_x, frame_center_y = width // 2, height // 2
        
        # Find largest obstacle
        largest_obstacle = max(obstacles, key=lambda x: x['area'])
        center_x, center_y = largest_obstacle['center']
        
        # Determine direction
        horizontal_threshold = width * 0.3
        vertical_threshold = height * 0.3
        
        if center_x < frame_center_x - horizontal_threshold:
            return 'left'
        elif center_x > frame_center_x + horizontal_threshold:
            return 'right'
        elif center_y < frame_center_y - vertical_threshold:
            return 'top'
        elif center_y > frame_center_y + vertical_threshold:
            return 'bottom'
        else:
            return 'front'

# Main execution with examples
if __name__ == "__main__":
    try:
        # Create autonomous drone system
        drone_system = AutonomousDroneSystem(
            camera_index=0,
            model_path='yolov8n.pt'
        )
        
        # Load configuration (optional)
        # drone_system.load_configuration('drone_config.json')
        
        # Example 1: Survey Mission
        # drone_system.create_mission('survey', 
        #     start_lat=40.1234567, start_lon=-74.1234567,
        #     width=100, height=100, altitude=120, spacing=20)
        
        # Example 2: Patrol Mission
        # perimeter = [
        #     {'lat': 40.1234567, 'lon': -74.1234567},
        #     {'lat': 40.1244567, 'lon': -74.1234567},
        #     {'lat': 40.1244567, 'lon': -74.1244567},
        #     {'lat': 40.1234567, 'lon': -74.1244567}
        # ]
        # drone_system.create_mission('patrol',
        #     perimeter_points=perimeter, altitude=120, loops=3)
        
        # Example 3: Search Mission
        # drone_system.create_mission('search',
        #     center_lat=40.1234567, center_lon=-74.1234567,
        #     search_radius=200, altitude=120, pattern='spiral')
        
        # Optional: Load mission from file
        # drone_system.mission_planner.load_mission_from_file('mission.json')
        
        # Start autonomous flight with integrated safety
        if drone_system.start_autonomous_flight():
            logger.info("Autonomous flight started successfully")
            
            # Monitor flight status
            try:
                while drone_system.autonomous_mode:
                    time.sleep(5.0)
                    
                    # Print comprehensive status
                    fc = drone_system.flight_controller
                    if fc.current_position:
                        logger.info(f"Status - Pos: {fc.current_position['lat']:.6f}, "
                                  f"{fc.current_position['lon']:.6f}, "
                                  f"Alt: {fc.current_position['relative_alt']:.1f}m, "
                                  f"Battery: {fc.battery_voltage:.1f}V, "
                                  f"Mode: {fc.current_mode}, "
                                  f"Waypoint: {drone_system.mission_planner.current_waypoint_index + 1}/"
                                  f"{len(drone_system.mission_planner.waypoints)}")
                        
                    # Check for emergency conditions
                    if drone_system.safety_system.emergency_procedures_active:
                        logger.warning("Emergency procedures active - monitoring closely")
                        
            except KeyboardInterrupt:
                logger.info("Flight interrupted by user")
                
        else:
            logger.error("Failed to start autonomous flight")
            
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        drone_system.stop_autonomous_flight()
        logger.info("System shutdown complete")