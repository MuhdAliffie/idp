#!/usr/bin/env python3
"""
Autonomous Drone Obstacle Detection and Avoidance System
- YOLOv8n for computer vision obstacle detection
- Ultrasonic sensor for proximity detection
- SpeedyBee F405 flight controller communication
- Autonomous waypoint navigation with obstacle avoidance
"""

import cv2
import numpy as np
import time
import threading
import queue
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import serial
import struct
from ultralytics import YOLO
import RPi.GPIO as GPIO
from pymavlink import mavutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

class UltrasonicSensor:
    """Ultrasonic sensor interface for HC-SR04"""
    
    def __init__(self, trigger_pin: int, echo_pin: int, max_distance: float = 4.0):
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self.max_distance = max_distance
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(trigger_pin, GPIO.OUT)
        GPIO.setup(echo_pin, GPIO.IN)
        GPIO.output(trigger_pin, False)
        
    def get_distance(self) -> float:
        """Get distance measurement in meters"""
        try:
            # Send trigger pulse
            GPIO.output(self.trigger_pin, True)
            time.sleep(0.00001)  # 10µs pulse
            GPIO.output(self.trigger_pin, False)
            
            # Measure echo time
            start_time = time.time()
            timeout = start_time + 0.1  # 100ms timeout
            
            while GPIO.input(self.echo_pin) == 0:
                pulse_start = time.time()
                if pulse_start > timeout:
                    return self.max_distance
                    
            while GPIO.input(self.echo_pin) == 1:
                pulse_end = time.time()
                if pulse_end > timeout:
                    return self.max_distance
                    
            # Calculate distance (speed of sound = 343 m/s)
            pulse_duration = pulse_end - pulse_start
            distance = pulse_duration * 343 / 2
            
            return min(distance, self.max_distance)
            
        except Exception as e:
            logger.error(f"Ultrasonic sensor error: {e}")
            return self.max_distance

class VisionDetector:
    """YOLOv8n-based obstacle detection"""
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.obstacle_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 
                                'bus', 'train', 'truck', 'boat', 'bird', 'cat', 'dog']
        
    def detect_obstacles(self, frame: np.ndarray) -> List[ObstacleInfo]:
        """Detect obstacles in camera frame"""
        try:
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            obstacles = []
            
            if results and len(results) > 0:
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            # Get class name
                            class_id = int(box.cls[0])
                            class_name = self.model.names[class_id]
                            
                            if class_name in self.obstacle_classes:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                confidence = float(box.conf[0])
                                
                                # Determine obstacle direction based on position in frame
                                frame_center = frame.shape[1] // 2
                                obstacle_center = (x1 + x2) // 2
                                
                                if obstacle_center < frame_center - 50:
                                    direction = 'left'
                                elif obstacle_center > frame_center + 50:
                                    direction = 'right'
                                else:
                                    direction = 'front'
                                
                                # Estimate distance based on bounding box size (rough approximation)
                                bbox_area = (x2 - x1) * (y2 - y1)
                                estimated_distance = max(1.0, 10000 / bbox_area)  # Rough estimation
                                
                                obstacles.append(ObstacleInfo(
                                    distance=estimated_distance,
                                    direction=direction,
                                    confidence=confidence,
                                    bbox=(x1, y1, x2, y2)
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
                'ACRO': 1,
                'ALT_HOLD': 2,
                'AUTO': 3,
                'GUIDED': 4,
                'LOITER': 5,
                'RTL': 6,
                'CIRCLE': 7,
                'LAND': 9,
                'DRIFT': 11,
                'SPORT': 13,
                'FLIP': 14,
                'AUTOTUNE': 15,
                'POSHOLD': 16,
                'BRAKE': 17,
                'THROW': 18,
                'AVOID_ADSB': 19,
                'GUIDED_NOGPS': 20,
                'SMART_RTL': 21,
                'FLOWHOLD': 22,
                'FOLLOW': 23,
                'ZIGZAG': 24,
                'SYSTEMID': 25,
                'AUTOROTATE': 26
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
    """Main autonomous drone controller with obstacle avoidance"""
    
    def __init__(self):
        # Initialize components
        self.flight_controller = FlightController()
        self.vision_detector = VisionDetector()
        self.ultrasonic_sensor = UltrasonicSensor(trigger_pin=18, echo_pin=24)
        
        # Navigation parameters
        self.waypoints: List[Waypoint] = []
        self.current_waypoint_index = 0
        self.mission_active = False
        self.avoidance_active = False
        self.original_waypoint: Optional[Waypoint] = None
        
        # Obstacle detection parameters
        self.obstacle_detection_distance = 3.0  # meters
        self.avoidance_distance = 5.0  # meters
        self.avoidance_speed = 2.0  # m/s
        self.cruise_speed = 3.0  # m/s
        self.cruise_altitude = 10.0  # meters
        
        # Threading
        self.running = False
        self.detection_thread = None
        self.telemetry_thread = None
        
        # Camera
        self.camera = None
        
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
    
    def detect_obstacles(self) -> List[ObstacleInfo]:
        """Detect obstacles using both vision and ultrasonic sensor"""
        obstacles = []
        
        # Vision-based detection
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                vision_obstacles = self.vision_detector.detect_obstacles(frame)
                obstacles.extend(vision_obstacles)
        
        # Ultrasonic sensor detection
        ultrasonic_distance = self.ultrasonic_sensor.get_distance()
        if ultrasonic_distance < self.obstacle_detection_distance:
            obstacles.append(ObstacleInfo(
                distance=ultrasonic_distance,
                direction='front',
                confidence=1.0
            ))
        
        return obstacles
    
    def plan_avoidance_maneuver(self, obstacles: List[ObstacleInfo]) -> Tuple[float, float, float]:
        """Plan avoidance maneuver based on detected obstacles"""
        # Default movement (forward)
        vx, vy, vz = self.avoidance_speed, 0, 0
        
        # Analyze obstacles
        front_obstacles = [obs for obs in obstacles if obs.direction == 'front']
        left_obstacles = [obs for obs in obstacles if obs.direction == 'left']
        right_obstacles = [obs for obs in obstacles if obs.direction == 'right']
        
        # If obstacles detected in front
        if front_obstacles:
            closest_front = min(front_obstacles, key=lambda x: x.distance)
            
            if closest_front.distance < self.avoidance_distance:
                # Stop forward movement
                vx = 0
                
                # Choose avoidance direction
                if len(left_obstacles) < len(right_obstacles):
                    # Go left
                    vy = -self.avoidance_speed
                    logger.info("Avoiding obstacle: moving left")
                else:
                    # Go right
                    vy = self.avoidance_speed
                    logger.info("Avoiding obstacle: moving right")
                
                # Gain altitude if very close
                if closest_front.distance < 2.0:
                    vz = -1.0  # Climb
                    logger.info("Avoiding obstacle: climbing")
        
        return vx, vy, vz
    
    def navigation_loop(self):
        """Main navigation loop"""
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
                
                # Detect obstacles
                obstacles = self.detect_obstacles()
                
                if obstacles:
                    # Obstacle detected - enter avoidance mode
                    if not self.avoidance_active:
                        self.avoidance_active = True
                        self.original_waypoint = current_waypoint
                        logger.info("Obstacle detected - entering avoidance mode")
                    
                    # Plan and execute avoidance maneuver
                    vx, vy, vz = self.plan_avoidance_maneuver(obstacles)
                    self.flight_controller.set_velocity(vx, vy, vz)
                    
                else:
                    # No obstacles - normal navigation
                    if self.avoidance_active:
                        self.avoidance_active = False
                        logger.info("Obstacle cleared - resuming normal navigation")
                    
                    # Navigate to waypoint
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
    
    def start_mission(self) -> bool:
        """Start autonomous mission"""
        if not self.waypoints:
            logger.error("No waypoints defined")
            return False
        
        logger.info("Starting autonomous mission...")
        
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
        
        GPIO.cleanup()
        
        logger.info("Cleanup completed")

# Example usage
def main():
    """Main function demonstrating the autonomous drone system"""
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
        drone.add_waypoint(lat=-35.363261, lon=149.165230, alt=10.0)  # Return to start
        
        # Start mission
        logger.info("Starting autonomous mission with obstacle avoidance...")
        drone.start_mission()
        
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

# Save this file as: autonomous_drone.py
# Run with: python3 autonomous_drone.py