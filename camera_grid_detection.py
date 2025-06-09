import cv2
import numpy as np
import time
import threading
import json
import math
from datetime import datetime
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraGridSystem:
    """Camera grid system with danger zones for object detection"""
    
    def __init__(self, frame_width=640, frame_height=480, grid_rows=6, grid_cols=8):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        # Calculate grid cell dimensions
        self.cell_width = frame_width // grid_cols
        self.cell_height = frame_height // grid_rows
        
        # Define danger zones (center area of the grid)
        self.danger_zones = self._create_danger_zones()
        self.safe_zones = self._create_safe_zones()
        
        # Grid state tracking
        self.grid_state = np.zeros((grid_rows, grid_cols), dtype=int)  # 0=safe, 1=danger, 2=object
        self.object_positions = {}
        
        # Colors for visualization
        self.colors = {
            'safe': (0, 255, 0),      # Green
            'danger': (0, 0, 255),    # Red
            'object': (255, 0, 0),    # Blue
            'grid': (128, 128, 128)   # Gray
        }
        
    def _create_danger_zones(self):
        """Define danger zones in the center of the grid"""
        danger_zones = []
        
        # Central area is considered dangerous (where objects pose immediate threat)
        center_row_start = self.grid_rows // 3
        center_row_end = 2 * self.grid_rows // 3
        center_col_start = self.grid_cols // 3
        center_col_end = 2 * self.grid_cols // 3
        
        for row in range(center_row_start, center_row_end + 1):
            for col in range(center_col_start, center_col_end + 1):
                danger_zones.append((row, col))
                
        return danger_zones
    
    def _create_safe_zones(self):
        """Define safe zones (areas outside danger zones)"""
        safe_zones = []
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                if (row, col) not in self.danger_zones:
                    safe_zones.append((row, col))
                    
        return safe_zones
    
    def get_grid_cell(self, x, y):
        """Convert pixel coordinates to grid cell"""
        col = min(int(x // self.cell_width), self.grid_cols - 1)
        row = min(int(y // self.cell_height), self.grid_rows - 1)
        return row, col
    
    def get_cell_center(self, row, col):
        """Get pixel coordinates of grid cell center"""
        center_x = col * self.cell_width + self.cell_width // 2
        center_y = row * self.cell_height + self.cell_height // 2
        return center_x, center_y
    
    def is_danger_zone(self, row, col):
        """Check if grid cell is in danger zone"""
        return (row, col) in self.danger_zones
    
    def is_safe_zone(self, row, col):
        """Check if grid cell is in safe zone"""
        return (row, col) in self.safe_zones
    
    def update_grid_with_objects(self, objects):
        """Update grid state with detected objects"""
        # Reset grid state
        self.grid_state.fill(0)  # 0 = empty
        
        # Mark danger zones
        for row, col in self.danger_zones:
            self.grid_state[row, col] = 1  # 1 = danger zone
        
        # Mark object positions
        self.object_positions = {}
        objects_in_danger = []
        
        for i, obj in enumerate(objects):
            center_x, center_y = obj['center']
            row, col = self.get_grid_cell(center_x, center_y)
            
            # Store object position
            self.object_positions[i] = {
                'row': row, 'col': col,
                'class': obj['class'],
                'confidence': obj['confidence'],
                'bbox': obj['bbox']
            }
            
            # Mark grid cell as occupied
            self.grid_state[row, col] = 2  # 2 = object present
            
            # Check if object is in danger zone
            if self.is_danger_zone(row, col):
                objects_in_danger.append({
                    'object_id': i,
                    'object': obj,
                    'grid_pos': (row, col)
                })
        
        return objects_in_danger
    
    def find_safe_direction(self, objects_in_danger):
        """Analyze objects in danger zones and find safe movement direction"""
        if not objects_in_danger:
            return None
        
        # Analyze object distribution in danger zones
        occupied_cells = [(obj['grid_pos']) for obj in objects_in_danger]
        
        # Calculate the center of mass of dangerous objects
        center_row = sum(row for row, col in occupied_cells) / len(occupied_cells)
        center_col = sum(col for row, col in occupied_cells) / len(occupied_cells)
        
        # Find the safest direction to move
        grid_center_row = self.grid_rows // 2
        grid_center_col = self.grid_cols // 2
        
        # Determine avoidance direction
        if center_col > grid_center_col:
            horizontal_direction = 'left'
        else:
            horizontal_direction = 'right'
            
        if center_row > grid_center_row:
            vertical_direction = 'up'
        else:
            vertical_direction = 'down'
        
        # Priority: horizontal movement first, then vertical
        return {
            'primary': horizontal_direction,
            'secondary': vertical_direction,
            'object_center': (center_row, center_col),
            'safe_zones': self._find_nearest_safe_zones(center_row, center_col)
        }
    
    def _find_nearest_safe_zones(self, danger_row, danger_col):
        """Find nearest safe zones from danger position"""
        safe_zones_with_distance = []
        
        for row, col in self.safe_zones:
            distance = math.sqrt((row - danger_row)**2 + (col - danger_col)**2)
            safe_zones_with_distance.append({
                'position': (row, col),
                'distance': distance
            })
        
        # Sort by distance and return top 3
        safe_zones_with_distance.sort(key=lambda x: x['distance'])
        return safe_zones_with_distance[:3]
    
    def draw_grid(self, frame):
        """Draw grid overlay on frame"""
        frame_with_grid = frame.copy()
        
        # Draw grid lines
        for i in range(1, self.grid_cols):
            x = i * self.cell_width
            cv2.line(frame_with_grid, (x, 0), (x, self.frame_height), self.colors['grid'], 1)
        
        for i in range(1, self.grid_rows):
            y = i * self.cell_height
            cv2.line(frame_with_grid, (0, y), (self.frame_width, y), self.colors['grid'], 1)
        
        # Fill danger zones
        for row, col in self.danger_zones:
            x1 = col * self.cell_width
            y1 = row * self.cell_height
            x2 = x1 + self.cell_width
            y2 = y1 + self.cell_height
            
            # Semi-transparent red overlay for danger zones
            overlay = frame_with_grid.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), self.colors['danger'], -1)
            cv2.addWeighted(frame_with_grid, 0.7, overlay, 0.3, 0, frame_with_grid)
        
        # Draw grid cell labels
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x, y = self.get_cell_center(row, col)
                
                # Label danger zones
                if self.is_danger_zone(row, col):
                    cv2.putText(frame_with_grid, 'D', (x-5, y+5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Show grid coordinates
                cv2.putText(frame_with_grid, f'{row},{col}', (x-15, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return frame_with_grid
    
    def draw_objects_on_grid(self, frame, objects):
        """Draw detected objects on grid"""
        frame_with_objects = frame.copy()
        
        for i, obj in enumerate(objects):
            # Draw bounding box
            x1, y1, x2, y2 = obj['bbox']
            cv2.rectangle(frame_with_objects, (int(x1), int(y1)), (int(x2), int(y2)), 
                         self.colors['object'], 2)
            
            # Draw object center
            center_x, center_y = obj['center']
            cv2.circle(frame_with_objects, (int(center_x), int(center_y)), 5, 
                      self.colors['object'], -1)
            
            # Get grid position
            row, col = self.get_grid_cell(center_x, center_y)
            
            # Draw label with grid position
            label = f"{obj['class']} ({row},{col})"
            if self.is_danger_zone(row, col):
                label += " [DANGER]"
                text_color = (0, 0, 255)  # Red for danger
            else:
                label += " [SAFE]"
                text_color = (0, 255, 0)  # Green for safe
            
            cv2.putText(frame_with_objects, label, (int(x1), int(y1-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        
        return frame_with_objects

class EnhancedObstacleDetector:
    """Enhanced obstacle detector with grid-based analysis"""
    
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5, frame_width=640, frame_height=480):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Initialize grid system
        self.grid_system = CameraGridSystem(frame_width, frame_height)
        
        # Obstacle classes to detect
        self.obstacle_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'bottle', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Avoidance parameters
        self.consecutive_danger_detections = 0
        self.danger_threshold = 3  # Number of consecutive detections before action
        self.avoidance_active = False
        
    def detect_obstacles(self, frame):
        """Detect obstacles in frame using YOLO"""
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
    
    def analyze_frame_with_grid(self, frame):
        """Analyze frame with grid system for danger detection"""
        # Detect objects
        objects = self.detect_obstacles(frame)
        
        # Update grid with detected objects
        objects_in_danger = self.grid_system.update_grid_with_objects(objects)
        
        # Analyze danger situation
        danger_analysis = None
        if objects_in_danger:
            self.consecutive_danger_detections += 1
            
            if self.consecutive_danger_detections >= self.danger_threshold:
                danger_analysis = self.grid_system.find_safe_direction(objects_in_danger)
                danger_analysis['objects_in_danger'] = objects_in_danger
                
                logger.warning(f"DANGER DETECTED: {len(objects_in_danger)} objects in danger zones")
                for obj_info in objects_in_danger:
                    obj = obj_info['object']
                    pos = obj_info['grid_pos']
                    logger.warning(f"  - {obj['class']} at grid position ({pos[0]},{pos[1]})")
        else:
            self.consecutive_danger_detections = 0
        
        # Create visualization
        frame_with_grid = self.grid_system.draw_grid(frame)
        frame_with_objects = self.grid_system.draw_objects_on_grid(frame_with_grid, objects)
        
        return {
            'frame': frame_with_objects,
            'objects': objects,
            'objects_in_danger': objects_in_danger,
            'danger_analysis': danger_analysis,
            'grid_state': self.grid_system.grid_state.copy()
        }

class GridBasedDroneController:
    """Drone controller with grid-based obstacle avoidance"""
    
    def __init__(self, flight_controller, obstacle_detector):
        self.flight_controller = flight_controller
        self.obstacle_detector = obstacle_detector
        self.avoidance_active = False
        self.avoidance_start_time = None
        self.max_avoidance_time = 30  # seconds
        
    def process_camera_frame(self, frame):
        """Process camera frame and handle obstacle avoidance"""
        analysis = self.obstacle_detector.analyze_frame_with_grid(frame)
        
        # Handle danger detection
        if analysis['danger_analysis'] and not self.avoidance_active:
            self.initiate_avoidance(analysis['danger_analysis'])
        
        # Check if avoidance is complete
        elif self.avoidance_active and not analysis['objects_in_danger']:
            self.complete_avoidance()
        
        # Handle avoidance timeout
        elif (self.avoidance_active and 
              time.time() - self.avoidance_start_time > self.max_avoidance_time):
            logger.warning("Avoidance timeout - returning to launch")
            self.flight_controller.return_to_launch()
            self.avoidance_active = False
        
        return analysis
    
    def initiate_avoidance(self, danger_analysis):
        """Initiate obstacle avoidance maneuver"""
        logger.warning("Initiating grid-based obstacle avoidance")
        self.avoidance_active = True
        self.avoidance_start_time = time.time()
        
        # Emergency stop first
        self.flight_controller.emergency_stop()
        
        # Calculate avoidance direction
        primary_direction = danger_analysis['primary']
        secondary_direction = danger_analysis['secondary']
        
        # Execute avoidance in separate thread
        avoidance_thread = threading.Thread(
            target=self.execute_grid_avoidance,
            args=(primary_direction, secondary_direction, danger_analysis)
        )
        avoidance_thread.daemon = True
        avoidance_thread.start()
    
    def execute_grid_avoidance(self, primary_direction, secondary_direction, danger_analysis):
        """Execute grid-based avoidance maneuver"""
        try:
            logger.info(f"Executing avoidance: {primary_direction} (primary), {secondary_direction} (secondary)")
            
            # Wait for stabilization
            time.sleep(2.0)
            
            # Set guided mode
            self.flight_controller.set_mode('GUIDED')
            time.sleep(1.0)
            
            # Calculate avoidance distance based on grid analysis
            avoidance_distance = 15.0  # meters
            
            # Execute primary direction movement
            if primary_direction == 'left':
                self.move_relative(-avoidance_distance, 0, 0)
            elif primary_direction == 'right':
                self.move_relative(avoidance_distance, 0, 0)
            elif primary_direction == 'up':
                self.move_relative(0, 0, -5)  # Move up
            elif primary_direction == 'down':
                self.move_relative(0, 0, 3)   # Move down slightly
            
            # Wait and assess
            time.sleep(3.0)
            
            # If still in danger, try secondary direction
            if secondary_direction:
                logger.info(f"Executing secondary avoidance: {secondary_direction}")
                
                if secondary_direction == 'up':
                    self.move_relative(0, 0, -5)
                elif secondary_direction == 'down':
                    self.move_relative(0, 0, 3)
                elif secondary_direction == 'left':
                    self.move_relative(-10, 0, 0)
                elif secondary_direction == 'right':
                    self.move_relative(10, 0, 0)
            
            # Move forward slightly to clear the area
            self.move_relative(0, 5, 0)
            
            logger.info("Grid-based avoidance maneuver completed")
            
        except Exception as e:
            logger.error(f"Grid avoidance failed: {e}")
            self.flight_controller.return_to_launch()
    
    def move_relative(self, dx, dy, dz):
        """Move relative to current position"""
        if not self.flight_controller.current_position:
            return
        
        current_pos = self.flight_controller.current_position
        
        # Calculate new position
        new_lat = current_pos['lat'] + (dy / 111320.0)
        new_lon = current_pos['lon'] + (dx / (111320.0 * math.cos(math.radians(current_pos['lat']))))
        new_alt = max(current_pos['relative_alt'] + dz, 5)  # Minimum 5m altitude
        
        # Navigate to new position
        waypoint = {'lat': new_lat, 'lon': new_lon, 'alt': new_alt}
        self.flight_controller.navigate_to_waypoint(waypoint)
        self.flight_controller.wait_for_waypoint_reached(waypoint, timeout=20)
    
    def complete_avoidance(self):
        """Complete avoidance maneuver and resume mission"""
        logger.info("Objects cleared from danger zones - avoidance complete")
        self.avoidance_active = False
        self.avoidance_start_time = None
        
        # Resume normal flight operations
        self.flight_controller.obstacle_avoidance_active = False

# Example usage and testing
class GridDetectionDemo:
    """Demo class to test grid-based detection system"""
    
    def __init__(self):
        self.detector = EnhancedObstacleDetector()
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    def run_demo(self):
        """Run grid detection demo"""
        logger.info("Starting grid-based detection demo")
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # Analyze frame
                analysis = self.detector.analyze_frame_with_grid(frame)
                
                # Display results
                cv2.imshow('Grid-Based Object Detection', analysis['frame'])
                
                # Print danger status
                if analysis['objects_in_danger']:
                    print(f"DANGER: {len(analysis['objects_in_danger'])} objects in danger zones")
                    if analysis['danger_analysis']:
                        print(f"Recommended action: {analysis['danger_analysis']['primary']}")
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.camera.release()
            cv2.destroyAllWindows()

# Integration with existing drone system
class EnhancedAutonomousDroneSystem:
    """Enhanced autonomous drone system with grid-based detection"""
    
    def __init__(self, camera_index=0, model_path='yolov8n.pt'):
        # Initialize components (assume flight_controller exists)
        self.enhanced_detector = EnhancedObstacleDetector(model_path)
        
        # Camera setup
        self.camera = cv2.VideoCapture(camera_index)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 60)
        
        # Grid controller (would integrate with actual flight controller)
        # self.grid_controller = GridBasedDroneController(self.flight_controller, self.enhanced_detector)
        
        self.running = False
        self.detection_thread = None
        
        # FPS calculation variables
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.fps_update_interval = 30  # Update FPS every 30 frames
        
        # Statistics
        self.detection_stats = {
            'total_detections': 0,
            'danger_detections': 0,
            'avoidance_maneuvers': 0
        }
    
    def start_enhanced_detection(self):
        """Start enhanced grid-based detection system"""
        logger.info("Starting enhanced grid-based detection system")
        self.running = True
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.enhanced_detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        return True
    
    def enhanced_detection_loop(self):
        """Enhanced detection loop with grid analysis"""
        while self.running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # Analyze frame with grid system
                analysis = self.enhanced_detector.analyze_frame_with_grid(frame)
                
                # Calculate and display FPS
                self.frame_count += 1
                current_time = time.time()
                if self.frame_count >= self.fps_update_interval:
                    elapsed_time = current_time - self.fps_start_time
                    self.current_fps = self.frame_count / elapsed_time
                    logger.info(f"Current FPS: {self.current_fps:.2f}")
                    
                    # Reset for next interval
                    self.frame_count = 0
                    self.fps_start_time = current_time
                
                # Add FPS display to frame
                frame_with_fps = analysis['frame'].copy()
                fps_text = f"FPS: {self.current_fps:.1f}"
                cv2.putText(frame_with_fps, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add additional info
                info_text = f"Objects: {len(analysis['objects'])}"
                cv2.putText(frame_with_fps, info_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if analysis['objects_in_danger']:
                    danger_text = f"DANGER: {len(analysis['objects_in_danger'])} objects"
                    cv2.putText(frame_with_fps, danger_text, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Update statistics
                self.detection_stats['total_detections'] += len(analysis['objects'])
                if analysis['objects_in_danger']:
                    self.detection_stats['danger_detections'] += 1
                
                # Handle danger detection (integrate with actual flight controller)
                if analysis['danger_analysis']:
                    self.handle_grid_based_danger(analysis['danger_analysis'])
                
                # Optional: Save frame for analysis
                if analysis['objects_in_danger']:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"danger_detection_{timestamp}.jpg", frame_with_fps)
                
                # Display for monitoring (comment out for production)
                cv2.imshow('Enhanced Drone Detection', frame_with_fps)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.033)  # ~30Hz detection rate
                
            except Exception as e:
                logger.error(f"Enhanced detection error: {e}")
                time.sleep(1.0)
    
    def handle_grid_based_danger(self, danger_analysis):
        """Handle danger detection with grid-based analysis"""
        logger.warning("Grid-based danger detection triggered")
        
        # Extract avoidance information
        primary_direction = danger_analysis['primary']
        secondary_direction = danger_analysis['secondary']
        safe_zones = danger_analysis['safe_zones']
        
        # Log detailed analysis
        logger.info(f"Primary avoidance direction: {primary_direction}")
        logger.info(f"Secondary avoidance direction: {secondary_direction}")
        logger.info(f"Nearest safe zones: {len(safe_zones)}")
        
        # In a real implementation, this would trigger flight controller actions
        # self.grid_controller.initiate_avoidance(danger_analysis)
        
        self.detection_stats['avoidance_maneuvers'] += 1
    
    def get_detection_stats(self):
        """Get detection statistics"""
        return self.detection_stats.copy()
    
    def stop_enhanced_detection(self):
        """Stop enhanced detection system"""
        logger.info("Stopping enhanced detection system")
        self.running = False
        
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Demo of enhanced system
    print("Enhanced Grid-Based Drone Detection System")
    print("==========================================")
    
    # Option 1: Run basic grid detection demo
    print("1. Basic Grid Detection Demo")
    demo = GridDetectionDemo()
    demo.run_demo()
    
    # Option 2: Run enhanced system demo (uncomment to use)
    # print("2. Enhanced Autonomous System Demo")
    # enhanced_system = EnhancedAutonomousDroneSystem()
    # enhanced_system.start_enhanced_detection()
    # 
    # try:
    #     while True:
    #         time.sleep(5)
    #         stats = enhanced_system.get_detection_stats()
    #         print(f"Detection Stats: {stats}")
    # except KeyboardInterrupt:
    #     enhanced_system.stop_enhanced_detection()
