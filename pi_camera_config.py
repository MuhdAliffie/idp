#!/usr/bin/env python3
"""
Raspberry Pi Camera Module configuration for the grid detection system
"""

import cv2
import numpy as np
import time
from picamera2 import Picamera2
from camera_grid_detection import EnhancedObstacleDetector, CameraGridSystem

class RaspberryPiCameraSetup:
    """Setup class for Raspberry Pi camera module"""
    
    def __init__(self, width=640, height=480, framerate=30):
        self.width = width
        self.height = height
        self.framerate = framerate
        self.picam2 = None
        
    def initialize_camera(self):
        """Initialize the Raspberry Pi camera"""
        try:
            print("Initializing Raspberry Pi Camera...")
            
            # Create Picamera2 instance
            self.picam2 = Picamera2()
            
            # Create camera configuration
            camera_config = self.picam2.create_preview_configuration(
                main={"format": 'XRGB8888', "size": (self.width, self.height)},
                controls={"FrameRate": self.framerate}
            )
            
            # Configure and start camera
            self.picam2.configure(camera_config)
            self.picam2.start()
            
            # Wait for camera to warm up
            time.sleep(2)
            
            print(f"Camera initialized: {self.width}x{self.height} @ {self.framerate}fps")
            return True
            
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False
    
    def capture_frame(self):
        """Capture a frame from the camera"""
        if self.picam2 is None:
            return None
            
        try:
            # Capture frame as numpy array
            frame = self.picam2.capture_array()
            
            # Convert from XRGB8888 to BGR for OpenCV
            if frame.shape[2] == 4:  # RGBA/XRGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            return frame
            
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
    
    def cleanup(self):
        """Clean up camera resources"""
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
            print("Camera cleaned up")

class PiGridDetectionDemo:
    """Grid detection demo specifically for Raspberry Pi"""
    
    def __init__(self, width=640, height=480, model_path='yolov8n.pt'):
        # Initialize camera
        self.camera_setup = RaspberryPiCameraSetup(width, height)
        
        # Initialize detector
        self.detector = EnhancedObstacleDetector(
            model_path=model_path,
            confidence_threshold=0.5,
            frame_width=width,
            frame_height=height
        )
        
        self.running = False
        
    def run_detection(self):
        """Run the detection loop with Pi camera"""
        print("Starting Raspberry Pi Grid Detection...")
        
        # Initialize camera
        if not self.camera_setup.initialize_camera():
            print("Failed to start camera!")
            return
        
        self.running = True
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.running:
                # Capture frame
                frame = self.camera_setup.capture_frame()
                if frame is None:
                    continue
                
                # Analyze frame with grid system
                analysis = self.detector.analyze_frame_with_grid(frame)
                
                # Display results (if display available)
                try:
                    cv2.imshow('Pi Grid Detection', analysis['frame'])
                    
                    # Exit on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except:
                    # No display available - just print status
                    pass
                
                # Print danger status
                if analysis['objects_in_danger']:
                    print(f"⚠️  DANGER: {len(analysis['objects_in_danger'])} objects in danger zones")
                    for obj_info in analysis['objects_in_danger']:
                        obj = obj_info['object']
                        pos = obj_info['grid_pos']
                        print(f"   - {obj['class']} at grid ({pos[0]},{pos[1]})")
                    
                    if analysis['danger_analysis']:
                        direction = analysis['danger_analysis']['primary']
                        print(f"   → Recommended action: Move {direction}")
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"📊 FPS: {fps:.1f}, Objects detected: {len(analysis['objects'])}")
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping detection...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.camera_setup.cleanup()
        cv2.destroyAllWindows()

# Alternative: Modify existing classes to work with Pi Camera
class PiEnhancedObstacleDetector(EnhancedObstacleDetector):
    """Enhanced detector modified for Raspberry Pi camera"""
    
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5, 
                 frame_width=640, frame_height=480):
        super().__init__(model_path, confidence_threshold, frame_width, frame_height)
        
        # Initialize Pi camera instead of USB camera
        self.camera_setup = RaspberryPiCameraSetup(frame_width, frame_height)
        self.camera_initialized = False
    
    def start_camera(self):
        """Start the Pi camera"""
        if not self.camera_initialized:
            self.camera_initialized = self.camera_setup.initialize_camera()
        return self.camera_initialized
    
    def get_frame(self):
        """Get frame from Pi camera"""
        if not self.camera_initialized:
            return None
        return self.camera_setup.capture_frame()
    
    def stop_camera(self):
        """Stop the Pi camera"""
        self.camera_setup.cleanup()
        self.camera_initialized = False

# Test camera functionality
def test_pi_camera():
    """Test Raspberry Pi camera basic functionality"""
    print("Testing Raspberry Pi Camera...")
    
    camera = RaspberryPiCameraSetup()
    
    if camera.initialize_camera():
        print("✅ Camera initialized successfully")
        
        # Test capture
        for i in range(5):
            frame = camera.capture_frame()
            if frame is not None:
                print(f"✅ Frame {i+1} captured: {frame.shape}")
                
                # Optionally save test image
                cv2.imwrite(f'test_frame_{i+1}.jpg', frame)
            else:
                print(f"❌ Failed to capture frame {i+1}")
            
            time.sleep(1)
        
        camera.cleanup()
        print("✅ Camera test completed")
    else:
        print("❌ Camera initialization failed")

if __name__ == "__main__":
    print("Raspberry Pi Camera Grid Detection System")
    print("=========================================")
    print("Choose an option:")
    print("1. Test camera functionality")
    print("2. Run grid detection demo")
    print("3. Run headless detection (no display)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        test_pi_camera()
    elif choice == "2":
        demo = PiGridDetectionDemo()
        demo.run_detection()
    elif choice == "3":
        # Headless version - just print detections
        print("Running headless detection...")
        demo = PiGridDetectionDemo()
        demo.run_detection()
    else:
        print("Invalid choice. Testing camera...")
        test_pi_camera()
