#!/usr/bin/env python3
"""
Version-compatible Raspberry Pi Camera setup for both libcamera v0.4 and v0.5
"""

import cv2
import numpy as np
import time
import subprocess
import sys
import os

class UniversalPiCamera:
    """Universal Pi Camera class that works with different libcamera versions"""
    
    def __init__(self, width=640, height=480, framerate=60):
        self.width = width
        self.height = height
        self.framerate = framerate
        self.camera = None
        self.method = None
        
    def detect_camera_method(self):
        """Detect which camera method to use"""
        methods = [
            self._try_picamera2,
            self._try_opencv_v4l2,
            self._try_legacy_picamera,
            self._try_libcamera_raw
        ]
        
        for method in methods:
            if method():
                return True
        
        print("❌ No compatible camera method found!")
        return False
    
    def _try_picamera2(self):
        """Try using picamera2 (modern method)"""
        try:
            from picamera2 import Picamera2
            print("🔍 Trying picamera2...")
            
            self.camera = Picamera2()
            
            # Try different configuration approaches for compatibility
            try:
                # Method 1: Modern picamera2 (v0.5+)
                config = self.camera.create_preview_configuration(
                    main={"format": 'XRGB8888', "size": (self.width, self.height)},
                    controls={"FrameRate": self.framerate}
                )
            except Exception:
                try:
                    # Method 2: Older picamera2 (v0.4)
                    config = self.camera.create_preview_configuration(
                        main={"format": 'XRGB8888', "size": (self.width, self.height)}
                    )
                except Exception:
                    # Method 3: Basic configuration
                    config = self.camera.create_preview_configuration()
            
            self.camera.configure(config)
            self.camera.start()
            time.sleep(2)
            
            # Test capture
            frame = self.camera.capture_array()
            if frame is not None:
                self.method = "picamera2"
                print("✅ picamera2 method working")
                return True
                
        except Exception as e:
            print(f"❌ picamera2 failed: {e}")
            if self.camera:
                try:
                    self.camera.stop()
                except:
                    pass
                self.camera = None
        
        return False
    
    def _try_opencv_v4l2(self):
        """Try using OpenCV with V4L2 backend"""
        try:
            print("🔍 Trying OpenCV V4L2...")
            
            # Try different device indices
            for device_id in [0, 1, 2]:
                try:
                    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    cap.set(cv2.CAP_PROP_FPS, self.framerate)
                    
                    # Test capture
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.camera = cap
                        self.method = "opencv"
                        print(f"✅ OpenCV V4L2 working on device {device_id}")
                        return True
                    cap.release()
                except Exception:
                    if 'cap' in locals():
                        cap.release()
                    continue
                        
        except Exception as e:
            print(f"❌ OpenCV V4L2 failed: {e}")
        
        return False
    
    def _try_legacy_picamera(self):
        """Try legacy picamera library"""
        try:
            import picamera
            import picamera.array
            print("🔍 Trying legacy picamera...")
            
            self.camera = picamera.PiCamera()
            self.camera.resolution = (self.width, self.height)
            self.camera.framerate = self.framerate
            time.sleep(2)
            
            # Test capture
            with picamera.array.PiRGBArray(self.camera) as stream:
                self.camera.capture(stream, format='bgr')
                if stream.array is not None:
                    self.method = "legacy"
                    print("✅ Legacy picamera working")
                    return True
                    
        except Exception as e:
            print(f"❌ Legacy picamera failed: {e}")
            if self.camera:
                try:
                    self.camera.close()
                except:
                    pass
                self.camera = None
        
        return False
    
    def _try_libcamera_raw(self):
        """Try using libcamera-vid with raw capture"""
        try:
            print("🔍 Trying libcamera-vid method...")
            
            # Test if libcamera-vid exists
            result = subprocess.run(['which', 'libcamera-vid'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.method = "libcamera_raw"
                print("✅ libcamera-vid method available")
                return True
                
        except Exception as e:
            print(f"❌ libcamera-vid failed: {e}")
        
        return False
    
    def capture_frame(self):
        """Capture frame using the detected method"""
        if self.method == "picamera2":
            return self._capture_picamera2()
        elif self.method == "opencv":
            return self._capture_opencv()
        elif self.method == "legacy":
            return self._capture_legacy()
        elif self.method == "libcamera_raw":
            return self._capture_libcamera_raw()
        else:
            return None
    
    def _capture_picamera2(self):
        """Capture using picamera2"""
        try:
            frame = self.camera.capture_array()
            if frame is not None and len(frame.shape) >= 3:
                # Convert RGBA/XRGB to BGR if needed
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame
        except Exception as e:
            print(f"Capture error: {e}")
        return None
    
    def _capture_opencv(self):
        """Capture using OpenCV"""
        try:
            ret, frame = self.camera.read()
            return frame if ret else None
        except Exception as e:
            print(f"OpenCV capture error: {e}")
        return None
    
    def _capture_legacy(self):
        """Capture using legacy picamera"""
        try:
            import picamera.array
            with picamera.array.PiRGBArray(self.camera) as stream:
                self.camera.capture(stream, format='bgr')
                return stream.array.copy()
        except Exception as e:
            print(f"Legacy capture error: {e}")
        return None
    
    def _capture_libcamera_raw(self):
        """Capture using libcamera-vid (fallback method)"""
        try:
            # This is a simplified version - you might need to implement
            # a more sophisticated capture method
            cmd = [
                'libcamera-still', '-o', '/tmp/capture.jpg',
                '--width', str(self.width), '--height', str(self.height),
                '--timeout', '1', '--nopreview'
            ]
            subprocess.run(cmd, capture_output=True)
            
            # Read the captured image
            frame = cv2.imread('/tmp/capture.jpg')
            if os.path.exists('/tmp/capture.jpg'):
                os.remove('/tmp/capture.jpg')
            return frame
        except Exception as e:
            print(f"libcamera-vid capture error: {e}")
        return None
    
    def cleanup(self):
        """Clean up camera resources"""
        if self.camera:
            try:
                if self.method == "picamera2":
                    self.camera.stop()
                    self.camera.close()
                elif self.method == "opencv":
                    self.camera.release()
                elif self.method == "legacy":
                    self.camera.close()
            except Exception as e:
                print(f"Cleanup error: {e}")
        
        self.camera = None
        print("🧹 Camera cleaned up")

# Modified demo class using universal camera
class CompatiblePiGridDemo:
    """Grid detection demo with universal camera compatibility"""
    
    def __init__(self, width=640, height=480):
        from camera_grid_detection import EnhancedObstacleDetectorWithDistance
        
        self.camera = UniversalPiCamera(width, height)
        self.detector = EnhancedObstacleDetectorWithDistance(
            model_path='yolov8n.pt',
            confidence_threshold=0.5,
            frame_width=width,
            frame_height=height
        )
        
    def run_detection(self):
        """Run detection with compatible camera setup"""
        print("🚀 Starting Compatible Pi Grid Detection...")
        
        # Detect and initialize camera
        if not self.camera.detect_camera_method():
            print("❌ No camera method worked!")
            return
        
        print(f"📷 Using camera method: {self.camera.method}")
        
        try:
            frame_count = 0
            while True:
                # Capture frame
                frame = self.camera.capture_frame()
                if frame is None:
                    print("⚠️ No frame captured, retrying...")
                    time.sleep(0.1)
                    continue
                
                # Analyze frame
                analysis = self.detector.analyze_frame_with_grid(frame)
                
                # Try to display (might not work in headless mode)
                try:
                    cv2.imshow('Compatible Pi Detection', analysis['frame'])
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except:
                    # Headless mode - just print status
                    pass
                
                # Print status every 30 frames
                frame_count += 1
                if frame_count % 30 == 0:
                    total_objects = len(analysis['objects'])
                    danger_objects = len(analysis['objects_in_danger'])
                    print(f"📊 Frame {frame_count}: {total_objects} objects, {danger_objects} in danger")
                
                # Print danger alerts
                if analysis['objects_in_danger']:
                    print(f"⚠️ DANGER: {len(analysis['objects_in_danger'])} objects in danger zones")
                
                time.sleep(0.1)  # Prevent overwhelming the system
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping detection...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.camera.cleanup()
        cv2.destroyAllWindows()

def test_camera_methods():
    """Test all camera methods to see which one works"""
    print("🔍 Testing Camera Methods...")
    camera = UniversalPiCamera()
    
    if camera.detect_camera_method():
        print(f"✅ Best method found: {camera.method}")
        
        # Test capture
        for i in range(3):
            frame = camera.capture_frame()
            if frame is not None:
                print(f"✅ Test capture {i+1}: {frame.shape}")
                cv2.imwrite(f'test_capture_{i+1}.jpg', frame)
            else:
                print(f"❌ Test capture {i+1} failed")
            time.sleep(1)
        
        camera.cleanup()
    else:
        print("❌ No working camera method found")

if __name__ == "__main__":
    print("🎯 Universal Pi Camera Grid Detection")
    print("=====================================")
    
    choice = input("Choose:\n1. Test camera methods\n2. Run detection\n\nChoice: ")
    
    if choice == "1":
        test_camera_methods()
    else:
        demo = CompatiblePiGridDemo()
        demo.run_detection()
