#!/usr/bin/env python3
"""
Test script for ultrasonic sensor on Raspberry Pi 5
Tests the GPIO functionality before integrating with the drone system
"""

import time
import sys
from gpiozero import DistanceSensor, Device
from gpiozero.pins.lgpio import LGPIOFactory

def test_ultrasonic_sensor(trigger_pin=18, echo_pin=24):
    """Test the ultrasonic sensor functionality"""
    
    print(f"Testing Ultrasonic Sensor on Raspberry Pi 5")
    print(f"Trigger Pin: GPIO {trigger_pin}")
    print(f"Echo Pin: GPIO {echo_pin}")
    print("-" * 50)
    
    try:
        # Set the pin factory for Pi 5
        Device.pin_factory = LGPIOFactory()
        
        # Create distance sensor
        sensor = DistanceSensor(
            echo=echo_pin,
            trigger=trigger_pin,
            max_distance=4.0,
            threshold_distance=0.1
        )
        
        print("Sensor initialized successfully!")
        print("Press Ctrl+C to stop\n")
        
        while True:
            # Get distance
            distance = sensor.distance
            
            if distance is None:
                print("No object detected (out of range)")
            else:
                # Convert to cm for easier reading
                distance_cm = distance * 100
                
                # Create visual bar
                bar_length = int(distance_cm / 2)
                bar = '#' * min(bar_length, 50)
                
                print(f"Distance: {distance_cm:6.1f} cm | {distance:4.2f} m | {bar}")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nTest stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure you have installed: sudo apt install python3-lgpio python3-gpiozero")
        print("2. Check your wiring:")
        print("   - VCC to 5V")
        print("   - GND to Ground")
        print(f"   - Trigger to GPIO {trigger_pin}")
        print(f"   - Echo to GPIO {echo_pin}")
        print("3. Make sure your user is in the gpio group: sudo usermod -a -G gpio $USER")
        print("4. Try running with sudo if permission issues persist")
    finally:
        try:
            sensor.close()
            print("Sensor cleaned up")
        except:
            pass

def test_gpio_access():
    """Test basic GPIO access on Pi 5"""
    print("Testing GPIO access on Raspberry Pi 5...")
    
    try:
        import lgpio
        chip = lgpio.gpiochip_open(0)
        lgpio.gpiochip_close(chip)
        print("✓ Direct lgpio access works!")
    except Exception as e:
        print(f"✗ lgpio error: {e}")
        
    try:
        from gpiozero import LED
        Device.pin_factory = LGPIOFactory()
        # Just test initialization, don't actually use it
        test_led = LED(25)
        test_led.close()
        print("✓ gpiozero with LGPIOFactory works!")
    except Exception as e:
        print(f"✗ gpiozero error: {e}")

if __name__ == "__main__":
    # First test basic GPIO access
    test_gpio_access()
    print("\n" + "="*50 + "\n")
    
    # Then test ultrasonic sensor
    # You can pass custom pins as arguments
    if len(sys.argv) == 3:
        trigger = int(sys.argv[1])
        echo = int(sys.argv[2])
        test_ultrasonic_sensor(trigger, echo)
    else:
        test_ultrasonic_sensor()  # Use default pins 18, 24
        
# Usage:
# python3 test_ultrasonic_pi5.py
# or with custom pins:
# python3 test_ultrasonic_pi5.py 23 24