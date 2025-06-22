import serial
import time

PORT = '/dev/ttyACM0'
BAUDRATE = 115200

try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)  # Wait for the connection to establish
    
    
    print(f"Connected to {PORT} at {BAUDRATE} baud.")
    
    response = ser.readline().decode('utf-8').strip()
    
    if response:
        print(f"Received: {response}")
    else:
        print("No response received but connection is established.")
        
    ser.close()
except serial.SerialException as e:
    print(f"Error: {e}")
    
