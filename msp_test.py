import serial
import time
import struct

PORT = '/dev/ttyACM0'
BAUDRATE = 115200

def build_msp_packet(cmd_id, payload=b''):
    header = b'$M<'
    size = len(payload).to_bytes(1, 'little')
    cmd = cmd_id.to_bytes(1, 'little')
    checksum = (sum(payload) ^ cmd_id ^ len(payload)) & 0xFF
    return header + size + cmd + payload + bytes([checksum])

def parse_msp_response(data):
    if not data.startswith(b'$M>'):
        return None, None
    payload_size = data[3]
    cmd = data[4]
    payload = data[5:5 + payload_size]
    
    if len(payload) != payload_size:
        print("Incomplete payload:", payload)
        return None

    return cmd, payload

with serial.Serial(PORT, BAUDRATE, timeout=2) as ser:
    
    time.sleep(2)  # Wait for the connection to establish
    
    msp_altitude = build_msp_packet(109)  # MSP_ALTITUDE command
    
    for attempt in range(3):
        print(f"\nAttempt {attempt +1}")
        ser.reset_input_buffer
        ser.write(msp_altitude)
        time.sleep(0.1)  # Wait for the response
        response = ser.read(32)
        print(f"Response: {response}")
    
    # msp_analog = build_msp_packet(110)  # MSP_ANALOG command
    # ser.write
    # response = ser.read(32) # Read up to 32 bytes
    result = parse_msp_response(response)
    if result:
        cmd, payload = result
        # vbat = payload[109]
        
        estalt_cm = struct.unpack('<i', payload[0:4])[0]
        estalt_cm = estalt_cm / 100.0  # Convert to cm
        print(f"Estimated Altitude: {estalt_cm} cm")
        # print (f"Battery Voltage: {vbat} V")
    else:
        print("No valid MSP response received.")