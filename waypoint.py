import serial
import struct
import time

PORT = "/dev/ttyACM0"
BAUD = 115200

MSP_SET_WP = 209
MSP_WP_CLEAR_ALL = 210
MSP_WP_GETINFO = 211
MSP_WP = 118

def build_msp_packet(cmd_id, payload=b''):
    header = b'$M<'
    size = len(payload).to_bytes(1, 'little')
    cmd = cmd_id.to_bytes(1, 'little')
    checksum = (sum(payload) ^ cmd_id ^ len(payload)) & 0xFF
    return header + size + cmd + payload + bytes([checksum])

def send_waypoint(ser, index, lat, lon, alt_cm, nav_action=16, flags=0):
    payload = struct.pack('<BiiiHHBB',
        index,
        int(lat * 1e7),
        int(lon * 1e7),
        int(alt_cm),
        0,      # heading (0 = default)
        0,      # stay time (in 1/10s)
        nav_action,
        flags
    )
    ser.write(build_msp_packet(MSP_SET_WP, payload))
    time.sleep(0.1)

def read_response(ser, expected_cmd, timeout=2):
    start = time.time()
    buffer = b''
    while time.time() - start < timeout:
        if ser.in_waiting:
            buffer += ser.read(ser.in_waiting)
            if b'$M>' in buffer:
                idx = buffer.index(b'$M>')
                if len(buffer[idx:]) >= 6:
                    size = buffer[idx + 3]
                    if len(buffer[idx:]) >= size + 6:
                        cmd = buffer[idx + 4]
                        if cmd == expected_cmd:
                            payload = buffer[idx + 5:idx + 5 + size]
                            return payload
    return None

def get_waypoint_count(ser):
    ser.write(build_msp_packet(MSP_WP_GETINFO))
    payload = read_response(ser, MSP_WP_GETINFO)
    if payload:
        wp_count = payload[1]
        print(f"[INFO] Waypoint count on FC: {wp_count}")
        return wp_count
    print("[ERROR] Failed to get waypoint count.")
    return 0

def get_waypoint(ser, index):
    ser.write(build_msp_packet(MSP_WP, bytes([index])))
    payload = read_response(ser, MSP_WP)
    if payload and len(payload) >= 15:
        idx, lat, lon, alt = struct.unpack('<Biii', payload[:13])
        lat_deg = lat / 1e7
        lon_deg = lon / 1e7
        alt_m = alt / 100.0
        print(f"[WAYPOINT {idx}] Lat: {lat_deg:.6f}, Lon: {lon_deg:.6f}, Alt: {alt_m:.2f} m")
    else:
        print(f"[ERROR] Failed to read waypoint {index}")

# === MAIN ===

waypoints = [
    (0, 3.1234567, 101.1234567, 5000),
    (1, 3.1235000, 101.1235000, 6000),
    (2, 3.1235500, 101.1235500, 4000)
]

with serial.Serial(PORT, BAUD, timeout=2) as ser:
    time.sleep(2)

    print("[STEP 1] Clearing all waypoints...")
    ser.write(build_msp_packet(MSP_WP_CLEAR_ALL))
    time.sleep(1)

    print("[STEP 2] Sending waypoints...")
    for index, lat, lon, alt_cm in waypoints:
        send_waypoint(ser, index, lat, lon, alt_cm)
        print(f"  ✓ Sent WP{index}")

    print("[STEP 3] Verifying waypoints on FC...")
    count = get_waypoint_count(ser)
    for i in range(count):
        get_waypoint(ser, i)
