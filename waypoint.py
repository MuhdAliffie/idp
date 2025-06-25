import serial
import struct
import time

# === Configuration ===
PORT = "/dev/ttyACM0"
BAUD = 115200
WAYPOINTS = [
    (3.1234567, 101.1234567, 5000),   # 50.00m
    (3.1235000, 101.1235000, 6000),   # 60.00m
    (3.1235500, 101.1235500, 4000)    # 40.00m
]

# === MSP Constants ===
MSP_SET_WP = 209
MSP_WP_CLEAR_ALL = 210
MSP_WP_GETINFO = 211
MSP_WP = 118
MSP_SET_MISSION_CONFIG = 217

def build_msp_packet(cmd_id, payload=b''):
    header = b'$M<'
    size = len(payload).to_bytes(1, 'little')
    cmd = cmd_id.to_bytes(1, 'little')
    checksum = (sum(payload) ^ cmd_id ^ len(payload)) & 0xFF
    return header + size + cmd + payload + bytes([checksum])

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

def clear_waypoints(ser):
    print("[1] Clearing existing waypoints...")
    ser.write(build_msp_packet(MSP_WP_CLEAR_ALL))
    time.sleep(0.5)

def send_waypoint(ser, index, lat, lon, alt_cm):
    payload = struct.pack('<BiiiHHBB',
        index,
        int(lat * 1e7),
        int(lon * 1e7),
        int(alt_cm),
        0,      # heading
        0,      # stay time
        16,     # nav_action = NAV_WAYPOINT
        0       # flags
    )
    ser.write(build_msp_packet(MSP_SET_WP, payload))
    time.sleep(0.1)
    print(f"   ✓ Sent WP{index}: {lat:.6f}, {lon:.6f}, {alt_cm / 100:.2f} m")

def set_mission_config(ser, total_wp):
    payload = struct.pack('<BB', total_wp, 0)  # waypoints count, repeat = 0
    ser.write(build_msp_packet(MSP_SET_MISSION_CONFIG, payload))
    time.sleep(0.1)
    print("[3] Sent mission config.")

def get_waypoint_count(ser):
    ser.write(build_msp_packet(MSP_WP_GETINFO))
    payload = read_response(ser, MSP_WP_GETINFO)
    if payload:
        count = payload[1]
        print(f"[4] FC reports {count} waypoints.")
        return count
    print("[ERROR] Could not read waypoint count.")
    return 0

def get_waypoint(ser, index):
    ser.write(build_msp_packet(MSP_WP, bytes([index])))
    payload = read_response(ser, MSP_WP)
    if payload and len(payload) >= 13:
        idx, lat, lon, alt = struct.unpack('<Biii', payload[:13])
        lat_deg = lat / 1e7
        lon_deg = lon / 1e7
        alt_m = alt / 100.0
        print(f"   WP{idx}: {lat_deg:.6f}, {lon_deg:.6f}, {alt_m:.2f} m")
    else:
        print(f"[ERROR] Failed to read WP{index}")

# === MAIN EXECUTION ===

with serial.Serial(PORT, BAUD, timeout=2) as ser:
    time.sleep(2)
    clear_waypoints(ser)

    print("[2] Sending waypoints...")
    for i, (lat, lon, alt_cm) in enumerate(WAYPOINTS):
        send_waypoint(ser, i, lat, lon, alt_cm)

    set_mission_config(ser, total_wp=len(WAYPOINTS))

    print("[4] Verifying waypoints...")
    count = get_waypoint_count(ser)
    for i in range(count):
        get_waypoint(ser, i)
