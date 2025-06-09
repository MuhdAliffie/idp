class SafetySystem:
    """Comprehensive safety monitoring and management"""
    
    def __init__(self, drone_controller):
        self.drone_controller = drone_controller
        
        # Safety parameters
        self.geofence = None
        self.battery_monitor = BatteryMonitor()
        self.weather_monitor = WeatherMonitor()
        self.flight_time_monitor = FlightTimeMonitor()
        
        # Safety state
        self.safety_violations = []
        self.emergency_procedures_active = False
        
    def initialize_safety_systems(self, config):
        """Initialize all safety systems with configuration"""
        if 'geofence' in config:
            self.geofence = GeofenceMonitor(
                config['geofence']['center_lat'],
                config['geofence']['center_lon'],
                config['geofence']['radius']
            )
            
        self.battery_monitor.configure(
            config.get('min_battery_voltage', 14.0),
            config.get('critical_battery_voltage', 13.5)
        )
        
        self.flight_time_monitor.set_max_flight_time(
            config.get('max_flight_time', 1800)  # 30 minutes default
        )
        
    def monitor_safety(self):
        """Main safety monitoring loop"""
        safety_status = {
            'geofence': True,
            'battery': True,
            'weather': True,
            'flight_time': True,
            'system_health': True
        }
        
        # Check geofence
        if self.geofence and self.drone_controller.current_position:
            pos = self.drone_controller.current_position
            if not self.geofence.check_boundary(pos['lat'], pos['lon']):
                safety_status['geofence'] = False
                self.handle_geofence_violation()
        
        # Check battery
        battery_status = self.battery_monitor.check_battery(
            self.drone_controller.battery_voltage,
            self.drone_controller.battery_remaining
        )
        if battery_status != "OK":
            safety_status['battery'] = False
            self.handle_battery_emergency(battery_status)
        
        # Check flight time
        if not self.flight_time_monitor.check_flight_time():
            safety_status['flight_time'] = False
            self.handle_flight_time_limit()
        
        # Check weather conditions
        if not self.weather_monitor.check_flight_conditions():
            safety_status['weather'] = False
            self.handle_weather_emergency()
        
        return safety_status
    
    def handle_geofence_violation(self):
        """Handle geofence boundary violation"""
        logger.critical("GEOFENCE VIOLATION - Initiating emergency RTL")
        self.drone_controller.return_to_launch()
        self.emergency_procedures_active = True
    
    def handle_battery_emergency(self, battery_status):
        """Handle battery-related emergencies"""
        if battery_status == "EMERGENCY_LAND":
            logger.critical("CRITICAL BATTERY - Emergency landing initiated")
            self.drone_controller.land_at_current_position()
        elif battery_status == "RTL":
            logger.warning("LOW BATTERY - Return to launch initiated")
            self.drone_controller.return_to_launch()
        
        self.emergency_procedures_active = True
    
    def handle_flight_time_limit(self):
        """Handle maximum flight time exceeded"""
        logger.warning("FLIGHT TIME LIMIT - Return to launch initiated")
        self.drone_controller.return_to_launch()
    
    def handle_weather_emergency(self):
        """Handle adverse weather conditions"""
        logger.warning("ADVERSE WEATHER - Return to launch initiated")
        self.drone_controller.return_to_launch()

class GeofenceMonitor:
    """Geofencing safety system"""
    
    def __init__(self, center_lat, center_lon, radius):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius = radius
        
    def check_boundary(self, current_lat, current_lon):
        """Check if current position is within geofence"""
        distance = self.calculate_distance(
            self.center_lat, self.center_lon,
            current_lat, current_lon
        )
        
        if distance > self.radius:
            logger.warning(f"Geofence violation! Distance: {distance:.1f}m, Limit: {self.radius}m")
            return False
        return True
    
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
        return R * c

class BatteryMonitor:
    """Battery monitoring and management"""
    
    def __init__(self):
        self.low_voltage = 14.0
        self.critical_voltage = 13.5
        self.rtl_triggered = False
        
    def configure(self, low_voltage, critical_voltage):
        """Configure battery thresholds"""
        self.low_voltage = low_voltage
        self.critical_voltage = critical_voltage
        
    def check_battery(self, voltage, remaining_percent):
        """Check battery status and return action needed"""
        if voltage < self.critical_voltage:
            return "EMERGENCY_LAND"
        elif voltage < self.low_voltage and not self.rtl_triggered:
            self.rtl_triggered = True
            return "RTL"
        return "OK"

class WeatherMonitor:
    """Weather condition monitoring"""
    
    def __init__(self):
        self.max_wind_speed = 15  # m/s
        self.max_rain_intensity = 0  # mm/h
        
    def check_flight_conditions(self):
        """Check if weather conditions are safe for flight"""
        # This is a placeholder - integrate with actual weather data
        # You could use APIs like OpenWeatherMap or onboard sensors
        return True  # Assume good conditions for now

class FlightTimeMonitor:
    """Flight time monitoring"""
    
    def __init__(self):
        self.max_flight_time = 1800  # 30 minutes
        self.flight_start_time = None
        
    def set_max_flight_time(self, max_time):
        """Set maximum flight time in seconds"""
        self.max_flight_time = max_time
        
    def start_flight_timer(self):
        """Start flight time monitoring"""
        self.flight_start_time = time.time()
        
    def check_flight_time(self):
        """Check if flight time limit has been exceeded"""
        if self.flight_start_time:
            elapsed_time = time.time() - self.flight_start_time
            if elapsed_time > self.max_flight_time:
                logger.warning(f"Flight time limit exceeded: {elapsed_time/60:.1f} minutes")
                return False
        return True