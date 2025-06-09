#!/usr/bin/env python3
"""
Usage Examples for Autonomous Drone System
Shows how to implement different flight modes and safety systems
"""

from main import AutonomousDroneSystem
import json
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_system_config():
    """Create system configuration file"""
    config = {
        "system_name": "Autonomous Survey Drone",
        "geofence": {
            "center_lat": 0,  # Will be set to current position if 0
            "center_lon": 0,
            "radius": 500,
            "max_altitude": 150
        },
        "battery": {
            "min_battery_voltage": 14.0,
            "critical_battery_voltage": 13.5,
            "landing_voltage": 13.0
        },
        "flight_parameters": {
            "max_flight_time": 1800,  # 30 minutes
            "cruise_speed": 8.0,
            "climb_rate": 3.0,
            "descent_rate": 2.0,
            "waypoint_radius": 2.0
        },
        "safety": {
            "max_wind_speed": 15,
            "max_rain_intensity": 0,
            "emergency_landing_enabled": True,
            "rtl_altitude": 120
        }
    }
    
    with open('drone_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Created drone_config.json")

def example_survey_mission():
    """Example: Agricultural field survey"""
    logger.info("=== SURVEY MISSION EXAMPLE ===")
    
    # Create drone system
    drone_system = AutonomousDroneSystem()
    
    # Load configuration
    drone_system.load_configuration('drone_config.json')
    
    # Create survey mission
    # This will create a systematic grid pattern
    drone_system.create_mission('survey',
        start_lat=40.1234567,  # Replace with your coordinates
        start_lon=-74.1234567,
        width=100,    # 100 meters wide
        height=150,   # 150 meters long
        altitude=120, # 120 meters altitude
        spacing=20    # 20 meters between lines
    )
    
    # Start autonomous flight
    if drone_system.start_autonomous_flight():
        logger.info("Survey mission started")
        
        # Monitor mission progress
        while drone_system.autonomous_mode:
            time.sleep(10)
            
            # Print mission status
            mission = drone_system.mission_planner
            logger.info(f"Mission Progress: {mission.current_waypoint_index + 1}/"
                       f"{len(mission.waypoints)} waypoints completed")
            
            # Check safety status
            if drone_system.safety_system.emergency_procedures_active:
                logger.warning("Emergency procedures active!")
                break
                
    drone_system.stop_autonomous_flight()

def example_patrol_mission():
    """Example: Security patrol around perimeter"""
    logger.info("=== PATROL MISSION EXAMPLE ===")
    
    drone_system = AutonomousDroneSystem()
    drone_system.load_configuration('drone_config.json')
    
    # Define patrol perimeter (square pattern)
    perimeter_points = [
        {'lat': 40.1234567, 'lon': -74.1234567},  # Corner 1
        {'lat': 40.1244567, 'lon': -74.1234567},  # Corner 2
        {'lat': 40.1244567, 'lon': -74.1244567},  # Corner 3
        {'lat': 40.1234567, 'lon': -74.1244567}   # Corner 4
    ]
    
    # Create patrol mission
    drone_system.create_mission('patrol',
        perimeter_points=perimeter_points,
        altitude=80,        # Lower altitude for better detection
        loops=5,           # 5 complete loops
        patrol_delay=15    # 15 seconds at each corner
    )
    
    # Start patrol
    if drone_system.start_autonomous_flight():
        logger.info("Patrol mission started")
        
        # Monitor for detections
        while drone_system.autonomous_mode:
            time.sleep(5)
            
            # Log any obstacle detections
            # (In real implementation, you might want to save images/locations)
            
    drone_system.stop_autonomous_flight()

def example_search_mission():
    """Example: Search and rescue spiral pattern"""
    logger.info("=== SEARCH MISSION EXAMPLE ===")
    
    drone_system = AutonomousDroneSystem()
    drone_system.load_configuration('drone_config.json')
    
    # Last known position for search
    search_center_lat = 40.1234567
    search_center_lon = -74.1234567
    
    # Create spiral search pattern
    drone_system.create_mission('search',
        center_lat=search_center_lat,
        center_lon=search_center_lon,
        search_radius=200,  # 200 meter search radius
        altitude=60,        # Lower for better visibility
        pattern='spiral'
    )
    
    # Start search mission
    if drone_system.start_autonomous_flight():
        logger.info("Search mission started")
        
        # Monitor search progress
        detection_log = []
        
        while drone_system.autonomous_mode:
            time.sleep(8)
            
            # In real implementation, log any person detections
            # with GPS coordinates for ground teams
            
    drone_system.stop_autonomous_flight()

def example_custom_mission_from_file():
    """Example: Load custom mission from JSON file"""
    logger.info("=== CUSTOM MISSION FROM FILE ===")
    
    # Create a custom mission file
    custom_mission = {
        "mission_name": "Infrastructure Inspection",
        "mission_type": "custom",
        "waypoints": [
            {"lat": 40.1234567, "lon": -74.1234567, "alt": 100, "action": "takeoff", "speed": 5.0, "delay": 3},
            {"lat": 40.1240000, "lon": -74.1234567, "alt": 100, "action": "waypoint", "speed": 6.0, "delay": 10},
            {"lat": 40.1240000, "lon": -74.1240000, "alt": 100, "action": "waypoint", "speed": 6.0, "delay": 10},
            {"lat": 40.1234567, "lon": -74.1240000, "alt": 100, "action": "waypoint", "speed": 6.0, "delay": 10},
            {"lat": 40.1234567, "lon": -74.1234567, "alt": 50, "action": "land", "speed": 2.0, "delay": 0}
        ]
    }
    
    # Save mission to file
    with open('custom_mission.json', 'w') as f:
        json.dump(custom_mission, f, indent=2)
    
    # Load and execute mission
    drone_system = AutonomousDroneSystem()
    drone_system.load_configuration('drone_config.json')
    
    if drone_system.mission_planner.load_mission_from_file('custom_mission.json'):
        if drone_system.start_autonomous_flight():
            logger.info("Custom mission started")
            
            while drone_system.autonomous_mode:
                time.sleep(5)
                
        drone_system.stop_autonomous_flight()

def example_safety_system_testing():
    """Example: Test safety systems"""
    logger.info("=== SAFETY SYSTEM TESTING ===")
    
    drone_system = AutonomousDroneSystem()
    
    # Create restrictive safety configuration for testing
    test_config = {
        "geofence": {
            "center_lat": 40.1234567,
            "center_lon": -74.1234567,
            "radius": 50,  # Very small geofence for testing
            "max_altitude": 30
        },
        "battery": {
            "min_battery_voltage": 15.0,  # High threshold for testing
            "critical_battery_voltage": 14.5
        },
        "flight_parameters": {
            "max_flight_time": 300  # 5 minutes max for testing
        }
    }
    
    # Save test config
    with open('test_config.json', 'w') as f:
        json.dump(test_config, f, indent=2)
    
    drone_system.load_configuration('test_config.json')
    
    # Create simple test mission
    drone_system.create_mission('survey',
        start_lat=40.1234567,
        start_lon=-74.1234567,
        width=30, height=30, altitude=25, spacing=10
    )
    
    if drone_system.start_autonomous_flight():
        logger.info("Safety test mission started")
        
        # Monitor safety violations
        while drone_system.autonomous_mode:
            time.sleep(2)
            
            safety_status = drone_system.safety_system.monitor_safety()
            logger.info(f"Safety Status: {safety_status}")
            
            if drone_system.safety_system.emergency_procedures_active:
                logger.warning("Emergency procedures triggered!")
                break
                
    drone_system.stop_autonomous_flight()

def example_multi_mode_mission():
    """Example: Mission with multiple phases"""
    logger.info("=== MULTI-MODE MISSION EXAMPLE ===")
    
    drone_system = AutonomousDroneSystem()
    drone_system.load_configuration('drone_config.json')
    
    # Phase 1: Quick patrol for initial assessment
    logger.info("Phase 1: Initial patrol")
    perimeter = [
        {'lat': 40.1234567, 'lon': -74.1234567},
        {'lat': 40.1244567, 'lon': -74.1234567},
        {'lat': 40.1244567, 'lon': -74.1244567},
        {'lat': 40.1234567, 'lon': -74.1244567}
    ]
    
    drone_system.create_mission('patrol',
        perimeter_points=perimeter,
        altitude=100, loops=1, patrol_delay=5
    )
    
    if drone_system.start_autonomous_flight():
        # Wait for patrol completion
        while drone_system.autonomous_mode:
            time.sleep(2)
    
    # Small delay between phases
    time.sleep(5)
    
    # Phase 2: Detailed survey of area
    logger.info("Phase 2: Detailed survey")
    drone_system.create_mission('survey',
        start_lat=40.1234567, start_lon=-74.1234567,
        width=80, height=80, altitude=80, spacing=15
    )
    
    if drone_system.start_autonomous_flight():
        while drone_system.autonomous_mode:
            time.sleep(2)
    
    # Phase 3: Focused search of points of interest
    logger.info("Phase 3: Focused search")
    drone_system.create_mission('search',
        center_lat=40.1240000, center_lon=-74.1240000,
        search_radius=50, altitude=60
    )
    
    if drone_system.start_autonomous_flight():
        while drone_system.autonomous_mode:
            time.sleep(2)
    
    drone_system.stop_autonomous_flight()
    logger.info("Multi-phase mission completed")

if __name__ == "__main__":
    # Create configuration file first
    create_system_config()
    
    print("\nSelect mission type:")
    print("1. Survey Mission (Agricultural field mapping)")
    print("2. Patrol Mission (Security perimeter monitoring)")
    print("3. Search Mission (Search and rescue pattern)")
    print("4. Custom Mission (Load from file)")
    print("5. Safety System Testing")
    print("6. Multi-Mode Mission (Combined operations)")
    
    try:
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            example_survey_mission()
        elif choice == '2':
            example_patrol_mission()
        elif choice == '3':
            example_search_mission()
        elif choice == '4':
            example_custom_mission_from_file()
        elif choice == '5':
            example_safety_system_testing()
        elif choice == '6':
            example_multi_mode_mission()
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    
    print("Example completed")