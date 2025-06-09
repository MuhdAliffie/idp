class FlightModes:
    """Autonomous flight mode implementations"""
    
    @staticmethod
    def create_survey_mission(start_lat, start_lon, width, height, altitude, spacing=20):
        """Create systematic survey pattern for mapping/inspection"""
        waypoints = []
        lines = int(height / spacing)
        
        # Add takeoff waypoint
        waypoints.append({
            'lat': start_lat, 'lon': start_lon, 'alt': altitude,
            'action': 'takeoff', 'speed': 5.0, 'delay': 3
        })
        
        for i in range(lines):
            lat_offset = (i * spacing) / 111320  # Convert meters to degrees
            
            if i % 2 == 0:  # Even lines: left to right
                wp1 = {
                    'lat': start_lat + lat_offset, 'lon': start_lon, 
                    'alt': altitude, 'action': 'waypoint', 'speed': 8.0, 'delay': 2
                }
                wp2 = {
                    'lat': start_lat + lat_offset, 'lon': start_lon + (width / 111320),
                    'alt': altitude, 'action': 'waypoint', 'speed': 8.0, 'delay': 2
                }
            else:  # Odd lines: right to left
                wp1 = {
                    'lat': start_lat + lat_offset, 'lon': start_lon + (width / 111320),
                    'alt': altitude, 'action': 'waypoint', 'speed': 8.0, 'delay': 2
                }
                wp2 = {
                    'lat': start_lat + lat_offset, 'lon': start_lon,
                    'alt': altitude, 'action': 'waypoint', 'speed': 8.0, 'delay': 2
                }
            
            waypoints.extend([wp1, wp2])
        
        # Add landing waypoint
        waypoints.append({
            'lat': start_lat, 'lon': start_lon, 'alt': start_lat,
            'action': 'land', 'speed': 2.0, 'delay': 0
        })
        
        return waypoints
    
    @staticmethod
    def create_patrol_mission(perimeter_points, altitude, loops=5, patrol_delay=10):
        """Create patrol mission around defined perimeter"""
        waypoints = []
        
        # Add takeoff
        if perimeter_points:
            waypoints.append({
                'lat': perimeter_points[0]['lat'], 'lon': perimeter_points[0]['lon'],
                'alt': altitude, 'action': 'takeoff', 'speed': 5.0, 'delay': 3
            })
        
        for loop in range(loops):
            for i, point in enumerate(perimeter_points):
                waypoint = {
                    'lat': point['lat'], 'lon': point['lon'], 'alt': altitude,
                    'action': 'waypoint', 'speed': 6.0,
                    'delay': patrol_delay if i == 0 else 3  # Longer delay at start point
                }
                waypoints.append(waypoint)
        
        # Return and land
        if perimeter_points:
            waypoints.append({
                'lat': perimeter_points[0]['lat'], 'lon': perimeter_points[0]['lon'],
                'alt': perimeter_points[0].get('alt', 100), 'action': 'land',
                'speed': 2.0, 'delay': 0
            })
        
        return waypoints
    
    @staticmethod
    def create_search_mission(center_lat, center_lon, search_radius, altitude, pattern='spiral'):
        """Create search pattern mission"""
        waypoints = []
        
        # Add takeoff at center
        waypoints.append({
            'lat': center_lat, 'lon': center_lon, 'alt': altitude,
            'action': 'takeoff', 'speed': 5.0, 'delay': 3
        })
        
        if pattern == 'spiral':
            num_points = 16
            spiral_turns = 4
            
            for turn in range(spiral_turns):
                for i in range(num_points):
                    angle = (2 * math.pi * i) / num_points
                    radius = search_radius * (turn + 1) / spiral_turns
                    
                    lat_offset = radius * math.cos(angle) / 111320
                    lon_offset = radius * math.sin(angle) / (111320 * math.cos(math.radians(center_lat)))
                    
                    waypoint = {
                        'lat': center_lat + lat_offset,
                        'lon': center_lon + lon_offset,
                        'alt': altitude, 'action': 'waypoint',
                        'speed': 5.0, 'delay': 5  # Longer delay for searching
                    }
                    waypoints.append(waypoint)
        
        # Return to center and land
        waypoints.append({
            'lat': center_lat, 'lon': center_lon, 'alt': 100,
            'action': 'land', 'speed': 2.0, 'delay': 0
        })
        
        return waypoints