import math

EARTH_RADIUS_M = 6371000

def haversine_distance(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))

def project_forward(lat, lon, bearing_deg, distance_m):
    R = EARTH_RADIUS_M
    bearing = math.radians(bearing_deg)
    lat1, lon1 = math.radians(lat), math.radians(lon)
    lat2 = math.asin(math.sin(lat1)*math.cos(distance_m/R) + math.cos(lat1)*math.sin(distance_m/R)*math.cos(bearing))
    lon2 = lon1 + math.atan2(math.sin(bearing)*math.sin(distance_m/R)*math.cos(lat1), math.cos(distance_m/R)-math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

def knots_to_mps(knots):
    return knots * 0.514444

def drift_offset(current_speed_ms, current_dir_deg, wind_speed_ms, wind_dir_deg, leeway_coefficient=0.03):
    wind_push_dir = (wind_dir_deg + 180) % 360
    def to_vector(speed, bearing_deg):
        rad = math.radians(bearing_deg)
        return speed * math.sin(rad), speed * math.cos(rad)
    cx, cy = to_vector(current_speed_ms, current_dir_deg)
    wx, wy = to_vector(wind_speed_ms * leeway_coefficient, wind_push_dir)
    total_x, total_y = cx + wx, cy + wy
    drift_speed = math.sqrt(total_x**2 + total_y**2)
    drift_bearing = math.degrees(math.atan2(total_x, total_y)) % 360
    return drift_speed, drift_bearing
