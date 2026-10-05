"""Small geodesy helpers. Local flat-earth (ENU) is accurate to well under 1 m
over the ~100 km scenario boxes used here."""
import numpy as np

R_EARTH = 6371000.0
M_PER_DEG_LAT = 111_320.0
KN = 0.514444  # m/s per knot


def m_per_deg_lon(lat_deg):
    return M_PER_DEG_LAT * np.cos(np.radians(lat_deg))


def enu_to_ll(x_east, y_north, lat0, lon0):
    lat = lat0 + np.asarray(y_north) / M_PER_DEG_LAT
    lon = lon0 + np.asarray(x_east) / m_per_deg_lon(lat0)
    return lat, lon


def ll_to_enu(lat, lon, lat0, lon0):
    y = (np.asarray(lat) - lat0) * M_PER_DEG_LAT
    x = (np.asarray(lon) - lon0) * m_per_deg_lon(lat0)
    return x, y


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def vel_to_sog_cog(ve, vn):
    sog = np.hypot(ve, vn)
    cog = (np.degrees(np.arctan2(ve, vn)) + 360) % 360
    return sog, cog
