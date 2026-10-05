import math

def nmea_checksum(sentence):
    """sentence should NOT include the leading '$' or trailing '*hh'"""
    cs = 0
    for ch in sentence:
        cs ^= ord(ch)
    return f"{cs:02X}"

def to_nmea_coord(decimal_deg, is_lat=True):
    hemi_pos = "N" if is_lat else "E"
    hemi_neg = "S" if is_lat else "W"
    hemi = hemi_pos if decimal_deg >= 0 else hemi_neg
    decimal_deg = abs(decimal_deg)
    degrees = int(decimal_deg)
    minutes = (decimal_deg - degrees) * 60
    if is_lat:
        coord_str = f"{degrees:02d}{minutes:07.4f}"
    else:
        coord_str = f"{degrees:03d}{minutes:07.4f}"
    return coord_str, hemi

def build_gga(lat, lon, time_str, satellites=8, hdop=0.9, altitude_m=5.0):
    lat_str, lat_hemi = to_nmea_coord(lat, is_lat=True)
    lon_str, lon_hemi = to_nmea_coord(lon, is_lat=False)
    body = (f"GPGGA,{time_str},{lat_str},{lat_hemi},{lon_str},{lon_hemi},"
            f"1,{satellites:02d},{hdop:.1f},{altitude_m:.1f},M,0.0,M,,")
    cs = nmea_checksum(body)
    return f"${body}*{cs}"

def build_rmc(lat, lon, time_str, date_str, speed_knots, heading_deg):
    lat_str, lat_hemi = to_nmea_coord(lat, is_lat=True)
    lon_str, lon_hemi = to_nmea_coord(lon, is_lat=False)
    body = (f"GPRMC,{time_str},A,{lat_str},{lat_hemi},{lon_str},{lon_hemi},"
            f"{speed_knots:.1f},{heading_deg:.1f},{date_str},,,A")
    cs = nmea_checksum(body)
    return f"${body}*{cs}"
