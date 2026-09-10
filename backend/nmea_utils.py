def nmea_checksum(sentence):
    checksum = 0
    for char in sentence:
        checksum ^= ord(char)
    return format(checksum, "02X")

def to_nmea_coord(value, is_lat):
    degrees = int(abs(value))
    minutes = (abs(value) - degrees) * 60
    direction = ("N" if value >= 0 else "S") if is_lat else ("E" if value >= 0 else "W")
    if is_lat:
        return f"{degrees:02d}{minutes:07.4f}", direction
    else:
        return f"{degrees:03d}{minutes:07.4f}", direction

def build_gga(lat, lon, timestamp_str, satellites=8, hdop=0.9):
    lat_str, lat_dir = to_nmea_coord(lat, True)
    lon_str, lon_dir = to_nmea_coord(lon, False)
    body = f"GPGGA,{timestamp_str},{lat_str},{lat_dir},{lon_str},{lon_dir},1,{satellites:02d},{hdop:.1f},10.0,M,0.0,M,,"
    return f"${body}*{nmea_checksum(body)}"

def build_rmc(lat, lon, timestamp_str, date_str, speed_knots, heading_deg):
    lat_str, lat_dir = to_nmea_coord(lat, True)
    lon_str, lon_dir = to_nmea_coord(lon, False)
    body = f"GPRMC,{timestamp_str},A,{lat_str},{lat_dir},{lon_str},{lon_dir},{speed_knots:.1f},{heading_deg:.1f},{date_str},,"
    return f"${body}*{nmea_checksum(body)}"