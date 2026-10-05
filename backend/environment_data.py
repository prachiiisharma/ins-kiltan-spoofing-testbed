import csv
import datetime as dt
from bisect import bisect_left

KMH_TO_MS = 1000.0 / 3600.0

def _find_header_row(path):
    with open(path, newline="", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip().lower().startswith("time,"):
                return i
    return 0

class EnvironmentData:
    """Loads Open-Meteo marine (ocean current) and weather (wind) CSV exports
    and gives nearest-hour (speed_m/s, direction_deg) lookups for the
    dead-reckoning drift model."""

    def __init__(self, marine_csv_path, weather_csv_path):
        self.current_times, self.current_speed_ms, self.current_dir_deg = self._load_marine(marine_csv_path)
        self.wind_times, self.wind_speed_ms, self.wind_dir_deg = self._load_weather(weather_csv_path)

    def _load_marine(self, path):
        header_row = _find_header_row(path)
        times, speeds, dirs = [], [], []
        with open(path, newline="", encoding="utf-8") as f:
            for _ in range(header_row):
                next(f)
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    t = dt.datetime.fromisoformat(row["time"])
                    spd_key = next(k for k in row if "current_velocity" in k or "ocean_current_velocity" in k)
                    dir_key = next(k for k in row if "current_direction" in k or "ocean_current_direction" in k)
                    speed_ms = float(row[spd_key]) * KMH_TO_MS
                    direction = float(row[dir_key])
                except (ValueError, StopIteration, KeyError):
                    continue
                times.append(t); speeds.append(speed_ms); dirs.append(direction)
        return times, speeds, dirs

    def _load_weather(self, path):
        header_row = _find_header_row(path)
        times, speeds, dirs = [], [], []
        with open(path, newline="", encoding="utf-8") as f:
            for _ in range(header_row):
                next(f)
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    t = dt.datetime.fromisoformat(row["time"])
                    spd_key = next(k for k in row if "windspeed" in k or "wind_speed" in k)
                    dir_key = next(k for k in row if "winddirection" in k or "wind_direction" in k)
                    speed_ms = float(row[spd_key]) * KMH_TO_MS
                    direction = float(row[dir_key])
                except (ValueError, StopIteration, KeyError):
                    continue
                times.append(t); speeds.append(speed_ms); dirs.append(direction)
        return times, speeds, dirs

    def _nearest(self, times, values, query_time):
        if not times:
            return 0.0
        idx = bisect_left(times, query_time)
        if idx <= 0:
            return values[0]
        if idx >= len(times):
            return values[-1]
        before, after = times[idx-1], times[idx]
        return values[idx-1] if (query_time - before) <= (after - query_time) else values[idx]

    def get_conditions(self, query_time):
        c_speed = self._nearest(self.current_times, self.current_speed_ms, query_time)
        c_dir = self._nearest(self.current_times, self.current_dir_deg, query_time)
        w_speed = self._nearest(self.wind_times, self.wind_speed_ms, query_time)
        w_dir = self._nearest(self.wind_times, self.wind_dir_deg, query_time)
        return c_speed, c_dir, w_speed, w_dir
