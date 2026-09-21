import csv
import bisect
from datetime import datetime

class EnvironmentData:
    """Loads Open-Meteo marine (current) and weather (wind) CSVs and
    serves the nearest-hour reading for any requested timestamp."""

    def __init__(self, marine_csv_path, weather_csv_path):
        self.current_times, self.current_speed_ms, self.current_dir_deg = self._load_marine(marine_csv_path)
        self.wind_times, self.wind_speed_ms, self.wind_dir_deg = self._load_weather(weather_csv_path)

    def _find_header_row(self, path, key_col="time"):
        with open(path, newline="") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith(key_col + ","):
                return i
        raise ValueError(f"Could not find header row starting with '{key_col},' in {path}")

    def _load_marine(self, path):
        header_row = self._find_header_row(path)
        times, speeds, dirs = [], [], []
        with open(path, newline="") as f:
            for _ in range(header_row):
                next(f)
            reader = csv.DictReader(f)
            for row in reader:
                t = datetime.fromisoformat(row["time"])
                # ocean_current_velocity column may be in km/h - convert to m/s
                vel_col = [k for k in row if k.startswith("ocean_current_velocity")][0]
                dir_col = [k for k in row if k.startswith("ocean_current_direction")][0]
                speed_kmh = float(row[vel_col])
                times.append(t)
                speeds.append(speed_kmh / 3.6)  # km/h -> m/s
                dirs.append(float(row[dir_col]))
        return times, speeds, dirs

    def _load_weather(self, path):
        header_row = self._find_header_row(path)
        times, speeds, dirs = [], [], []
        with open(path, newline="") as f:
            for _ in range(header_row):
                next(f)
            reader = csv.DictReader(f)
            for row in reader:
                t = datetime.fromisoformat(row["time"])
                speed_col = [k for k in row if k.startswith("wind_speed")][0]
                dir_col = [k for k in row if k.startswith("wind_direction")][0]
                speed_kmh = float(row[speed_col])
                times.append(t)
                speeds.append(speed_kmh / 3.6)
                dirs.append(float(row[dir_col]))
        return times, speeds, dirs

    def _nearest(self, times, values, query_time):
        idx = bisect.bisect_left(times, query_time)
        if idx == 0:
            return values[0]
        if idx >= len(times):
            return values[-1]
        before, after = times[idx - 1], times[idx]
        return values[idx - 1] if (query_time - before) <= (after - query_time) else values[idx]

    def get_conditions(self, query_time):
        """Returns (current_speed_ms, current_dir_deg, wind_speed_ms, wind_dir_deg)
        for the given datetime, using the nearest available hourly reading."""
        c_speed = self._nearest(self.current_times, self.current_speed_ms, query_time)
        c_dir = self._nearest(self.current_times, self.current_dir_deg, query_time)
        w_speed = self._nearest(self.wind_times, self.wind_speed_ms, query_time)
        w_dir = self._nearest(self.wind_times, self.wind_dir_deg, query_time)
        return c_speed, c_dir, w_speed, w_dir