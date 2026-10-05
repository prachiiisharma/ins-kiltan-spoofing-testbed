"""
Step 6: live spoofing detector - the piece that goes into your testbed hub.

It loads the model trained on MARSIM and runs the SAME feature code on any NMEA stream:

    from detector import SpoofDetector
    det = SpoofDetector("results/iforest_single/model.joblib", primary_rx="ownship")
    for event in det.feed("ownship", time.time(), nmea_line):
        if event["alarm"]:
            ...  # raise alert, switch to dead-reckoning fallback

Command line:
    python detector.py --model results/iforest_single/model.joblib --pcap data/marsim/<file>.pcap
    python detector.py --model results/iforest_single/model.joblib --serial COM5          # NEO-6M on USB-TTL
"""
import argparse
import math
import statistics
import time
from collections import deque

import joblib
import numpy as np

from nmea_features import NmeaFeatureExtractor, haversine_m

ROLL = 10


def _roll(values, fn, min_periods=3):
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(v) < min_periods:
        return None
    return fn(v)


class SpoofDetector:
    def __init__(self, model_path, primary_rx, secondary_rx=None, rx_spacing_m=None):
        b = joblib.load(model_path)
        self.model, self.features, self.fill = b["model"], b["features"], b["fill"]
        self.threshold, self.K, self.warmup = b["threshold"], b["K"], b["warmup_s"]
        if b["mode"] == "dual" and secondary_rx is None:
            raise ValueError("this model was trained in dual-receiver mode: pass secondary_rx")
        self.primary, self.secondary, self.spacing = primary_rx, secondary_rx, rx_spacing_m
        self.fx = NmeaFeatureExtractor()
        self.hist = deque(maxlen=ROLL)
        self.last_secondary = None
        self.dist_base = []
        self.t0 = None
        self.run = 0
        self.alarm_since = None

    def _row_features(self, row):
        self.hist.append(row)
        h = list(self.hist)
        row["dr_resid_m_r10"] = _roll([r.get("dr_resid_m") for r in h], statistics.fmean)
        row["speed_resid_kn_r10"] = _roll([r.get("speed_resid_kn") for r in h], statistics.fmean)
        row["clock_step_s_r10std"] = _roll([r.get("clock_step_s") for r in h], statistics.stdev)
        row["cn0_std_r10"] = _roll([r.get("cn0_std") for r in h], statistics.fmean)
        s = self.last_secondary
        if s is not None and abs(s["t"] - row["t"]) <= 1.1:
            d = haversine_m(row["lat"], row["lon"], s["lat"], s["lon"])
            if self.spacing is None and len(self.dist_base) < 10:
                self.dist_base.append(d)
            base = self.spacing if self.spacing is not None else statistics.median(self.dist_base)
            row["rx_dist_dev_m"] = abs(d - base)
            if row.get("sog_kn") is not None and s.get("sog_kn") is not None:
                row["rx_sog_diff_kn"] = abs(row["sog_kn"] - s["sog_kn"])
            if row.get("clock_offset_s") is not None and s.get("clock_offset_s") is not None:
                row["rx_clock_diff_s"] = abs(row["clock_offset_s"] - s["clock_offset_s"])
        x = []
        for f in self.features:
            v = row.get(f)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                v = self.fill.get(f, 0.0)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    v = 0.0
            x.append(v)
        return np.asarray([x], dtype=np.float32)

    def feed(self, rx_id, t, sentence):
        """Feed one NMEA sentence. Returns a list of evaluated epochs (usually 0 or 1)."""
        events = []
        for row in self.fx.feed(rx_id, t, sentence):
            if row["rx"] == self.secondary:
                self.last_secondary = row
                continue
            if row["rx"] != self.primary:
                continue
            if self.t0 is None:
                self.t0 = row["t"]
            x = self._row_features(row)
            t_rel = row["t"] - self.t0
            if t_rel < self.warmup:
                continue
            score = float(-self.model.score_samples(x)[0])
            anomalous = score > self.threshold
            self.run = self.run + 1 if anomalous else 0
            alarm = self.run >= self.K
            if alarm and self.alarm_since is None:
                self.alarm_since = row["t"]
            events.append({"t": row["t"], "t_rel": t_rel, "lat": row["lat"], "lon": row["lon"],
                           "score": score, "anomalous": anomalous, "alarm": alarm,
                           "new_alarm": alarm and self.alarm_since == row["t"]})
        return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pcap", help="replay a MARSIM pcap through the detector")
    ap.add_argument("--serial", help="read a live GPS on this serial port, e.g. COM5 or /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--primary-rx", default="192.168.0.10")
    ap.add_argument("--secondary-rx", default=None)
    ap.add_argument("--rx-spacing-m", type=float, default=None)
    a = ap.parse_args()

    if a.pcap:
        from pcap_reader import iter_nmea
        det = SpoofDetector(a.model, a.primary_rx, a.secondary_rx, a.rx_spacing_m)
        first = None
        for src, t, s in iter_nmea(a.pcap):
            for e in det.feed(src, t, s):
                if e["new_alarm"]:
                    first = e
        print(f"first alarm at t = {first['t_rel']:.0f} s (score {first['score']:.3f})" if first
              else "no alarm raised")
    elif a.serial:
        import serial
        det = SpoofDetector(a.model, a.serial, None, None)
        with serial.Serial(a.serial, a.baud, timeout=1) as port:
            print(f"listening on {a.serial} ... Ctrl+C to stop")
            while True:
                line = port.readline().decode("ascii", errors="ignore").strip()
                for e in det.feed(a.serial, time.time(), line):
                    flag = "ALARM" if e["alarm"] else ("anom " if e["anomalous"] else "ok   ")
                    print(f"{flag} t={e['t_rel']:6.0f}s score={e['score']:.3f} "
                          f"pos=({e['lat']:.5f},{e['lon']:.5f})")
    else:
        ap.error("give --pcap or --serial")


if __name__ == "__main__":
    main()
