import time
import random

USE_REAL_GNSS = False   # flip to True once your u-blox module is connected

def read_real_gnss(serial_port="COM3", baud=9600):
    import serial
    import pynmea2
    ser = serial.Serial(serial_port, baud, timeout=1)
    while True:
        line = ser.readline().decode("ascii", errors="replace").strip()
        if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
            try:
                msg = pynmea2.parse(line)
                yield {
                    "hdop": float(msg.horizontal_dil) if msg.horizontal_dil else 1.0,
                    "satellites": int(msg.num_sats) if msg.num_sats else 0,
                    "timestamp": time.time(),
                }
            except pynmea2.ParseError:
                continue

def read_simulated_gnss():
    while True:
        yield {
            "hdop": round(random.uniform(0.7, 1.3), 2),
            "satellites": random.randint(6, 11),
            "timestamp": time.time(),
        }
        time.sleep(1)

def get_gnss_stream():
    return read_real_gnss() if USE_REAL_GNSS else read_simulated_gnss()