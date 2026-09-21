import json
import time
import socket
import datetime
import paho.mqtt.client as mqtt

from geo_utils import project_forward
from gnss_source import get_gnss_stream
from nmea_utils import build_gga, build_rmc

ANCHOR_LAT, ANCHOR_LON = 12.0080, 75.1750

udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
OPENCPN_ADDR = ("127.0.0.1", 10110)

def send_to_opencpn(lat, lon, speed_knots, heading_deg):
    now = datetime.datetime.utcnow()
    time_str = now.strftime("%H%M%S.00")
    date_str = now.strftime("%d%m%y")
    gga = build_gga(lat, lon, time_str)
    rmc = build_rmc(lat, lon, time_str, date_str, speed_knots, heading_deg)
    udp_sock.sendto((gga + "\r\n").encode(), OPENCPN_ADDR)
    udp_sock.sendto((rmc + "\r\n").encode(), OPENCPN_ADDR)


class OwnShip:
    def __init__(self):
        self.lat = ANCHOR_LAT
        self.lon = ANCHOR_LON
        self.speed_knots = 10.0
        self.heading_deg = 45.0
        self.last_tick = time.time()
        self.attack_type = "none"
        self.attack_start = None

    def set_control(self, speed=None, heading=None):
        if speed is not None:
            self.speed_knots = speed
        if heading is not None:
            self.heading_deg = heading
        print(f"Control updated: speed={self.speed_knots} heading={self.heading_deg}")

    def set_attack(self, attack_type):
        self.attack_type = attack_type
        self.attack_start = time.time() if attack_type != "none" else None

    def step(self):
        now = time.time()
        dt = now - self.last_tick
        self.last_tick = now
        # guard against a huge dt (e.g. after the script was paused) causing a big jump
        dt = min(dt, 2.0)
        distance_m = self.speed_knots * 0.514444 * dt
        self.lat, self.lon = project_forward(self.lat, self.lon, self.heading_deg, distance_m)
        true_lat, true_lon = self.lat, self.lon

        gps_lat, gps_lon = true_lat, true_lon
        if self.attack_type == "jump" and self.attack_start:
            gps_lat += 0.01
            gps_lon -= 0.01
        elif self.attack_type == "drift" and self.attack_start:
            elapsed = now - self.attack_start
            gps_lat += min(elapsed * 0.00003, 0.02)
            gps_lon -= min(elapsed * 0.00002, 0.015)

        return true_lat, true_lon, gps_lat, gps_lon, now


def on_message(client, userdata, msg):
    print("RECEIVED:", msg.topic, msg.payload.decode())
    try:
        payload = json.loads(msg.payload.decode())
    except json.JSONDecodeError:
        print("Bad JSON, ignoring:", msg.payload.decode())
        return
    if msg.topic == "ship/control":
        ship.set_control(speed=payload.get("speed"), heading=payload.get("heading"))
    elif msg.topic == "attack/gps_control":
        ship.set_attack(payload.get("type", "none"))


ship = OwnShip()
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
client.on_message = on_message
client.connect("localhost", 1883, 60)
result1 = client.subscribe("ship/control")
result2 = client.subscribe("attack/gps_control")
print("Subscribe results:", result1, result2)
client.loop_start()

print("Own ship running.")
for gnss_meta in get_gnss_stream():
    true_lat, true_lon, gps_lat, gps_lon, ts = ship.step()
    base = {"speed": ship.speed_knots, "heading": ship.heading_deg,
            "hdop": gnss_meta["hdop"], "satellites": gnss_meta["satellites"], "timestamp": ts}
    client.publish("ship/true_position", json.dumps({**base, "lat": true_lat, "lon": true_lon}))
    client.publish("ship/gps_position", json.dumps({**base, "lat": gps_lat, "lon": gps_lon}))
    send_to_opencpn(true_lat, true_lon, ship.speed_knots, ship.heading_deg)
    print(f"lat={true_lat:.5f} lon={true_lon:.5f} speed={ship.speed_knots} heading={ship.heading_deg}")