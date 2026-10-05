import json
import time
import socket
import random
import datetime as dt
import paho.mqtt.client as mqtt
from pyais.encode import encode_dict

UDP_SOCK = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
OPENCPN_ADDR = ("127.0.0.1", 10110)

# Realistic background AIS traffic near Ezhimala / Lakshadweep approach
real_ships = {
    412345001: {"name": "MV KERALA STAR", "lat": 11.95, "lon": 75.30, "speed": 11.0, "heading": 200.0},
    412345002: {"name": "INS SUJATA",     "lat": 12.05, "lon": 75.10, "speed": 14.0, "heading": 260.0},
    412345003: {"name": "MT MALABAR COAST", "lat": 11.85, "lon": 75.40, "speed": 9.0,  "heading": 20.0},
}

# PNS Khyber ghost contact (spawned on command via attack/ais_control)
khyber = {"active": False, "mmsi": 863456789, "lat": 11.30, "lon": 72.90, "speed": 16.0, "heading": 95.0}


def step_ship(ship):
    dist_deg = (ship["speed"] * 0.514444 * 2.0) / 111000.0
    import math
    ship["lat"] += dist_deg * math.cos(math.radians(ship["heading"]))
    ship["lon"] += dist_deg * math.sin(math.radians(ship["heading"])) / math.cos(math.radians(ship["lat"]))


def broadcast_aivdm(mmsi, lat, lon, speed_kn, heading_deg):
    msg = {
        "type": 1,
        "mmsi": mmsi,
        "lat": lat,
        "lon": lon,
        "speed": speed_kn,
        "course": heading_deg,
        "heading": int(heading_deg),
        "status": 0,
        "turn": 0,
        "accuracy": 0,
        "maneuver": 0,
        "raim": False,
        "radio": 0,
    }
    try:
        sentences = encode_dict(msg, talker_id="AI")
        for s in sentences:
            UDP_SOCK.sendto((s + "\r\n").encode(), OPENCPN_ADDR)
    except Exception as e:
        print("AIVDM encode error:", e)


def on_message(client, userdata, msg):
    print("RECEIVED:", msg.topic, msg.payload.decode())
    try:
        payload = json.loads(msg.payload.decode())
    except json.JSONDecodeError:
        return
    if msg.topic == "attack/ais_control":
        action = payload.get("action", "spawn")
        khyber["active"] = (action == "spawn")
        if "lat" in payload: khyber["lat"] = payload["lat"]
        if "lon" in payload: khyber["lon"] = payload["lon"]
        if "speed" in payload: khyber["speed"] = payload["speed"]
        if "heading" in payload: khyber["heading"] = payload["heading"]
        print(f"PNS Khyber ghost contact -> active={khyber['active']}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
client.on_message = on_message
client.connect("localhost", 1883, 60)
client.subscribe("attack/ais_control")
client.loop_start()

print("Enemy ship / AIS simulator running.")
while True:
    contacts = []
    for mmsi, ship in real_ships.items():
        step_ship(ship)
        broadcast_aivdm(mmsi, ship["lat"], ship["lon"], ship["speed"], ship["heading"])
        contacts.append({"mmsi": mmsi, "name": ship["name"], "lat": ship["lat"], "lon": ship["lon"],
                          "sog": ship["speed"], "cog": ship["heading"], "status": "background"})

    if khyber["active"]:
        step_ship(khyber)
        broadcast_aivdm(khyber["mmsi"], khyber["lat"], khyber["lon"], khyber["speed"], khyber["heading"])
        contacts.append({"mmsi": khyber["mmsi"], "name": "PNS KHYBER (ghost)", "lat": khyber["lat"],
                          "lon": khyber["lon"], "sog": khyber["speed"], "cog": khyber["heading"],
                          "status": "ghost"})

    # Dashboard (frontend/index.html) expects ONE flat message per contact
    # on ais/contacts, not a batched {"contacts": [...]} envelope.
    for c in contacts:
        client.publish("ais/contacts", json.dumps(c))

    # Detector (detection/ais_detector.py) reads the same per-contact stream.
    print(f"Broadcast {len(contacts)} AIS contacts (khyber_active={khyber['active']})")
    time.sleep(2.0)
