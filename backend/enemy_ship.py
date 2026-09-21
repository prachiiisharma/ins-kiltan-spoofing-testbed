import json
import time
import socket
import paho.mqtt.client as mqtt
from geo_utils import project_forward
from pyais.encode import encode_dict

OPENCPN_ADDR = ("127.0.0.1", 10110)
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send_ais_to_opencpn(mmsi, lat, lon, sog, cog):
    try:
        data = {
            "type": 1,
            "mmsi": mmsi,
            "lat": lat,
            "lon": lon,
            "speed": sog,
            "course": cog,
            "heading": int(cog),
            "status": 0,
            "turn": 0,
            "accuracy": 0,
            "maneuver": 0,
            "raim": False,
            "radio": 0,
            "repeat": 0,
            "second": 0,
        }
        sentences = encode_dict(data, talker_id="AI")
        for s in sentences:
            udp_sock.sendto((s + "\r\n").encode(), OPENCPN_ADDR)
    except Exception as e:
        print("AIVDM encode error (non-fatal):", e)


real_ships = [
    {"mmsi": 419123456, "name": "MV Kerala Star", "lat": 12.0250, "lon": 75.1900, "sog": 8.0, "cog": 120},
    {"mmsi": 419234567, "name": "INS Sujata", "lat": 12.0050, "lon": 75.2000, "sog": 12.0, "cog": 200},
    {"mmsi": 419345678, "name": "MT Malabar Coast", "lat": 12.0300, "lon": 75.1650, "sog": 6.0, "cog": 45},
]

enemy_ship = {
    "mmsi": 863456789,
    "name": "PNS Khyber",
    "lat": 10.5700, "lon": 72.6400,
    "sog": 16.0, "cog": 100,
    "active": False,
}


def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    if payload.get("type") == "spawn_ghost":
        enemy_ship["active"] = True
        print("PNS Khyber AIS spoofing activated")
    elif payload.get("type") == "stop_ghost":
        enemy_ship["active"] = False
        print("PNS Khyber AIS spoofing stopped")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
client.on_message = on_message
client.connect("localhost", 1883, 60)
client.subscribe("attack/ais_control")
client.loop_start()

print("Enemy ship module running. Broadcasting real background traffic.")
print("Send attack/ais_control {type: spawn_ghost} to activate PNS Khyber.")
print("Also broadcasting AIVDM to OpenCPN on UDP 10110.")

while True:
    for ship in real_ships:
        ship["lat"], ship["lon"] = project_forward(ship["lat"], ship["lon"], ship["cog"], ship["sog"] * 0.514 * 2)
        payload = {
            "mmsi": ship["mmsi"], "name": ship["name"],
            "lat": ship["lat"], "lon": ship["lon"],
            "sog": ship["sog"], "cog": ship["cog"],
            "source": "real", "timestamp": time.time()
        }
        client.publish("ais/contacts", json.dumps(payload))
        send_ais_to_opencpn(ship["mmsi"], ship["lat"], ship["lon"], ship["sog"], ship["cog"])

    if enemy_ship["active"]:
        enemy_ship["lat"], enemy_ship["lon"] = project_forward(
            enemy_ship["lat"], enemy_ship["lon"], enemy_ship["cog"], enemy_ship["sog"] * 0.514 * 2
        )
        payload = {
            "mmsi": enemy_ship["mmsi"], "name": enemy_ship["name"],
            "lat": enemy_ship["lat"], "lon": enemy_ship["lon"],
            "sog": enemy_ship["sog"], "cog": enemy_ship["cog"],
            "source": "ghost", "timestamp": time.time()
        }
        client.publish("ais/contacts", json.dumps(payload))
        send_ais_to_opencpn(enemy_ship["mmsi"], enemy_ship["lat"], enemy_ship["lon"], enemy_ship["sog"], enemy_ship["cog"])
        print(f"PNS Khyber broadcasting: lat={enemy_ship['lat']:.4f} lon={enemy_ship['lon']:.4f}")

    time.sleep(2)
