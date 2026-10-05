import json, sys, os, time
import paho.mqtt.client as mqtt
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from geo_utils import haversine_distance

VALID_MID_PREFIXES = {"419", "412", "413"}  # India-registered MID prefixes
MAX_PLAUSIBLE_SPEED_KN = 40

contact_history = {}
alert_state = {}


def check_contact(client, c):
    mmsi = str(c["mmsi"])
    reasons = []

    mid = mmsi[:3]
    if mid not in VALID_MID_PREFIXES:
        reasons.append("invalid_mid_prefix")

    now = time.time()
    prev = contact_history.get(mmsi)
    contact_history[mmsi] = {"lat": c["lat"], "lon": c["lon"], "t": now}
    if prev is not None:
        dt_s = max(now - prev["t"], 0.1)
        dist_m = haversine_distance(prev["lat"], prev["lon"], c["lat"], c["lon"])
        implied_speed_kn = (dist_m / dt_s) / 0.514444
        if implied_speed_kn > MAX_PLAUSIBLE_SPEED_KN:
            reasons.append("implausible_kinematic_speed")

    is_spoofed = len(reasons) > 0
    was_spoofed = alert_state.get(mmsi, False)
    if is_spoofed != was_spoofed:
        alert_state[mmsi] = is_spoofed
        # Dashboard (frontend/index.html) checks data.status === 'ghost' and
        # keys contactData[data.mmsi] by mmsi, so the mmsi type must match
        # what it stored from ais/contacts (a JS number for a numeric MMSI).
        payload = {
            "mmsi": c["mmsi"], "name": c.get("name", ""),
            "status": "ghost" if is_spoofed else "normal",
            "reasons": reasons, "timestamp": now,
        }
        client.publish("alerts/ais_spoof", json.dumps(payload))
        print("ALERT:", payload)


def on_message(client, userdata, msg):
    try:
        c = json.loads(msg.payload.decode())
    except json.JSONDecodeError:
        return
    if "mmsi" in c and "lat" in c and "lon" in c:
        check_contact(client, c)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
client.on_message = on_message
client.connect("localhost", 1883, 60)
client.subscribe("ais/contacts")
print("AIS spoof detector (Layer 2b) running...")
client.loop_forever()
