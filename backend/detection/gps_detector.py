import json, sys, os, time
import paho.mqtt.client as mqtt
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from geo_utils import haversine_distance

BASE_THRESHOLD_M = 30
GROWTH_RATE = 4

last_true = None
last_gps = None
last_trusted_time = time.time()
currently_spoofed = False


def compute_threshold(now):
    elapsed = now - last_trusted_time
    return BASE_THRESHOLD_M + GROWTH_RATE * (elapsed ** 0.5)


def check_and_publish(client):
    global last_trusted_time, currently_spoofed
    if last_true is None or last_gps is None:
        return
    now = time.time()
    residual_m = haversine_distance(last_true["lat"], last_true["lon"], last_gps["lat"], last_gps["lon"])
    threshold_m = compute_threshold(now)
    spoofed_now = residual_m > threshold_m

    if not spoofed_now:
        last_trusted_time = now

    if spoofed_now != currently_spoofed:
        currently_spoofed = spoofed_now
        payload = {
            "status": "spoofed" if spoofed_now else "clean",
            "residual_m": round(residual_m, 1),
            "threshold_m": round(threshold_m, 1),
            "timestamp": now,
        }
        client.publish("alerts/gps_spoof", json.dumps(payload))
        print("ALERT:", payload)


def on_message(client, userdata, msg):
    global last_true, last_gps
    try:
        data = json.loads(msg.payload.decode())
    except json.JSONDecodeError:
        return
    if msg.topic == "ship/true_position":
        last_true = data
    elif msg.topic == "ship/gps_position":
        last_gps = data
    check_and_publish(client)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
client.on_message = on_message
client.connect("localhost", 1883, 60)
client.subscribe("ship/true_position")
client.subscribe("ship/gps_position")
print("GPS spoof detector (Layer 2) running...")
client.loop_forever()
