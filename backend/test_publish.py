import paho.mqtt.client as mqtt
import time
import json

client = mqtt.Client()
client.connect("localhost", 1883, 60)

lat, lon = 12.0080, 75.1750

print("Publishing fake position every second. Press Ctrl+C to stop.")
try:
    while True:
        lat += 0.0002
        lon += 0.00015
        payload = json.dumps({"lat": lat, "lon": lon, "timestamp": time.time()})
        client.publish("ship/true_position", payload)
        print("Published:", payload)
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopped.")