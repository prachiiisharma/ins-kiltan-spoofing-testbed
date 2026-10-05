import json, sys
import paho.mqtt.publish as publish

# Usage: python send_ais_attack.py [spawn|withdraw]
action = sys.argv[1] if len(sys.argv) > 1 else "spawn"
publish.single("attack/ais_control", json.dumps({
    "action": action, "lat": 11.30, "lon": 72.90, "speed": 16.0, "heading": 95.0
}), hostname="localhost")
print(f"Sent AIS attack command: {action} (PNS Khyber)")
