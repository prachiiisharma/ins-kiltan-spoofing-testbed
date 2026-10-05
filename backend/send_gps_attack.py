import json, sys
import paho.mqtt.publish as publish

# Usage: python send_gps_attack.py [none|jump|drift]
attack_type = sys.argv[1] if len(sys.argv) > 1 else "jump"
publish.single("attack/gps_control", json.dumps({"type": attack_type}), hostname="localhost")
print(f"Sent GPS attack command: {attack_type}")
