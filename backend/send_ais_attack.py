import json
import sys
import paho.mqtt.publish as publish

action = sys.argv[1]
attack_type = "spawn_ghost" if action == "spawn" else "stop_ghost"
publish.single("attack/ais_control", json.dumps({"type": attack_type}), hostname="localhost")
print("Sent:", attack_type)