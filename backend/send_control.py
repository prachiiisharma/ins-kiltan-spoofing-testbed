import json
import sys
import paho.mqtt.publish as publish

speed = float(sys.argv[1])
heading = float(sys.argv[2])

payload = json.dumps({"speed": speed, "heading": heading})
publish.single("ship/control", payload, hostname="localhost")
print("Sent:", payload)