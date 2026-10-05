import json, sys
import paho.mqtt.publish as publish

# Usage: python send_control.py <speed_knots> <heading_deg>
speed = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
heading = float(sys.argv[2]) if len(sys.argv) > 2 else 270.0
publish.single("ship/control", json.dumps({"speed": speed, "heading": heading}), hostname="localhost")
print(f"Sent control: speed={speed} heading={heading}")
