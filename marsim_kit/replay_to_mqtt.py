"""
Step 7: replay MARSIM attacks into your testbed over MQTT (data-layer injection, no RF).

The pcap's NMEA sentences are published with their original timing, so the hub sees
exactly what a spoofed receiver would send.

    python replay_to_mqtt.py --pcap data/marsim/<spoofed file>.pcap --broker 192.168.1.50
        --topic testbed/ownship/gps/nmea --rx 192.168.0.10 --speed 1

Options:
  --rx      only replay this receiver (MARSIM has two: 192.168.0.10 and 192.168.0.11)
  --speed   2 = twice as fast, 0 = as fast as possible
  --json    publish {"rx": ..., "t": ..., "nmea": ...} instead of the raw NMEA line
Change --topic / payload format to whatever your hub subscribes to.
"""
import argparse
import json
import time

import paho.mqtt.client as mqtt

from pcap_reader import iter_nmea


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--broker", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic", default="testbed/ownship/gps/nmea")
    ap.add_argument("--rx", default="192.168.0.10")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)  # paho-mqtt >= 2.0
    except AttributeError:
        client = mqtt.Client()
    client.connect(a.broker, a.port, 60)
    client.loop_start()

    t_first, wall_first, n = None, time.time(), 0
    for src, t, s in iter_nmea(a.pcap):
        if a.rx and src != a.rx:
            continue
        if t_first is None:
            t_first = t
        if a.speed > 0:
            wait = (t - t_first) / a.speed - (time.time() - wall_first)
            if wait > 0:
                time.sleep(wait)
        payload = json.dumps({"rx": src, "t": time.time(), "nmea": s}) if a.json else s
        client.publish(a.topic, payload)
        n += 1
        if n % 100 == 0:
            print(f"  t = {t - t_first:5.0f} s  ({n} sentences)")
    client.loop_stop()
    client.disconnect()
    print(f"replayed {n} sentences")


if __name__ == "__main__":
    main()
