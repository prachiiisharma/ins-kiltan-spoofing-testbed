"""
Step 1 (optional): make a tiny fake dataset in MARSIM's format to test the pipeline
BEFORE the big download finishes. Numbers from this data mean nothing - it only checks
that every script runs on your PC.

    python make_test_pcaps.py --out data/test_marsim --per-group 15
"""
import argparse
import json
import math
import os
import random
import struct

from nmea_features import destination, haversine_m

KN = 0.514444
RX = ["192.168.0.10", "192.168.0.11"]


def cs(body):
    c = 0
    for ch in body:
        c ^= ord(ch)
    return f"${body}*{c:02X}"


def ddmm(v, lat):
    a = abs(v)
    d = int(a)
    m = (a - d) * 60
    return (f"{d:02d}{m:07.4f}" if lat else f"{d:03d}{m:07.4f}"), ("N" if v >= 0 else "S") if lat else \
        ("E" if v >= 0 else "W")


def sentences(utc, lat, lon, sog, cog, alt, sats):
    hh = f"{int(utc // 3600) % 24:02d}{int(utc // 60) % 60:02d}{utc % 60:05.2f}"
    la, ns = ddmm(lat, True)
    lo, ew = ddmm(lon, False)
    out = [cs(f"GPRMC,{hh},A,{la},{ns},{lo},{ew},{sog:.2f},{cog:.2f},010118,,,A"),
           cs(f"GPGGA,{hh},{la},{ns},{lo},{ew},1,{len(sats):02d},0.9,{alt:.1f},M,46.9,M,,"),
           cs("GPGSA,A,3," + ",".join([p for p, *_ in sats][:12] + [""] * (12 - min(12, len(sats))))
              + ",1.6,0.9,1.3")]
    n = math.ceil(len(sats) / 4)
    for i in range(n):
        grp = sats[i * 4:(i + 1) * 4]
        body = f"GPGSV,{n},{i + 1},{len(sats):02d}," + ",".join(f"{p},{e},{az},{snr}" for p, e, az, snr in grp)
        out.append(cs(body))
    out.append(cs(f"GPVTG,{cog:.2f},T,,M,{sog:.2f},N,{sog * 1.852:.2f},K,A"))
    return out


def eth_ip_udp(src, payload):
    udp = struct.pack(">HHHH", 10110, 10110, 8 + len(payload), 0) + payload
    ip = struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + len(udp), 0, 0, 64, 17, 0,
                     bytes(int(x) for x in src.split(".")), bytes([192, 168, 0, 255]))
    return b"\xff" * 6 + b"\x02" * 6 + b"\x08\x00" + ip + udp


def make_file(path, scenario, spoofed, rng):
    lat0, lon0 = 54.3 + rng.random() * 0.2, 10.1 + rng.random() * 0.2
    cog, sog, t0 = rng.random() * 360, 20.0, 1514800000.0
    utc0 = 36000.0 + rng.random() * 1000
    sats = [(f"{p:02d}", rng.randint(10, 80), rng.randint(0, 359), 0) for p in rng.sample(range(1, 33), 9)]
    pos = [(lat0, lon0), destination(lat0, lon0, cog + 90, 4.0)]
    shift = (rng.uniform(50, 400), rng.uniform(0, 360), rng.uniform(0.5, 5), rng.uniform(1, 30))
    recs = []
    for k in range(120):
        t = t0 + k
        cog_k = (cog + 0.5 * k) % 360
        for r in (0, 1):
            lat, lon = pos[r]
            utc, s_sog, s_cog, ss = utc0 + k + 10.55e-6 * k, sog, cog_k, sats
            if r == 1:
                lat, lon = destination(*pos[0], cog_k + 90, 4.0)
            if spoofed and k >= 60:
                j = k - 60
                if scenario == "A1":     # replay: old track shifted sideways, time in the past
                    lat, lon = destination(*pos[0], cog_k + 90, shift[0])
                    utc -= shift[3]
                elif scenario == "A2":   # meaconing: both receivers see the relay antenna, delayed
                    lat, lon = destination(*pos[0], shift[1], shift[0])
                    utc -= shift[2]
                else:                    # simulator: slow drag away, both receivers same point
                    lat, lon = destination(*pos[0], shift[1], shift[2] * j)
                ss = [(p, e, a, 0) for p, e, a, _ in sats]
            snr = [(p, e, a, (44 + rng.randint(-1, 1)) if (spoofed and k >= 60) else
                    max(15, min(50, 20 + e // 3 + rng.randint(-4, 4)))) for p, e, a, _ in ss]
            jitter = lambda: rng.gauss(0, 1.5e-6)
            ts = t + 0.01 * r + rng.random() * 0.02
            for s in sentences(utc, lat + jitter(), lon + jitter(), s_sog + rng.gauss(0, 0.1), s_cog,
                               0.3 + rng.gauss(0, 0.2), snr):
                ts += 0.002 + rng.random() * 0.005  # sentences of one receiver stay in order
                recs.append((ts, RX[r], (s + "\r\n").encode()))
        pos[0] = destination(*pos[0], cog_k, sog * KN)
    recs.sort()
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for t, src, payload in recs:
            pkt = eth_ip_udp(src, payload)
            f.write(struct.pack("<IIII", int(t), int((t % 1) * 1e6), len(pkt), len(pkt)) + pkt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("data", "test_marsim"))
    ap.add_argument("--per-group", type=int, default=15)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(1)
    index = []
    for scn in ("A1", "A2", "A3"):
        for label in ("spoofed", "unspoofed"):
            for i in range(a.per_group):
                name = f"{scn}-{label}-{i}-test_param-{i}.pcap"
                make_file(os.path.join(a.out, name), scn, label == "spoofed", rng)
                index.append({"filename": name, "scenario": scn, "label": label, "index": str(i),
                              "parameters": {"test_param": str(i)}})
    with open(os.path.join(a.out, "dataset.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(f"wrote {len(index)} test pcaps to {a.out}")


if __name__ == "__main__":
    main()
