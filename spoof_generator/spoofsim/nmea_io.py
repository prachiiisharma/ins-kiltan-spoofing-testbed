"""Raw-log writers/readers: NMEA 0183 GNSS sentences and AIVDM/AIVDO."""
import numpy as np
import pandas as pd
from pyais.encode import encode_dict


def checksum(body: str) -> str:
    c = 0
    for ch in body:
        c ^= ord(ch)
    return f"{c:02X}"


def _s(body):
    return f"${body}*{checksum(body)}"


def _lat(lat):
    a = abs(lat)
    d = int(a)
    return f"{d:02d}{(a - d) * 60:08.5f}", "N" if lat >= 0 else "S"


def _lon(lon):
    a = abs(lon)
    d = int(a)
    return f"{d:03d}{(a - d) * 60:08.5f}", "E" if lon >= 0 else "W"


def gnss_epoch_sentences(r) -> list:
    utc = pd.Timestamp(r.utc)
    hms = utc.strftime("%H%M%S") + f".{int(utc.microsecond / 1e4):02d}"
    dmy = utc.strftime("%d%m%y")
    out = []
    if r.fix_quality and np.isfinite(r.lat):
        la, ns = _lat(r.lat)
        lo, ew = _lon(r.lon)
        out.append(_s(f"GPGGA,{hms},{la},{ns},{lo},{ew},1,{int(r.nsat_used):02d},{r.hdop:.2f},"
                      f"{r.alt_m:.1f},M,-92.3,M,,"))
        out.append(_s(f"GPRMC,{hms},A,{la},{ns},{lo},{ew},{r.sog_kn:.3f},{r.cog_deg:.2f},{dmy},,,A"))
        prns = list(r.sat_prn)[:12] + [""] * (12 - min(len(r.sat_prn), 12))
        out.append(_s("GPGSA,A,3," + ",".join(f"{p:02d}" if p != "" else "" for p in prns) + f",{r.pdop:.2f},{r.hdop:.2f},{r.vdop:.2f}"))
    else:
        out.append(_s(f"GPGGA,{hms},,,,,0,00,99.99,,,,,,"))
        out.append(_s(f"GPRMC,{hms},V,,,,,,,{dmy},,,N"))
        out.append(_s("GPGSA,A,1,,,,,,,,,,,,,99.99,99.99,99.99"))
    sats = list(zip(r.sat_prn, r.sat_el, r.sat_az, r.sat_snr))
    nmsg = max(1, int(np.ceil(len(sats) / 4)))
    for k in range(nmsg):
        chunk = sats[4 * k:4 * k + 4]
        f = ",".join(f"{p:02d},{e:02d},{a:03d},{s:02d}" for p, e, a, s in chunk)
        out.append(_s(f"GPGSV,{nmsg},{k + 1},{len(sats):02d}" + ("," + f if f else "")))
    return out


def write_gnss_log(df: pd.DataFrame, path):
    with open(path, "w") as fh:
        for r in df.itertuples(index=False):
            for s in gnss_epoch_sentences(r):
                fh.write(s + "\n")


def ais_sentences(r, seq=[0]) -> list:
    talker_sentence = "VDO" if r.is_own else "VDM"
    mt = int(r.msg_type)
    if mt in (1, 2, 3, 18):
        lat = r.lat if np.isfinite(r.lat) else 91.0
        lon = r.lon if np.isfinite(r.lon) else 181.0
        d = dict(type=mt, mmsi=int(r.mmsi), lat=lat, lon=lon,
                 speed=float(r.sog_kn) if np.isfinite(r.sog_kn) else 102.3,
                 course=float(r.cog_deg) if np.isfinite(r.cog_deg) else 360.0,
                 heading=int(r.heading) if np.isfinite(r.heading) else 511,
                 second=int(pd.Timestamp(r.utc).second), accuracy=1)
        if mt != 18:
            d.update(status=int(r.nav_status), turn=float(np.clip(r.rot_dpm, -708, 708)))
    elif mt == 5:
        d = dict(type=5, mmsi=int(r.mmsi), shipname=str(r.ship_name or "")[:20], shiptype=int(r.ship_type or 0),
                 callsign="", destination="", to_bow=50, to_stern=50, to_port=10, to_starboard=10)
    else:  # 24 part A
        d = dict(type=24, mmsi=int(r.mmsi), partno=0, shipname=str(r.ship_name or "")[:20])
    seq[0] = (seq[0] + 1) % 10
    return encode_dict(d, radio_channel=r.channel, talker_id="AI", sentence_type=talker_sentence,
                       seq_id=seq[0])


def write_ais_log(df: pd.DataFrame, path):
    """AIS-catcher-style log: '<ISO UTC> <sentence>' per line (multi-part kept together)."""
    with open(path, "w") as fh:
        for r in df.itertuples(index=False):
            ts = pd.Timestamp(r.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-4] + "Z"
            for s in ais_sentences(r):
                fh.write(f"{ts} {s}\n")
