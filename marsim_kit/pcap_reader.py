"""
Fast reader for the NMEA-over-UDP pcap files in the MARSIM dataset.

MARSIM has 43,320 pcaps. Scapy takes a fraction of a second per file, which adds
up to hours, so this module reads classic .pcap files directly with `struct`
(~50x faster) and falls back to scapy only for .pcapng files.

Usage:
    for rx_ip, t, sentence in iter_nmea("A1-spoofed-....pcap"):
        print(rx_ip, t, sentence)

    rx_ip    -> source IP of the GPS receiver, e.g. "192.168.0.10"
    t        -> packet arrival time, float seconds (Unix epoch)
    sentence -> one NMEA sentence as text, e.g. "$GPGGA,....*5C"
"""
import struct

# link-layer types we know how to strip
LINK_NULL, LINK_ETHERNET, LINK_RAW, LINK_LINUX_SLL, LINK_IPV4, LINK_LINUX_SLL2 = 0, 1, 101, 113, 228, 276


def _ip_udp_payload(buf, off):
    """Return (src_ip, payload) for an IPv4/UDP packet starting at buf[off], else None."""
    if len(buf) < off + 20 or (buf[off] >> 4) != 4:
        return None
    ihl = (buf[off] & 0x0F) * 4
    if buf[off + 9] != 17:  # not UDP
        return None
    src = ".".join(str(b) for b in buf[off + 12:off + 16])
    udp = off + ihl
    if len(buf) < udp + 8:
        return None
    udp_len = struct.unpack(">H", buf[udp + 4:udp + 6])[0]
    end = min(len(buf), udp + udp_len) if udp_len >= 8 else len(buf)
    return src, buf[udp + 8:end]


def _strip_link(buf, linktype):
    """Return the offset of the IPv4 header for this link type, or None."""
    if linktype == LINK_ETHERNET:
        off, etype = 14, struct.unpack(">H", buf[12:14])[0]
        while etype in (0x8100, 0x88A8):  # VLAN tags
            etype = struct.unpack(">H", buf[off + 2:off + 4])[0]
            off += 4
        return off if etype == 0x0800 else None
    if linktype in (LINK_RAW, LINK_IPV4):
        return 0
    if linktype == LINK_NULL:
        return 4
    if linktype == LINK_LINUX_SLL:
        return 16 if struct.unpack(">H", buf[14:16])[0] == 0x0800 else None
    if linktype == LINK_LINUX_SLL2:
        return 20 if struct.unpack(">H", buf[0:2])[0] == 0x0800 else None
    return None


def _iter_classic_pcap(data):
    magic = data[:4]
    if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
        endian = "<"
    elif magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
        endian = ">"
    else:
        raise ValueError("not a classic pcap file")
    nano = magic in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d")
    linktype = struct.unpack(endian + "I", data[20:24])[0] & 0x0FFFFFFF
    div = 1e9 if nano else 1e6
    off, n = 24, len(data)
    rec = struct.Struct(endian + "IIII")
    while off + 16 <= n:
        ts_s, ts_frac, incl, _ = rec.unpack_from(data, off)
        off += 16
        pkt = data[off:off + incl]
        off += incl
        ip_off = _strip_link(pkt, linktype)
        if ip_off is None:
            continue
        res = _ip_udp_payload(pkt, ip_off)
        if res:
            yield res[0], ts_s + ts_frac / div, res[1]


def _iter_scapy(path):
    from scapy.all import PcapReader  # only needed for pcapng
    from scapy.layers.inet import IP, UDP
    with PcapReader(path) as reader:
        for pkt in reader:
            if IP in pkt and UDP in pkt:
                yield pkt[IP].src, float(pkt.time), bytes(pkt[UDP].payload)


def iter_udp(path):
    """Yield (src_ip, time, udp_payload_bytes) for every UDP packet in the file."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] == b"\x0a\x0d\x0d\x0a":  # pcapng
        yield from _iter_scapy(path)
    else:
        yield from _iter_classic_pcap(data)


def iter_nmea(path):
    """Yield (src_ip, time, sentence) for every NMEA sentence in the file."""
    for src, t, payload in iter_udp(path):
        for line in payload.replace(b"\r", b"\n").split(b"\n"):
            line = line.strip()
            if line.startswith((b"$", b"!")):
                yield src, t, line.decode("ascii", errors="ignore")
