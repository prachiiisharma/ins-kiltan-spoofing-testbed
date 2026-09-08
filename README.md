\# INS Kiltan spoofing testbed



B.Tech final year project — a fused GPS and AIS spoofing detection testbed.

Own ship (real GNSS + dead reckoning + spoof injector) and an enemy ship

(ghost AIS generator) feed a detection engine, visualized on a live

Leaflet/three.js dashboard with a hardware alarm.



\## Structure

\- `backend/` — own ship, enemy ship, and detection logic

\- `frontend/` — dashboard (map, console, bridge view)

\- `hardware/` — Raspberry Pi/ESP32 alarm controller

\- `models/` — LSTM training on historical AIS data

