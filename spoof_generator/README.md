# Maritime GNSS + AIS Spoofing Dataset Generator

**Project:** AI Based Maritime Cybersecurity Risk Detection and Management System for Modern Shipping Networks
**Team:** Syndicate-9 · Indian Naval Academy, Ezhimala

This repo builds a **labelled dataset** for training and evaluating the spoofing detectors:

- **GNSS stream:** own-ship receiver output, NEO-6M-like, 1 Hz.
- **IMU stream:** MPU6050-like, 10 Hz.
- **Gyro/log stream:** ship's gyro heading and speed log, 1 Hz.
- **AIS stream:** own ship plus 4–10 traffic vessels, ITU reporting rates.
- **Labelled spoofing attacks** on both the GNSS and AIS streams.
- **Benign hard negatives:** manoeuvres, multipath, fix outages and AIS reception gaps. These test false alarms.

All spoofing is injected at the **data layer** (NMEA / AIVDM). No RF is generated, so the work stays within the legal and safety scope.
The same injectors work on **real recordings** (NEO-6M logs, AIS-catcher logs, DMA/NOAA AIS CSVs). This lets you mix simulated and real authentic data.

---

## Quick start

```bash
pip install numpy pandas pyarrow pyais pynmea2 pyyaml matplotlib
python build_dataset.py                    # ~4 min on 2 cores -> data/syndicate9_spoof_v1/
python validate_dataset.py data/syndicate9_spoof_v1
```

The generator is fully deterministic: the same config and `base_seed` produce the same dataset.
Change `n_voyages`, the attack probabilities and other settings in `config/default.yaml`.

## Output files

| File | One row per | Key columns |
|---|---|---|
| `gnss.parquet` / `.csv` | own-ship GNSS epoch (1 Hz) | `utc, fix_quality, lat, lon, sog_kn, cog_deg, hdop, vdop, pdop, nsat_used, nsat_visible, snr_mean/std/max/min, sat_prn/el/az/snr` (lists), `gyro_heading_deg, log_stw_kn` |
| `ais.parquet` / `.csv` | received AIS message | `t, utc, mmsi, msg_type (1/18 position, 5/24 static), channel, lat, lon, sog_kn, cog_deg, heading, rot_dpm, nav_status, ship_name, is_own` |
| `imu.parquet` | IMU sample (10 Hz) | `ax/ay/az_ms2, gx/gy/gz_rads` (body frame, with bias + bias drift) |
| `truth.parquet` | ground truth (own 1 Hz, traffic 10 s) | `lat, lon, heading_deg, stw_ms, sog_ms, cog_deg, rot_dps, drift_e/n_ms` (current + leeway) |
| `attacks.csv` | injected attack | `attack_id, voyage_id, split, stream, type, start, duration`, plus the attack's parameters |
| `voyages.json` | voyage | seed, start UTC, environment (current, wind), benign events, attacks, split |
| `raw/<voyage>/gnss.nmea`, `ais.nmea` | raw sentences | test-split voyages, for end-to-end replay through your parser |
| `figures/`, `validation_report.md` | — | output of `validate_dataset.py` |

### Label columns (in `gnss` and `ais`)

- `is_spoofed`: 0 or 1. This is the **target for binary detection**.
- `attack_type`: the attack name, or `none`. Use it for **multi-class** classification and for per-attack metrics.
- `attack_id`: the attack instance. Use it for **detection latency** (first alert time minus the attack's `start`).
- `benign_event`: a legitimate but unusual condition. These rows have `is_spoofed = 0`; report false-alarm rate on them separately.
- `spoof_signature` (GNSS only): 1 if the spoofer overpowered the sky (uniform high C/N0). Stealthy attacks have 0.
- `vessel_uid` (AIS only): the *true* transmitter. Ghost and duplicated-MMSI messages come from `spoofer_*`. **Do not use this column as a feature.** It is ground truth.
- `truth_lat` / `truth_lon` (AIS only): the true position of a real transmitter. **Do not use these as features either.**

## Attack catalogue

**GNSS (own ship):**

| Type | What happens | Why it matters |
|---|---|---|
| `jump` | Position offset of 0.3–5 km, velocity stays consistent | Easy for a residual check; tests the basic layer |
| `drag_off` | Slow, Doppler-consistent pull-off at 0.002–0.02 m/s² up to 0.3–3 m/s | The realistic capture attack; hardest to catch |
| `replay` | Position, velocity and UTC replayed from 30–600 s earlier | A meaconing/replay signature |
| `freeze` | Position and velocity frozen | Caught by DR and IMU disagreement |
| `time_jump` | UTC offset only; position untouched | A timing attack; needs a clock or time-consistency feature |

About 60% of GNSS attacks also carry the **signal-level signature**: uniform, high C/N0 and a clean geometry.

**AIS:**

| Type | What happens |
|---|---|
| `position_jump` | A real vessel's reports are shifted by 1–20 km |
| `position_drift` | A real vessel is dragged off at 0.5–4 m/s |
| `ghost_vessel` | A fake MMSI appears. Its track is either plausible or impossible (stretched 3–8×, so implied speed ≠ SOG) |
| `mmsi_duplication` | A second transmitter uses a real MMSI 10–60 km away. 30% of these impersonate **own ship** |
| `kinematic_mismatch` | Reported SOG or COG contradicts the positional track (e.g. "at anchor" while moving) |
| `invalid_identity` | The MMSI is replaced by an invalid or unallocated one |
| `propagated_gnss_*` | Own-ship AIS (VDO) carries the spoofed GNSS position, just as on a real ship |

## Splits

Splits are made **by voyage**: 70% train, 15% val, 15% test. The builder reshuffles until every split contains every attack type.
Never split by row. Neighbouring 1 Hz epochs are nearly identical, so a row split leaks the answer and inflates your scores.

## Using real recordings

```python
from spoofsim.real_data import load_nmea_log, load_ais_csv, load_ais_nmea_log
from spoofsim import attacks_gnss as AG, attacks_ais as AA
import numpy as np

g = AG.init_labels(load_nmea_log("neo6m_run1.nmea"))       # your NEO-6M log
rng = np.random.default_rng(0)
spec = AG.random_gnss_attack(rng, "drag_off", start=600, duration=600)
g = AG.apply_gnss_attack(g, spec, "REAL1_G0", ref_lat=12.0, ref_lon=74.6, rng=rng)

ais = AA.init_labels(load_ais_csv("aisdk-2026-06-01.csv", bbox=(54, 58, 8, 13)))  # DMA file
# or: load_ais_nmea_log("ais_catcher.log")                  # RTL-SDR + AIS-catcher
```

Real recordings are treated as authentic (`is_spoofed = 0`) before injection.
Free real AIS sources are the Danish Maritime Authority (daily CSVs) and NOAA MarineCadastre (US waters).
These are useful for testing whether a model trained on simulated traffic transfers to real traffic.

## Package layout

```
spoofsim/
  vessel.py        ship kinematics: waypoints, rate-limited steering, current + wind leeway drift
  gnss.py          receiver model: Gauss-Markov error, constellation geometry, DOPs, C/N0, degradations
  imu.py           MPU6050 model: waves, bias, bias random walk
  ais.py           AIS transmit/receive: ITU-R M.1371 intervals, range-dependent reception
  attacks_gnss.py  GNSS attack injectors
  attacks_ais.py   AIS attack injectors
  scenario.py      one voyage = own ship + traffic + attacks + benign events
  nmea_io.py       NMEA 0183 / AIVDM writers
  real_data.py     loaders for real NMEA logs, AIS-catcher logs, DMA/NOAA CSVs
build_dataset.py   generates the dataset
validate_dataset.py  checks labels and the raw-log round trip, and draws figures
```

## Known limitations (state these in the report)

- The data is simulated. The error and noise models are standard (Gauss-Markov GNSS error, ITU AIS rates, an MPU6050-class IMU), but they are not calibrated against your own hardware yet. Recording 1–2 hours of real NEO-6M and IMU data and comparing their statistics is the recommended next validation step.
- The signal-level features come from NMEA (C/N0, DOP, satellite counts). Correlation-peak or AGC monitoring would need raw-IF access, which is future work.
- The spoofer's satellite list mirrors the authentic one. A real spoofer may present a different PRN set.
