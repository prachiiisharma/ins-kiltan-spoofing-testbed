# MARSIM toolkit for the GNSS spoofing detection project

Scripts to download the MARSIM dataset, turn it into features, train and evaluate the
Isolation Forest detector, compare it against MANA, and plug the trained model into the
testbed hub.

MARSIM: 43,320 pcap files of NMEA-0183 over UDP, 120 s each, two GPS receivers
(192.168.0.10 / .11) 4 m apart, half the files spoofed from t = 60 s by a replay,
meaconing or simulator attacker.
Paper: Spravil et al., *Detecting Maritime GPS Spoofing Attacks Based on NMEA Sentence
Integrity Monitoring*, JMSE 11(5):928, 2023. Dataset DOI: 10.5281/zenodo.8202802.

## Files

| File | Step | What it does |
|---|---|---|
| `make_test_pcaps.py` | 1 | makes a small fake MARSIM-format dataset to test the scripts (results are meaningless) |
| `download_marsim.py` | 2 | downloads from Zenodo (resumable, MD5 check) and extracts to `data/marsim` |
| `pcap_reader.py` | – | fast pcap → NMEA reader (no scapy needed for .pcap) |
| `nmea_features.py` | – | streaming NMEA feature extractor, shared by training and live detection |
| `build_features.py` | 3 | pcaps → labelled Parquet feature table (one row per second) |
| `train_iforest.py` | 4 | trains and evaluates the Isolation Forest, saves `model.joblib` + metrics |
| `run_mana_baseline.py` + `mana_methods.json` | 5 | runs the original MANA detectors on the same test files |
| `detector.py` | 6 | live detector class for the hub; also runs on a pcap or a serial GPS |
| `replay_to_mqtt.py` | 7 | replays a MARSIM attack into the testbed over MQTT |

## Windows (PowerShell) quick start

```powershell
cd ins-kiltan-spoofing-testbed
# copy this folder in as  marsim_kit\
cd marsim_kit
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt

python make_test_pcaps.py                                            # 1. smoke test
python build_features.py --data data\test_marsim --out features\test
python train_iforest.py --features features\test --out results\test --mode single

python download_marsim.py                                            # 2. real data
python build_features.py --data data\marsim --out features\marsim    # 3.
python train_iforest.py --features features\marsim --out results\iforest_single --mode single   # 4.
python train_iforest.py --features features\marsim --out results\iforest_dual   --mode dual

git clone https://github.com/fkie-cad/mana ..\..\mana                 # 5. baseline
python run_mana_baseline.py --mana-dir ..\..\mana --data data\marsim `
    --split-from results\iforest_single\per_file_results.csv --out results\mana_baseline

python detector.py --model results\iforest_single\model.joblib --pcap data\marsim\<some spoofed file>.pcap   # 6.
python detector.py --model results\iforest_single\model.joblib --serial COM5                                 # NEO-6M
python replay_to_mqtt.py --pcap data\marsim\<file>.pcap --broker <hub IP> --topic <your NMEA topic>          # 7.
```

## Notes

* **Single vs dual mode.** Your hardware has one NEO-6M per ship, so `--mode single` is the
  model to deploy. `--mode dual` adds the two-receiver distance check (MARSIM's strongest
  cue); report it as an upper bound / future-work hardware option.
* **Features are consistency checks**, not absolute values: speed reported vs speed implied by
  position change, dead-reckoning residual, course vs track, receiver clock vs arrival clock,
  C/N0 spread across satellites, satellite-set changes. No raw lat/lon or speed goes into the
  model, so it is not tied to MARSIM's route or the 20 kn ship speed.
* **Split by file, never by row.** Rows from one recording are strongly correlated; a random
  row split would leak and inflate results.
* **Strict vs file-level metrics.** "File-level" matches the MANA paper (was the file flagged?).
  "Strict" only counts alarms after the attack starts at 60 s; earlier alarms are false alarms.
  Report both, plus median detection delay.
* **MANA quirks fixed here:** its `setup.py` lists the dead `sklearn` package and its scapy-based
  pcap reader crashes on current scapy (it then reports zero detections without an error), so
  `run_mana_baseline.py` uses MANA from the clone path and feeds it with `pcap_reader.py`.
* **Replay speed.** Use `--speed 1` for realistic results; faster replay distorts the clock features.
* Do not commit `data/`, `downloads/`, `features/` (gigabytes). `.gitignore` already excludes them.
* License: the dataset and MANA use a "supplemented MIT" license – read it and cite the paper and
  the Zenodo DOI in your report.
