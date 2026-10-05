"""MPU6050-like 6-axis IMU in the ship body frame (x fwd, y port, z up).

Includes wave-induced roll/pitch/heave, constant turn-on bias and gyro bias
random walk, so dead-reckoning drift in the dataset is realistic rather than
perfect.
"""
import numpy as np
import pandas as pd

G = 9.80665


def simulate_imu(rng, truth: pd.DataFrame, rate_hz=10):
    t_truth = truth.t.values
    t = np.arange(t_truth[0], t_truth[-1] + 1e-9, 1.0 / rate_hz)
    n = len(t)
    rot = np.interp(t, t_truth, np.radians(truth.rot_dps.values))      # rad/s yaw rate
    afwd = np.interp(t, t_truth, truth.accel_fwd_ms2.values)
    v = np.interp(t, t_truth, truth.stw_ms.values)

    sea = rng.uniform(0.3, 1.0)                                        # sea-state scale
    Tr, Tp, Th = rng.uniform(7, 12), rng.uniform(5, 9), rng.uniform(6, 10)
    roll = np.radians(4 * sea) * np.sin(2 * np.pi * t / Tr + rng.uniform(0, 6))
    pitch = np.radians(1.5 * sea) * np.sin(2 * np.pi * t / Tp + rng.uniform(0, 6))
    heave_acc = 0.4 * sea * np.sin(2 * np.pi * t / Th + rng.uniform(0, 6))
    droll = np.gradient(roll, t)
    dpitch = np.gradient(pitch, t)

    a_bias = rng.normal(0, 0.05, 3)
    g_bias = rng.normal(0, np.radians(0.3), 3)
    g_rw = np.cumsum(rng.normal(0, np.radians(0.002), (n, 3)), axis=0)  # bias instability

    ax = afwd + G * np.sin(pitch) + a_bias[0] + rng.normal(0, 0.035, n)
    ay = v * rot - G * np.sin(roll) + a_bias[1] + rng.normal(0, 0.035, n)   # centripetal
    az = G * np.cos(roll) * np.cos(pitch) + heave_acc + a_bias[2] + rng.normal(0, 0.045, n)
    gx = droll + g_bias[0] + g_rw[:, 0] + rng.normal(0, 0.002, n)
    gy = dpitch + g_bias[1] + g_rw[:, 1] + rng.normal(0, 0.002, n)
    gz = rot + g_bias[2] + g_rw[:, 2] + rng.normal(0, 0.002, n)
    return pd.DataFrame({"t": np.round(t, 3), "ax_ms2": ax, "ay_ms2": ay, "az_ms2": az,
                         "gx_rads": gx, "gy_rads": gy, "gz_rads": gz})
