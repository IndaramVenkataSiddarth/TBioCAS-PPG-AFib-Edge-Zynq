#!/usr/bin/env python3
"""
Board test for the final 8-bit AFib accelerator - run ON the PYNQ-Z2 (Jupyter terminal / notebook cell: %run pynq_board_test.py ...).

Needs in one folder on the board:  final8.bit + final8.hwh (same base name, from Vivado) and the package files
  X_test.npy, y_test.npy, hls4ml_bitaccurate_predictions.npy, weights_float.npz, ppg_windows_example.npy

   python3 pynq_board_test.py --bit final8.bit --pkg . --runs 1000
   (no board? plumbing test only:)  python3 pynq_board_test.py --simulate --pkg ./final_8bit_rndwrap_package --runs 200

Register map of the custom AXI-Lite wrapper (paper Table IV):  AP_CTRL 0x00 | OUT_R 0x10 (float32) | IN_R[0..10] 0x40..0x7F (float32 each).
If your Vivado/HLS build uses other offsets, change the three constants below.

It reports:  (1) accuracy on the 348 test windows and the number of mismatches vs the bit-accurate hls4ml prediction (expected 0),
             (2) end-to-end latency per window through this Python driver,
             (3) ARM-only software inference (NumPy float MLP) per window,
             (4) feature-extraction time per 20 s PPG window on the ARM core.
Results -> board_results.csv and board_predictions.npy
"""
import argparse, struct, time, csv, os
import numpy as np

AP_CTRL, OUT_R, IN_R = 0x00, 0x10, 0x40
f2u = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]
u2f = lambda u: struct.unpack("<f", struct.pack("<I", int(u) & 0xFFFFFFFF))[0]


class FakeIP:           # plumbing test only: float MLP behind the same read/write interface
    def __init__(self, w):
        self.w, self.regs, self.out = w, {}, 0.0
    def write(self, off, val):
        self.regs[off] = val
        if off == AP_CTRL and val & 1:
            x = np.array([u2f(self.regs.get(IN_R + 4 * i, 0)) for i in range(11)], dtype=np.float32)
            h = np.maximum(x @ self.w["d1_kernel"] + self.w["d1_bias"], 0)
            h = np.maximum(h @ self.w["d2_kernel"] + self.w["d2_bias"], 0)
            self.out = float(1 / (1 + np.exp(-(h @ self.w["d3_kernel"] + self.w["d3_bias"])[0])))
            self.regs[AP_CTRL] = 0x2
    def read(self, off):
        return f2u(self.out) if off == OUT_R else self.regs.get(off, 0)


def infer(ip, x):
    for i in range(11):
        ip.write(IN_R + 4 * i, f2u(x[i]))
    ip.write(AP_CTRL, 1)                          # ap_start
    while not (ip.read(AP_CTRL) & 0x2):           # ap_done
        pass
    return u2f(ip.read(OUT_R))


def stats(t):
    t = np.asarray(t) * 1e6
    return dict(mean_us=round(t.mean(), 1), std_us=round(t.std(), 1), median_us=round(np.median(t), 1), n=len(t))


def numpy_mlp(w, x):
    h = np.maximum(x @ w["d1_kernel"] + w["d1_bias"], 0)
    h = np.maximum(h @ w["d2_kernel"] + w["d2_bias"], 0)
    return 1 / (1 + np.exp(-(h @ w["d3_kernel"] + w["d3_bias"])[0]))


def features_from_window(w, fs=125):          # same peak detection + 11 PRV features as the paper pipeline
    from scipy.signal import find_peaks
    from scipy.stats import skew
    rng = np.ptp(w)
    peaks, _ = find_peaks(w, distance=int(0.4 * fs), prominence=0.15 * rng if rng > 0 else 0)
    if len(peaks) < 2:
        return None
    iv = np.diff(peaks / fs)
    if ((iv > 0.33) & (iv < 1.5)).mean() < 0.8 or len(iv) < 4:
        return None
    d = np.diff(iv); m, s = iv.mean(), iv.std()
    return [m, s, np.sqrt(np.mean(d ** 2)), iv.min(), iv.max(), iv.max() - iv.min(), s / m if m > 0 else 0, len(iv) + 1,
            np.median(iv), skew(iv), np.mean(np.abs(d) > 0.05)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bit", default="final8.bit"); ap.add_argument("--pkg", default=".")
    ap.add_argument("--ip", default=None, help="IP name in the overlay (default: first containing 'myproject')")
    ap.add_argument("--runs", type=int, default=1000); ap.add_argument("--simulate", action="store_true")
    a = ap.parse_args()
    P = lambda f: os.path.join(a.pkg, f)
    x, y, exp = np.load(P("X_test.npy")), np.load(P("y_test.npy")), np.load(P("hls4ml_bitaccurate_predictions.npy"))
    w = dict(np.load(P("weights_float.npz")))
    if a.simulate:
        ip = FakeIP(w); print("SIMULATION (no board): plumbing test only; mismatches vs fixed-point are expected here")
    else:
        from pynq import Overlay
        ol = Overlay(a.bit)
        name = a.ip or next(k for k in ol.ip_dict if "myproject" in k.lower())
        ip = getattr(ol, name); print("using IP:", name)

    # (1) accuracy / agreement with the hls4ml bit-accurate prediction
    p = np.array([infer(ip, x[i]) for i in range(len(x))])
    acc = float(((p > .5) == (y > .5)).mean()); mism = int(((p > .5) != (exp > .5)).sum())
    print(f"board accuracy {acc:.4f} | prediction mismatches vs bit-accurate simulation: {mism} / {len(x)} | max |p - p_sim| = {np.abs(p - exp).max():.4f}")
    np.save("board_predictions.npy", p)

    # (2) end-to-end latency through this Python driver
    idx = np.arange(a.runs) % len(x); t = []
    for i in idx:
        t0 = time.perf_counter(); infer(ip, x[i]); t.append(time.perf_counter() - t0)
    s_fpga = stats(t); print("FPGA via Python driver:", s_fpga)

    # (3) ARM-only software inference (float NumPy MLP)
    t = []
    for i in idx:
        t0 = time.perf_counter(); numpy_mlp(w, x[i].astype(np.float32)); t.append(time.perf_counter() - t0)
    s_arm = stats(t); print("ARM NumPy float MLP:", s_arm)

    # (4) feature extraction per 20 s window
    s_feat = None
    try:
        ex = np.load(P("ppg_windows_example.npy")); t = []
        features_from_window(ex[0].astype(np.float64))      # warm-up (imports scipy) so it is not timed
        for r in range(max(1, a.runs // 20)):
            for wv in ex:
                t0 = time.perf_counter(); features_from_window(wv.astype(np.float64)); t.append(time.perf_counter() - t0)
        s_feat = stats(t); print("feature extraction per 20 s window:", s_feat)
    except Exception as e:
        print("feature-extraction timing skipped:", repr(e), "(install scipy on the board if missing)")

    with open("board_results.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["measurement", "mean_us", "std_us", "median_us", "n", "note"])
        wr.writerow(["accuracy", acc, "", "", len(x), f"mismatches_vs_hls4ml={mism}"])
        for nm, s in (("fpga_python_driver", s_fpga), ("arm_numpy_float_mlp", s_arm), ("feature_extraction_per_window", s_feat)):
            if s:
                wr.writerow([nm, s["mean_us"], s["std_us"], s["median_us"], s["n"], ""])
    print("saved board_results.csv, board_predictions.npy")


if __name__ == "__main__":
    main()
