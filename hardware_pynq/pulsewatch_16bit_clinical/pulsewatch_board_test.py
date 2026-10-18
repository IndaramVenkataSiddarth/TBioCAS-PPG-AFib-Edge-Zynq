#!/usr/bin/env python3
"""
Pulsewatch board test v2 (run ON the PYNQ-Z2).  Reports accuracy AND sensitivity/specificity AND agreement with the bit-accurate simulation.
Files in the same folder: design_1.bit, design_1.hwh, and for each set NAME in (balanced_10000, natural_10000):
    X_test_NAME.npy, y_test_NAME.npy, hls4ml_pred_NAME.npy      (made by notebook 08)
    python3 pulsewatch_board_test_v2.py
"""
import time, struct, csv
import numpy as np
from pynq import Overlay, MMIO
AP_CTRL, OUT_R, IN_R = 0x00, 0x10, 0x40
f2u = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]
u2f = lambda u: struct.unpack("<f", struct.pack("<I", int(u) & 0xFFFFFFFF))[0]

def main():
    ol = Overlay("design_1.bit")
    name = next(k for k in ol.ip_dict if "myproject" in k.lower())
    mmio = MMIO(ol.ip_dict[name]["phys_addr"], ol.ip_dict[name]["addr_range"])
    def infer(s):
        mmio.write(IN_R, np.array([f2u(v) for v in s], dtype=np.uint32).tobytes()); mmio.write(AP_CTRL, 1)
        while not (mmio.read(AP_CTRL) & 0x2): pass
        return u2f(mmio.read(OUT_R))
    for _ in range(20): infer(np.zeros(11, dtype=np.float32))
    rows = []
    for nm in ("balanced_10000", "natural_10000"):
        x, y, sim = np.load(f"X_test_{nm}.npy"), (np.load(f"y_test_{nm}.npy") > .5).astype(int), np.load(f"hls4ml_pred_{nm}.npy")
        t, p = [], []
        for s in x:
            t0 = time.perf_counter(); p.append(infer(s)); t.append(time.perf_counter() - t0)
        p = np.array(p); pr = (p > .5).astype(int)
        tp = int(((pr == 1) & (y == 1)).sum()); tn = int(((pr == 0) & (y == 0)).sum()); fp = int(((pr == 1) & (y == 0)).sum()); fn = int(((pr == 0) & (y == 1)).sum())
        r = dict(set=nm, n=len(y), AF_share=round(float(y.mean()) * 100, 1), accuracy=round((tp + tn) / len(y) * 100, 2),
                 sensitivity=round(tp / max(tp + fn, 1) * 100, 2), specificity=round(tn / max(tn + fp, 1) * 100, 2),
                 TP=tp, FN=fn, TN=tn, FP=fp, mismatches_vs_sim=int(((p > .5) != (sim > .5)).sum()),
                 mean_latency_us=round(float(np.mean(t)) * 1e6, 1), median_latency_us=round(float(np.median(t)) * 1e6, 1))
        print(r); rows.append(r)
    with open("pulsewatch_board_results_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("saved pulsewatch_board_results_v2.csv")

if __name__ == "__main__":
    main()
