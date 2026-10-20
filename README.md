# Resource-Efficient Edge Deployment of a Quantized PRV-Feature Neural Network for PPG-Based AFib Screening

Official open-source implementation repository for the revised manuscript submitted to **IEEE Transactions on Biomedical Circuits and Systems (TBioCAS)**:  
*Paper ID:* **TBioCAS-2026-Aug-0332**

---

## 🌟 Key Features & Pipeline Overview

1. **Clinical PPG Processing & 4× Physiological Data Augmentation:**
   - **Primary Dataset:** 35 clinical patients from **MIMIC-III PERform AF** (125 Hz transmissive finger-clip PPG, 19 AFib, 16 Normal Sinus Rhythm).
   - **Physiological Augmentation:** Implemented following Cheng et al. (2020) using additive Gaussian sensor noise ($\sigma = 0.05 \cdot \text{STD}$), amplitude scaling ($\alpha \sim \mathcal{N}(1.0, 0.15)$), and $0.2\text{ Hz}$ respiratory baseline wander.
   - **Evaluated Dataset:** Expanded 4-fold to **7,706 evaluated 20-second windows** (**3,424 Non-AFib** and **4,282 AFib**) after applying a pulse plausibility filter (retaining 93.76% AFib and 89.24% Non-AFib windows).
   - **PRV Feature Extraction:** 11 time-domain Pulse Rate Variability (PRV) features (Mean, STD, RMSSD, Min, Max, Range, Median, CV, Beat Count, Skewness, pNN50).

2. **Rigorous Validation & Baseline Benchmarking:**
   - **Leave-One-Subject-Out (LOSO) Validation (All 35 Subjects):** 
     - Pooled Accuracy: **96.30%** (95% CI: 93.60% – 98.37%)
     - Sensitivity / Recall: **98.22%** (95% CI: 96.10% – 99.63%)
     - Specificity: **93.91%** (95% CI: 87.82% – 97.97%)
     - ROC-AUC: **0.987** (95% CI: 0.971 – 0.996)
     - Mean per-subject accuracy: **91.82%** (95% CI: 83.79% – 97.97%)
   - **Comparison with 7 Baseline Classifiers:** Logistic Regression (96.09%), Random Forest (96.46%), and SVM-RBF (96.04%) show comparable accuracy on these features, establishing that the paper's contribution lies in hardware-verified post-training quantization and edge deployment rather than model complexity.
   - **Subject-Wise 5-Fold Cross-Validation:** **95.80% ± 2.73%** (Float32) and **96.45% ± 2.05%** (8-bit PTQ).
   - **Held-Out Test Split (1,394 windows):** **98.06%** (software float) and **97.63%** (on-chip physical PYNQ-Z2 bitstream; 1,361/1,394 correct).

3. **Data-Driven 8-Bit Profiled Post-Training Quantization (PTQ):**
   - Fixed-point bitwidths profiled strictly from training set value ranges:
     - Inputs & Layer Outputs: `ap_fixed<8,5>`
     - Hidden Weights & Biases: `ap_fixed<8,1>`
     - Output Weights & Biases: `ap_fixed<8,2>`
     - Internal Accumulators: `ap_fixed<16,8>`
     - Output Pre-Activation (Logit): `ap_fixed<12,5>` (prevents overflow sign-flip without requiring high-cost saturation logic)
   - **Rounding & Overflow:** Round-to-nearest (`AP_RND`) and wrap-around (`AP_WRAP`).
   - **PTQ vs. QAT:** 8-bit PTQ matches the unquantized float model (differs by only −0.05 percentage points in LOSO validation), demonstrating that fine-tuning with Quantization-Aware Training (QAT) is unnecessary at 8 bits.

4. **Physical Silicon Verification on PYNQ-Z2 (Zynq-7020):**
   - **Clock & Timing:** 100 MHz target clock, meeting all constraints (WNS = +1.834 ns, WHS = +0.010 ns).
   - **Fabric Utilization:** 7,762 LUTs (14.59%), 8,457 FFs (7.95%), 30 DSPs (13.64%), and 0.5 BRAM tiles (1 block of 18 Kb, 0.36%).
   - **Driver Latencies Measured on Board:**
     - Per-word Python register driver: **0.517 ms**
     - Direct memory-mapped array driver: **0.188 ms** (1.87× faster than ARM software)
     - Pre-buffered memory-mapped array driver: **0.087 ms** (4.03× faster than ARM software)
   - **Software Baseline & Break-Even:** Native ARM Cortex-A9 software inference takes **0.351 ms**, and feature extraction takes **5.86 ms**. For a single 20 s PPG channel, one ARM core can serve ~3,200 channels in real time; dedicated FPGA acceleration becomes computationally impactful when feature extraction is also moved to programmable logic.
   - **Power Estimates:** Custom PL accelerator takes **~0.20 W at 100 MHz** (Vivado routed report); ARM PS baseline takes **1.525 W** (total SoC: **~1.73 W**).

5. **External Consistency Check on Pulsewatch Smartwatch Dataset:**
   - **Independent Ambulatory Population:** 101 outpatient subjects recorded with a smartwatch PPG at 50 Hz from the **Pulsewatch clinical trial** (Synapse `syn23565056`, used with permission from D. Han et al.).
   - **Silicon Deployment (Standalone Test Build: 7 DSPs, 4,578 LUTs):**
     - Evaluated across **10,000 real-world ambulatory windows**.
     - Physical Board Accuracy: **91.57%** (97.52% agreement with bit-accurate simulation)
     - AFib Sensitivity: **95.42%** (5,394 of 5,653 AFib windows detected)
     - Specificity: **86.57%** (reflecting expected real-world motion artifacts)
     - Hardware Execution Latency: **0.325 ms**

---

## 📁 Repository Directory Structure

```text
TBioCAS-PPG-AFib-Edge-Zynq/
├── .gitignore                                 # Git ignore rules
├── LICENSE                                    # Open-source MIT License
├── README.md                                  # Repository documentation
├── requirements.txt                           # Python dependencies
├── firmware_hls/                              # HLS4ML C++ firmware headers and AXI-Lite wrapper
│   ├── defines.h                              # Fixed-point typedefs (ap_fixed configurations)
│   ├── parameters.h                           # Layer dimensions, weights, and bias definitions
│   ├── myproject.h                            # Core neural network top function prototype
│   ├── myproject_axi.h                        # AXI4-Lite interface packet structures
│   └── myproject_axi.cpp                      # AXI4-Lite hardware interface wrapper
├── hardware_pynq/                             # Synthesized bitstreams and board drivers
│   ├── mimic_8bit_augmented/                  # Deployed 8-bit MIMIC-III system (30 DSPs, 7,762 LUTs)
│   │   ├── design_1.bit                       # Synthesized bitstream for Zynq-7020 PYNQ-Z2
│   │   ├── design_1.hwh                       # Hardware handoff definition
│   │   └── pynq_board_test.py                 # Python MMIO driver for live board inference
│   └── pulsewatch_16bit_clinical/             # Standalone Pulsewatch test build (7 DSPs, 4,578 LUTs)
│       ├── design_1_pulsewatch.bit            # Pulsewatch test bitstream
│       ├── design_1_pulsewatch.hwh            # Pulsewatch hardware handoff
│       └── pulsewatch_board_test.py           # PYNQ test script for clinical dataset
├── models/                                    # Pre-trained models and physiological feature scalers
│   ├── ppg_afib_model_augmented.h5            # Keras float32 trained baseline (337 parameters)
│   └── feature_scaler_augmented.pkl           # StandardScaler fit strictly on training subjects
└── notebooks/                                 # Executed Jupyter workflows
    ├── 01_mimic_augmented_edge_pipeline.ipynb # End-to-end training, 5-fold CV, PTQ, and HLS IP export
    ├── 02_pulsewatch_clinical_validation.ipynb# Independent clinical trial validation (101 smartwatch patients)
    ├── 10_qat_vs_ptq.ipynb                    # Controlled comparison: QAT vs. PTQ across bitwidths
    ├── 11_rounding_overflow_experiment.ipynb  # Evaluation of rounding and overflow saturation modes
    ├── 12_reuse_factor_sweep.ipynb            # Latency vs. resource trade-offs (Reuse Factors 1 to 64)
    ├── 13_loso_confidence_intervals.ipynb     # Leave-One-Subject-Out 95% subject-level bootstrap CIs
    ├── 14_baseline_classifiers.ipynb          # Classical baseline classifiers (LR, RF, SVM, k-NN)
    └── 15_filter_sensitivity.ipynb            # Signal quality filter and peak detector sensitivity sweep
```

---

## 📊 Summary of Silicon & Clinical Evaluation

### 1. MIMIC-III Clinical Benchmark (Primary Dataset)
* **Leave-One-Subject-Out (LOSO) Validation (All 35 Subjects):**
  * **Accuracy:** **96.30%** (95% CI: 93.60% – 98.37%)
  * **Sensitivity / Recall:** **98.22%** (95% CI: 96.10% – 99.63%)
  * **Specificity:** **93.91%** (95% CI: 87.82% – 97.97%)
  * **ROC-AUC:** **0.987** (95% CI: 0.971 – 0.996)
  * **Mean Per-Subject Accuracy:** **91.82%** (95% CI: 83.79% – 97.97%)
* **Physical Silicon Deployment on PYNQ-Z2 (1,394 Held-Out Windows):**
  * **On-Chip Accuracy:** **97.63%** (1,361 of 1,394 correct; matches simulation within 2 windows)
  * **Hardware Latency:** **0.188 ms** (direct MMIO array) | **0.087 ms** (pre-buffered array)
  * **Fabric Utilization:** 30 DSPs (13.64%), 7,762 LUTs (14.59%), 8,457 FFs (7.95%), 0.5 BRAM tiles (18 Kb)
  * **Power Consumption:** **~0.20 W** PL accelerator (100 MHz) | **1.525 W** ARM PS baseline (~1.73 W total SoC)

### 2. Pulsewatch Smartwatch Validation (Independent Clinical Cohort)
* **10,000 Real-World Ambulatory Windows (101 Outpatient Participants at 50 Hz):**
  * **Physical Board Accuracy:** **91.57%** (97.52% agreement with bit-accurate simulation)
  * **AFib Sensitivity:** **95.42%** (5,394 of 5,653 AFib windows detected)
  * **Specificity:** **86.57%** (reflecting real-world ambulatory motion artifacts)
  * **Hardware Execution Latency:** **0.325 ms**
  * **Standalone Hardware Footprint:** 7 DSP blocks, 4,578 LUTs, 0.26 W dynamic PL power

> **Note:** Accuracies apply to quality-filtered accepted windows (93.8% AFib, 89.2% Non-AFib). Preprocessing and feature extraction (5.86 ms/window) run on the ARM Processing System; the FPGA accelerates the neural network classifier.

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
git clone https://github.com/IndaramVenkataSiddarth/TBioCAS-PPG-AFib-Edge-Zynq.git
cd TBioCAS-PPG-AFib-Edge-Zynq
pip install -r requirements.txt
```

### 2. Running the Core Pipelines
* **Augmented Edge Pipeline:** Open `notebooks/01_mimic_augmented_edge_pipeline.ipynb` to reproduce feature extraction, 5-fold cross-validation, 8-bit profiled PTQ, and Vivado HLS IP export.
* **LOSO & Baselines:** Run `notebooks/13_loso_confidence_intervals.ipynb` and `notebooks/14_baseline_classifiers.ipynb` to reproduce the leave-one-subject-out uncertainty analysis and scikit-learn baseline comparisons.
* **Quantization & Ablation Sweeps:** Run notebooks `10` through `12` to inspect QAT vs. PTQ, rounding/wrap-around modes, and the reuse-factor trade-offs.

### 3. Live PYNQ-Z2 Hardware Inference
Transfer the `hardware_pynq/mimic_8bit_augmented/` directory to the PYNQ-Z2 Linux environment and execute:
```bash
python3 pynq_board_test.py
```

---

## 📜 Citations & References

```bibtex
@article{tbiocas_afib_2026,
  title={Resource-Efficient Edge Deployment of a Quantized PRV-Feature Neural Network for PPG-Based AFib Screening},
  author={Samrina, Md. and Indaram, Venkata Siddarth and Kumar, V. Ramesh},
  journal={IEEE Transactions on Biomedical Circuits and Systems (TBioCAS)},
  year={2026},
  note={Manuscript TBioCAS-2026-Aug-0332}
}
```

---

## ⚖️ License
This project is open-source under the **MIT License**. See [LICENSE](LICENSE) for full details.
