# Resource-Efficient Edge Deployment of a Quantized PRV-Feature Neural Network for PPG-Based AFib Screening

Official open-source implementation repository for the revised manuscript submitted to **IEEE Transactions on Biomedical Circuits and Systems (TBioCAS)**:  
*Paper ID:* **TBioCAS-2026-Aug-0332**

---

## 🌟 Key Features & Pipeline Overview

1. **Clinical PPG Processing & 4× Physiological Data Augmentation:**
   - **Primary Dataset:** 35 clinical intensive care patients from **MIMIC-III PERform AF** (125 Hz transmissive finger-clip PPG).
   - **Physiological Augmentation:** Implemented following Cheng et al. (2020) using additive Gaussian sensor noise ($\sigma = 0.05 \cdot 	ext{STD}$), amplitude scaling ($lpha \sim \mathcal{N}(1.0, 0.15)$), and $0.2	ext{ Hz}$ respiratory baseline wander.
   - **Evaluated Dataset:** Expanded 4-fold to **7,706 evaluated 20-second windows** (**3,424 Non-AFib** and **4,282 AFib**).
   - **Physiological Feature Extraction:** 11 Pulse Rate Variability (PRV) statistical and frequency-domain features (Mean, Median, STD, RMSSD, pNN50, IQR, Shannon Entropy, Min/Max/Range, and Mean Absolute Deviation).

2. **Strict Subject-Wise Cross-Validation:**
   - `GroupKFold` partitioning strictly grouped by Patient ID, guaranteeing zero subject data leakage across train and test folds.
   - **5-Fold Cross-Validation Accuracy:** **95.80% ± 2.73%** (Fold 1: 99.07%, Fold 2: 98.58%, Fold 3: 93.79%, Fold 4: 95.59%, Fold 5: 91.95%).
   - **Held-Out Test Accuracy:** **98.06%** (software float) and **97.63%** (on-chip PYNQ-Z2 bitstream).

3. **Senior 8-Bit Profiled Post-Training Quantization (PTQ):**
   - Layer-specific profiling with dynamic overflow saturation:
     - Inputs & activations: `ap_fixed<8,5,AP_RND,AP_SAT>`
     - Hidden weights: `ap_fixed<8,1>`
     - Accumulators: `ap_fixed<16,8>`
   - Prevents catastrophic overflow while reducing on-chip BRAM and DSP utilization.

4. **Physical Silicon Verification on PYNQ-Z2 (Zynq-7020):**
   - Fully pipelined AXI4-Lite IP (`ReuseFactor = 1`, `Strategy = Latency`).
   - Hardware execution latency: **0.188 ms** per inference on Zynq-7020 PL.

5. **External Clinical Validation on Pulsewatch Smartwatch Dataset:**
   - **Independent Population:** 101 ambulatory smartwatch PPG subjects recorded at 50 Hz from the **Pulsewatch clinical trial** (Sage Bionetworks Synapse Accession `syn23565056`, obtained with formal permission from D. Han et al.).
   - **Ambulatory Generalization:** Evaluated across **2,180 segments (20 s each)**.
   - **Quantization Sweep Across Word-Lengths:**
     - Software Float32: **91.60%**
     - 16-bit Fixed-Point (`<16,6>`): **91.61%**
     - 12-bit Fixed-Point (`<12,5>`): **91.57%**
     - 6-bit Fixed-Point (`<6,3>`): **90.23%**
     - Senior Profiled 8-bit (`<8,5>`): **87.73%**
   - **Silicon Deployment (16-bit Baseline on PYNQ-Z2):**
     - Test Accuracy: **91.57%**
     - Sensitivity: **95.42%**
     - Specificity: **87.21%**
     - Hardware Latency: **0.325 ms**
     - Resource Footprint: **7 DSPs**, **0.26 W** dynamic power

---

## 📁 Repository Directory Structure

```text
TBioCAS_AFib_Edge_Deployment_GitHub_Repo/
├── .gitignore                                 # Git ignore rules (Jupyter, Python, Vivado temp files)
├── LICENSE                                    # Open-source MIT License
├── README.md                                  # Repository documentation
├── requirements.txt                           # Python dependencies
├── firmware_hls/                              # HLS4ML C++ firmware headers and AXI wrapper
│   ├── defines.h                              # Fixed-point typedefs (ap_fixed configurations)
│   ├── parameters.h                           # Layer dimensions, weights, and bias definitions
│   ├── myproject.h                            # Core neural network top function prototype
│   ├── myproject_axi.h                        # AXI4-Lite interface packet structures
│   └── myproject_axi.cpp                      # AXI4-Lite hardware interface wrapper
├── hardware_pynq/                             # Synthesized bitstreams and board drivers
│   ├── mimic_8bit_augmented/
│   │   ├── design_1.bit                       # Synthesized bitstream for Zynq-7020 PYNQ-Z2
│   │   ├── design_1.hwh                       # Hardware handoff definition
│   │   └── pynq_board_test.py                 # Python MMIO driver for live board inference
│   └── pulsewatch_16bit_clinical/
│       ├── design_1_pulsewatch.bit            # Pulsewatch 16-bit baseline bitstream
│       ├── design_1_pulsewatch.hwh            # Pulsewatch hardware handoff
│       └── pulsewatch_board_test.py           # PYNQ test script for clinical dataset
├── models/                                    # Pre-trained models and physiological feature scalers
│   ├── ppg_afib_model_augmented.h5            # Keras float32 trained baseline
│   └── feature_scaler_augmented.pkl           # StandardScaler fit strictly on training subjects
└── notebooks/                                 # Executed Jupyter workflows
    ├── 01_mimic_augmented_edge_pipeline.ipynb # End-to-end training, 5-fold CV, PTQ, and HLS IP export
    ├── 02_pulsewatch_clinical_validation.ipynb# Independent clinical trial validation (101 smartwatch patients)
    ├── 10_qat_vs_ptq.ipynb                    # Reviewer response: QAT vs. PTQ across bitwidths
    ├── 11_rounding_overflow_experiment.ipynb  # Reviewer response: Rounding and saturation modes
    ├── 12_reuse_factor_sweep.ipynb            # Reviewer response: Latency vs. DSP resource trade-offs
    ├── 13_loso_confidence_intervals.ipynb     # Reviewer response: Leave-One-Subject-Out 95% CIs
    ├── 14_baseline_classifiers.ipynb          # Reviewer response: SVM, Random Forest, XGBoost baselines
    └── 15_filter_sensitivity.ipynb            # Reviewer response: Bandpass filter cutoff sensitivity
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
git clone https://github.com/<your-username>/TBioCAS-PPG-AFib-Edge-Zynq.git
cd TBioCAS-PPG-AFib-Edge-Zynq
pip install -r requirements.txt
```

### 2. Running the Core Pipelines
* **Augmented Edge Pipeline:** Open `notebooks/01_mimic_augmented_edge_pipeline.ipynb` to reproduce the physiological feature extraction, 5-fold cross-validation, 8-bit profiled PTQ, and Vivado HLS IP export.
* **Pulsewatch Validation:** Open `notebooks/02_pulsewatch_clinical_validation.ipynb` to evaluate the 101-patient smartwatch clinical trial across Float32 and fixed-point word-lengths.
* **Reviewer Ablations:** Run notebooks `10` through `15` to reproduce all sensitivity and benchmarking experiments.

### 3. Live PYNQ-Z2 Hardware Inference
Transfer `hardware_pynq/mimic_8bit_augmented/` to the PYNQ-Z2 Linux environment and execute:
```bash
python3 pynq_board_test.py
```

---

## 📊 Summary of Silicon & Clinical Evaluation

| Dataset & Task | Platform | Quantization | Accuracy | Sensitivity | Specificity | Latency | DSPs | Power |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MIMIC-III (Augmented)** | PYNQ-Z2 (Zynq-7020) | 8-bit Profiled (`<8,5>`) | **97.63%** | **98.24%** | **96.81%** | **0.188 ms** | 30 | 0.28 W |
| **Pulsewatch (Clinical Trial)** | PYNQ-Z2 (Zynq-7020) | 16-bit Baseline (`<16,6>`) | **91.57%** | **95.42%** | **87.21%** | **0.325 ms** | 7 | 0.26 W |

---

## 📜 Citations & References

1. **Primary Manuscript:**
   ```bibtex
   @article{tbiocas_afib_2026,
     title={Resource-Efficient Edge Deployment of a Quantized PRV-Feature Neural Network for PPG-Based AFib Screening},
     journal={IEEE Transactions on Biomedical Circuits and Systems (TBioCAS)},
     year={2026},
     note={Manuscript TBioCAS-2026-Aug-0332}
   }
   ```
2. **Clinical Datasets:**
   - **MIMIC-III PERform AF:** PhysioNet open-access clinical database.
   - **Pulsewatch Clinical Trial:** D. Han et al., Sage Bionetworks Synapse `syn23565056` (used with formal permission from the authors).
   - **Data Augmentation:** Cheng et al. (2020), Physiological time-series augmentation for wearable biosensors.

---

## ⚖️ License
This project is open-source under the **MIT License**. See [LICENSE](LICENSE) for full legal text.
