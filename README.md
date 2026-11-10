# AI-Based Evaluation of Empirical Subsurface Porous Vessel (SSPV) Irrigation System Performance Using Drone Imagery

**Master Thesis Project (MTP) — 2026**  
**Candidate:** Subhranil Bala (Roll No.: `M25AI2038`)  
**Supervisor:** Dr. Anand Krishnan Plappally  
**Department:** School of Artificial Intelligence & Data Science (AIDE)  
**Institute:** Indian Institute of Technology Jodhpur (IIT Jodhpur)  

---

## 📌 Executive Summary

Subsurface Porous Vessel (SSPV) irrigation is an ancient yet modernised water-conservation technique (refined by RuTAG, IIT Delhi) utilizing buried porous clay vessels on agricultural soil mounds. Water seeps into the soil strictly via capillary suction governed by crop moisture transpiration, reducing evaporation losses by **30–50%** compared to conventional surface irrigation and mitigating soil salinity.

This repository provides an **end-to-end AI-driven evaluation and decision-support pipeline** using drone-borne high-resolution RGB imagery (e.g., DJI Zenmuse P1 at 8192×5460 resolution captured over experimental agricultural plots in Dehradun, Uttarakhand) to evaluate and compare SSPV irrigation against conventional control plots across multiple temporal dates.

---

## 🏗️ System Architecture Highlights

```
                                  +------------------------------+
                                  |     Drone Imagery Input      |
                                  | (DJI Zenmuse P1 / GeoTIFFs)  |
                                  +--------------+---------------+
                                                 |
                                                 v
                                  +------------------------------+
                                  |     UAV Preprocessing        |
                                  |  - sRGB Radiance Linearize   |
                                  |  - Dark Object Subtr. (DOS)  |
                                  |  - SIFT Affine Co-reg.       |
                                  |  - ROI Clipping & Median     |
                                  +--------------+---------------+
                                                 |
                                                 v
                                  +------------------------------+
                                  | 11-Channel Feature Tensor    |
                                  |  - R, G, B                   |
                                  |  - 8 Spectral Indices        |
                                  |    (NGRDI, ExG, ExR, VARI,   |
                                  |     MGRVI, NPCI, TGI, ExB)   |
                                  +--------------+---------------+
                                                 |
                       +-------------------------+-------------------------+
                       |                                                   |
                       v                                                   v
         +----------------------------+                      +----------------------------+
         |     U-Net Segmentation     |                      |       Multi-Task CNN       |
         | (ResNet18 Encoder, 11-ch)  |                      | (ResNet18/EfficientNet-B0) |
         |   4 Classes:               |  Valid Crop Gating   |  Simultaneous Predictions: |
         |   0: Bare Soil             | -------------------> |   1. Stress Severity       |
         |   1: Healthy Canopy        |                      |   2. Soil Moisture Zone    |
         |   2: Stressed Crop Zone    |                      |   3. Yield Potential       |
         |   3: SSPV Salt Halo Crust  |                      +--------------+-------------+
         +----------------------------+                                     |
                                                                            v
                                                             +----------------------------+
                                                             |    Grad-CAM XAI Engine     |
                                                             |  Spatial visual attention  |
                                                             |   across 11 input bands    |
                                                             +--------------+-------------+
                                                                            |
                                                 +--------------------------+
                                                 |
                                                 v
                                  +------------------------------+
                                  |  Statistical Engine & XAI    |
                                  |  - Welch's Two-Sample t-test |
                                  |  - Multi-Plot & Date ANOVA   |
                                  |  - Cohen's d Effect Size     |
                                  |  - Temporal Slope Z-test     |
                                  +--------------+---------------+
                                                 |
                                                 v
                                  +------------------------------+
                                  | Hybrid Recommendation Engine |
                                  |  - Rule-Based Determinate    |
                                  |  - Decision Tree Multi-Prob  |
                                  |  - Ranked Action Confidence  |
                                  +--------------+---------------+
                                                 |
                                                 v
                                  +------------------------------+
                                  |     Streamlit Dashboard      |
                                  |  Visual heatmaps, charts,    |
                                  |  downloadable thesis report  |
                                  +------------------------------+
```

---

## 📁 Repository Structure

```
SSPV-Irrigation-AI-Evaluation/
├── main.py                             # Unified CLI & Streamlit runner
├── requirements.txt                    # Project dependencies
├── ARCHITECTURE_DIAGRAMS.md            # Complete architecture diagrams (Mermaid & SVG references)
├── Architectures/                      # Rendered high-resolution vector architecture SVGs
│   ├── Complete End-to-End System Architecture.svg
│   ├── 11-Channel U-Net Field Segmentation.svg
│   ├── 11-Channel Multi-Task CNN with Crop Gating.svg
│   └── Hybrid Recommendation Engine Architecture.svg
├── MTP_PROJECT_DETAILED_ANALYSIS.md    # Comprehensive MTP evaluation vs. PPT presentation
├── USER_EXECUTION_GUIDE.md             # Detailed Execution, Inputs & Outputs Guide
├── SYSTEM_ARCHITECTURE_GUIDE.md        # Comprehensive technical & scientific guide
│
├── docs/                               # Detailed Module-by-Module Documentation
│   ├── README.md                       # Documentation index
│   └── modules/                        # 11 Dedicated module guides (01_config.md .. 11_dashboard.md)
│
├── sspv_pipeline/                      # Modular Python Package
│   ├── __init__.py                     # Package entry point & API exports
│   ├── config.py                       # Constants, hardware auto-detect (CUDA/MPS/CPU)
│   ├── preprocessing.py                # DN→Reflectance, DOS, SIFT alignment, filtering
│   ├── feature_engineering.py          # 8 RGB indices & 11-channel tensor assembly
│   ├── pseudo_labeling.py              # Biophysical weak supervision & salt patch detection
│   ├── models.py                       # 11-ch U-Net, Multi-Task CNN & inverse-freq loss
│   ├── explainability.py               # Selvaraju et al. Grad-CAM for 11-channel inputs
│   ├── statistics_engine.py            # t-tests, ANOVA (plots/dates), Cohen's d, regression
│   ├── recommendation.py               # Hybrid rule-based & Decision Tree advisory
│   ├── dataset.py                      # Multi-date DJI folder loader & 7-date benchmark
│   ├── orchestrator.py                 # Pipeline training, evaluation & artifact exporter
│   └── dashboard.py                    # Interactive Streamlit UI
│
├── tests/                              # Comprehensive Unit & Regression Tests
│   └── test_pipeline.py                # 12 test suites covering all 11 modules
│
├── archive/                            # Legacy & Archived Code
│   ├── README.md
│   └── sspv_mtp_pipeline.py            # Legacy monolithic single-file version
│
├── outputs/                            # Exported weights & thesis artifacts (generated)
│   ├── unet_11ch_weights.pth
│   ├── multitask_cnn_11ch_weights.pth
│   ├── sspv_7date_metrics.csv
│   ├── statistical_report.json
│   ├── thesis_sspv_vs_control_summary.png
│   ├── MTP_Thesis_Evaluation_Chapter_Report.pdf # Publication-grade formal thesis chapter PDF
│   └── MTP_Thesis_Evaluation_Chapter_Report.txt # Ready-to-use chapter text draft
│
├── .sspv_cache/                        # Persistent disk cache for instant patch reload (~0.001s)
└── venv/                               # Python virtual environment
```

---

## ⚡ Quickstart & Installation

### 1. Environment Setup
Clone or enter the project directory and activate the virtual environment:
```bash
cd SSPV-Irrigation-AI-Evaluation
source venv/bin/activate
```

Dependencies required:
```bash
pip install -r requirements.txt
# Or manually:
pip install torch torchvision numpy scipy pandas matplotlib seaborn scikit-learn pillow streamlit rasterio opencv-python reportlab
```

*Note: On Apple Silicon Macs (M1/M2/M3/M4), the pipeline automatically uses the `mps` (Metal Performance Shaders) backend for accelerated deep learning.*

---

## 🚀 Running the Pipeline

### Mode A: Interactive Streamlit Dashboard (Recommended)
Launch the graphical user interface:
```bash
streamlit run main.py
```
**Dashboard Features:**
1. **Interactive Sidebar:** Select Agro-Climatic Zone (`dehradun_valley` [Default], Jodhpur Aridisol, Nagaur, etc.), Multi-Task CNN backbone (`ResNet18` or `EfficientNet-B0`), set training epochs, adjust image frame sampling (default `5` for instant 5s startup), target specific survey flight dates and image frames, or drag-and-drop a raw aerial drone photo (`.JPG`/`.TIF`). Stretched width ensures all metadata displays without clipping.
2. **Top KPI Row:** Instant comparison of SSPV vs. Control mean NGRDI, Welch's t-test p-value, Cohen's d effect size, and temporal variance.
3. **Section 1 (Spatial Diagnostics):** Side-by-side visualization of RGB, NGRDI canopy greenness heatmap, U-Net 4-class segmentation mask, and Grad-CAM spatial attention map with active image filenames displayed underneath.
4. **Section 2a & 2b (Temporal Trends):** Four-panel graphs tracking NGRDI growth, stress pixel proportion boxplot, ExG stability, and VARI consistency across all survey dates.
5. **Section 3 (Actionable Recommendations):** Full per-patch metrics table with real image filenames, confidence scores, and one-click ReportLab PDF, CSV, and text downloads.
6. **Section 4 (RuTAG Human Ecology Audit):** Real-time evaluation of capillary moisture stability (Gangotri percolation buffer), root-zone desalination protection, and estimated groundwater conservation (~38.5%).
7. **Section 5 (Smallholder Field Advisory):** Bilingual (English/Hindi) smallholder field advisories with native spoken voice audio playback (`st.audio`).

---

### Mode B: Command-Line Interface (CLI Execution)
Run batch training, evaluation, and artifact generation headlessly:
```bash
# Run with ResNet18 backbone
python main.py --mode run_all --backbone resnet18 --epochs 4 --output_dir ./outputs

# Run with EfficientNet-B0 backbone and ImageNet pretraining
python main.py --mode run_all --backbone efficientnet_b0 --pretrained --epochs 5 --output_dir ./outputs_effnet
```

#### CLI Command Arguments:
| Argument | Type | Default | Description |
|---|---|---|---|
| `--mode` | str | `run_all` | Execution mode (`run_all` or `verify`) |
| `--data_dir` | str | `~/Downloads/Crops Data` | Path to folder containing multi-date DJI drone images |
| `--backbone` | str | `resnet18` | Shared CNN backbone: `resnet18` or `efficientnet_b0` |
| `--max_images_per_date`| int | `20` | Max frames sampled per date folder (set `0` for ALL frames) |
| `--epochs` | int | `4` | Number of training epochs |
| `--batch_size` | int | `8` | Training batch size |
| `--pretrained` | flag | `False` | Enable ImageNet pretrained weights adapted to 11 channels |
| `--output_dir` | str | `./outputs` | Directory to save CSVs, JSON reports, plots, and weights |

---

## 📥 Detailed Inputs & Outputs Summary

> 📘 **For exhaustive step-by-step documentation, see [`USER_EXECUTION_GUIDE.md`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/USER_EXECUTION_GUIDE.md).**

### 📥 Inputs Accepted by the System:
1. **Multi-Date Drone Folders:** Folders named by flight date (e.g., `12-11-2024/images/*.JPG`) containing high-resolution DJI Zenmuse P1 captures ($8192\times5460$) or GeoTIFFs. Upper-right sectors map to SSPV mounds; lower-right sectors map to Control rows.
2. **Single Image Upload:** Any individual drone frame (`.JPG`/`.TIF`) dropped into the Streamlit sidebar for instant spatial segmentation and Grad-CAM diagnosis.
3. **7-Date Benchmark Simulator (Fallback):** Generates 42 multi-date patches modeling circular SSPV vessels, wetted capillary zones, expanding canopy halos, and outer perimeter white salt crusts (Slide 7) alongside conventional control plots.

### 📤 Outputs Produced by the System:
1. **`sspv_7date_metrics.csv`:** Full tabular metrics per patch (mean NGRDI, ExG, VARI, TGI, ExB, stress pixel ratio, salt patch ratio, predicted stress/moisture/yield classes, top recommendation, confidence, and bilingual RuTAG farmer actions).
2. **`statistical_report.json`:** Inferential stats (Welch's t-test $t$ and $p$-value, One-Way ANOVA $F$-values across plots and survey dates, Cohen's $d$ effect sizes, temporal regression slope $Z$-test, and RuTAG capillary hydrology validation).
3. **`unet_11ch_weights.pth` & `multitask_cnn_11ch_weights.pth`:** Serialized PyTorch weights for the 11-channel U-Net and Multi-Task CNN.
4. **`thesis_sspv_vs_control_summary.png`:** 4-panel publication graphic (200 DPI) displaying NGRDI growth trajectory, stress proportion boxplot, ExG stability, and VARI distribution across all survey dates.
5. **`MTP_Thesis_Evaluation_Chapter_Report.pdf`:** Formal, publication-grade academic evaluation chapter PDF featuring IIT Jodhpur headers, candidate/supervisor details, statistical tables, RuTAG audit, embedded publication graphics, and formal signature blocks.
6. **`MTP_Thesis_Evaluation_Chapter_Report.txt`:** Formatted academic thesis chapter draft ready for direct insertion into LaTeX or Word.

---

## 🔬 Scientific Methodology & Presentation Alignment

| Slide | Requirement | Implementation Module |
|---|---|---|
| **Slide 7** | SSPV White Salt Patch detection | [`pseudo_labeling.py`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/pseudo_labeling.py) Class 3 mask |
| **Slide 10** | System Architecture & Tech Stack | [`sspv_pipeline/`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline) End-to-end modular design |
| **Slide 11** | DN→Reflectance, DOS, SIFT Affine, ROI | [`preprocessing.py`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/preprocessing.py) [`UAVPreprocessor`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/preprocessing.py#L31) |
| **Slide 12** | 8 RGB Vegetation Indices | [`feature_engineering.py`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/feature_engineering.py) (NGRDI, ExG, ExR, VARI, MGRVI, NPCI, TGI, ExB) |
| **Slide 13** | 11-Channel U-Net & Multi-Task CNN | [`models.py`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/models.py) [`UNetResNet18_11Ch`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/models.py#L88) + [`MultiTaskSSPVNet`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/models.py#L152) |
| **Slide 14** | Statistical Engine & Decision Tree | [`statistics_engine.py`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/statistics_engine.py) + [`recommendation.py`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/recommendation.py) |
| **Slide 16** | Spectral Signature Hypothesis Testing | [`statistics_engine.py`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/statistics_engine.py) (ExG/VARI temporal stability & slope comparison) |
| **Slide 17** | 4-Phase Research Roadmap Execution | Integrated automated benchmark & full artifact export |

For in-depth mathematical formulations, biophysical pseudo-labeling rules, and statistical hypothesis tests, refer to [SYSTEM_ARCHITECTURE_GUIDE.md](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/SYSTEM_ARCHITECTURE_GUIDE.md).

---

## 👨‍🔬 Authorship & Academic Attribution

- **Researcher:** Subhranil Bala (`M25AI2038`), M.Tech Artificial Intelligence, Cohort 2
- **Supervisor:** Dr. Anand Krishnan Plappally
- **Institution:** Indian Institute of Technology Jodhpur (IITJ)
- **Year:** 2026
