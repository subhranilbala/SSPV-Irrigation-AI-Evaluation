# M.Tech Thesis Architecture & Scientific Specification Guide

**Title:** AI Based Evaluation of Empirical Subsurface Porous Vessel (SSPV) Irrigation System Performance Using Images  
**Author:** Subhranil Bala (Roll No.: `M25AI2038`)  
**Supervisor:** Dr. Anand Krishnan Plappally  
**Department:** School of Artificial Intelligence & Data Science, IIT Jodhpur  
**Date:** April 2026  

---

## 1. Domain Background & Agronomic Foundation

### 1.1 The Agricultural Water Challenge
Agriculture consumes approximately **70% of global freshwater withdrawals**. Conventional surface irrigation (flood/furrow) and conventional drip systems incur substantial evaporation losses and non-uniform wetting patterns in arid environments like Western Rajasthan. Moreover, standard point sensors (soil moisture probes, tensiometers) lack spatial granularity across heterogeneous field plots.

### 1.2 Subsurface Porous Vessel (SSPV) Irrigation
Subsurface Porous Vessel (SSPV) irrigation is an engineered modernization of ancient clay pot (*matka* or *olla*) irrigation, developed and deployed through **RuTAG, IIT Delhi** in collaboration with Arpan Seva Sansthan and tested at agricultural research stations (e.g., Nagaur campus and Dehradun Doon Valley field sites):
1. **Frustum Geometry:** Unlike spherical pots which disturb the soil mound upon extraction, frustum-shaped clay vessels allow non-destructive installation and removal, supporting conservation agriculture.
2. **Capillary-Driven Demand:** Water discharges into the rhizosphere strictly under matric suction gradients created by root uptake and soil moisture deficits. When surrounding soil is near field capacity, seepage naturally ceases, achieving **30–50% water savings**.
3. **Saline Soil Reclamation & Regional Adaptability:** In arid Rajasthan, as water percolates outward through porous vessel walls, soluble salts are pushed outward down mound slopes. In foothill regions like Dehradun, Uttarakhand (where the real multi-date UAV imagery was captured), the system prevents rapid gravitational percolation and slope runoff, moderates root-zone temperatures, and eliminates weed flushes caused by flood irrigation.

---

## 2. End-to-End Pipeline Overview

The computational pipeline translates raw high-resolution drone imagery into biophysical insights and actionable irrigation scheduling across six interconnected stages:

```
[Raw UAV RGB GeoTIFF/JPG]
           │
           ▼
┌────────────────────────────────────────┐
│ Stage 1: UAV Preprocessing             │
│  - sRGB Inverse Gamma Linearization    │
│  - Dark Object Subtraction (DOS)       │
│  - SIFT Feature Affine Alignment       │
│  - Plot Boundary Clipping & Median     │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ Stage 2: Feature Engineering (11-Ch)   │
│  - 3 Radiance Bands: R, G, B           │
│  - 8 Vegetation Indices:               │
│    NGRDI, ExG, ExR, VARI, MGRVI,       │
│    NPCI, TGI, ExB                      │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ Stage 3: Biophysical Pseudo-Labeling   │
│  - Spectral-heuristic supervision      │
│  - Mound outer salt crust detection    │
│  - 4-Class masks & 3-head targets      │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ Stage 4: Deep Learning Inference       │
│  - 11-Channel U-Net (ResNet18 encoder) │
│  - Valid crop soft-gating              │
│  - Multi-Task CNN (Stress, Soil, Yield)│
│  - Grad-CAM Spatial Explainability     │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ Stage 5: Statistical Comparison Engine │
│  - Welch's two-sample t-test           │
│  - Multi-plot & multi-date ANOVA       │
│  - Cohen's d practical effect size     │
│  - Temporal OLS slope Z-test           │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ Stage 6: Decision Support & Dashboard  │
│  - Rule-based determinate logic        │
│  - Decision Tree indeterminate ranker  │
│  - Streamlit UI with multi-date trends │
└────────────────────────────────────────┘
```

---

## 3. Mathematical & Algorithmic Formulations

### 3.1 Preprocessing Pipeline

#### 3.1.1 Digital Number (DN) to Surface Reflectance Linearization
Commercial UAV sensors (e.g., DJI Zenmuse P1) capture 8-bit or 16-bit JPEG images encoded with non-linear sRGB transfer curves. To restore radiometric linearity proportional to scene radiance:
$$
\rho_{\text{linear}} = 
\begin{cases} 
\frac{\rho_n}{12.92}, & \rho_n \le 0.04045 \\
\left(\frac{\rho_n + 0.055}{1.055}\right)^{2.4}, & \rho_n > 0.04045 
\end{cases}
$$
where $\rho_n = \frac{\text{DN}}{255.0}$ (or $\frac{\text{DN}}{65535.0}$).

#### 3.1.2 Dark Object Subtraction (DOS) Atmospheric Correction
To eliminate additive path radiance caused by atmospheric aerosols in low-altitude drone surveys:
$$
\rho_{\text{corr}}(\lambda) = \frac{\max\left(0, \rho_{\text{linear}}(\lambda) - \text{Percentile}_{1.0}(\rho_{\text{linear}}(\lambda))\right)}{\text{Percentile}_{99.9}(\rho_{\text{linear}}(\lambda)) + \epsilon}
$$

#### 3.1.3 Multi-Date SIFT Affine Co-Registration
For multi-date orthomosaic patch alignment across survey flights, SIFT keypoints are matched using Euclidean distance with Lowe's ratio test ($d_1 < 0.75 d_2$). The 2D partial affine transformation matrix $M \in \mathbb{R}^{2 \times 3}$ is estimated via RANSAC:
$$
\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} s \cos\theta & -s \sin\theta & t_x \\ s \sin\theta & s \cos\theta & t_y \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

---

### 3.2 11-Channel Spectral Index Engineering

From normalized surface reflectances $R, G, B \in [0, 1]$, eight vegetation indices are derived pixel-wise:

| Index Name | Abbr. | Mathematical Formula | Agronomic Significance |
|---|---|---|---|
| **Normalized Green-Red Difference Index** | `NGRDI` | $\frac{G - R}{G + R}$ | Direct proxy for photosynthetic vigor and canopy greenness |
| **Excess Green Index** | `ExG` | $\frac{2G - R - B}{R + G + B}$ | Discriminated canopy vegetation from arid Rajasthan soil |
| **Excess Red Index** | `ExR` | $\frac{1.4R - G}{R + G + B}$ | Captures plant chlorosis, senescence, and soil background |
| **Visible Atmospherically Resistant Index**| `VARI` | $\frac{G - R}{G + R - B}$ | Minimizes atmospheric aerosol and illumination variation |
| **Modified Green-Red Vegetation Index** | `MGRVI` | $\frac{G^2 - R^2}{G^2 + R^2}$ | Quadratic sensitivity to chlorophyll biomass accumulation |
| **Normalized Pigment Chlorophyll Ratio** | `NPCI` | $\frac{B - R}{B + R}$ | Correlates with carotenoid-to-chlorophyll degradation under drought |
| **Triangular Greenness Index** | `TGI` | $G - 0.39R - 0.61B$ | Sensitive to relative leaf chlorophyll concentrations |
| **Excess Blue Index** | `ExB` | $1.4B - G$ | Detects wet soil surface moisture and waterlogging accumulation |

The resulting tensor:
$$
\mathbf{X} \in \mathbb{R}^{11 \times H \times W} = \left[ R, G, B, \text{NGRDI}, \text{ExG}, \text{ExR}, \text{VARI}, \text{MGRVI}, \text{NPCI}, \text{TGI}, \text{ExB} \right]^T
$$

---

### 3.3 Weakly-Supervised Biophysical Pseudo-Labeling

To overcome the lack of manual pixel-level segmentation annotations during initial deployment phases, a biophysical heuristic engine generates pseudo-labels:

#### 1. Mound Salt Halo Crust (Class 3)
White salt crusts pushed outward down the mound by SSPV capillary percolation exhibit high luminance, minimal color variation, and low vegetation greenness:
$$
\mathcal{M}_{\text{salt}} = \left( \frac{R+G+B}{3} > \text{P}_{92} \right) \land \left( \sigma_{RGB} < 0.06 \right) \land \left( \text{ExG} < 0.05 \right)
$$

#### 2. Canopy Segmentation (Classes 0, 1, 2)
- **Class 0 (Soil / Background):** $\text{ExG} \le \tau_{\text{exg}} \land \text{NGRDI} \le 0.04$
- **Class 1 (Healthy Canopy):** $\text{Candidate} \land (\text{NGRDI} \ge 0.05) \land (\text{TGI} \ge 0.01) \land \neg \mathcal{M}_{\text{salt}}$
- **Class 2 (Stressed Crop Zone):** $\text{Candidate} \land \neg \text{Healthy} \land (\text{ExR} > 0.0) \land \neg \mathcal{M}_{\text{salt}}$

#### 3. Patch Multi-Task Targets
- **Stress Severity ($y_s \in \{0, 1, 2\}$):** Derived from mean canopy NGRDI and ExR ($0 = \text{Healthy}, 1 = \text{Mild}, 2 = \text{Severe}$).
- **Soil Moisture Zone ($y_m \in \{0, 1, 2\}$):** Derived from soil luminance and ExB ($0 = \text{Dry}, 1 = \text{Optimal}, 2 = \text{Wet}$).
- **Yield Potential ($y_y \in \{0, 1, 2\}$):** Evaluated from biomass fraction and integrated chlorophyll index:
  $$S_y = 0.5 f_{\text{crop}} + 2.5 \max(0, \overline{\text{NGRDI}}) + 2.0 \max(0, \overline{\text{TGI}})$$

---

## 4. Deep Learning Neural Architectures

### 4.1 11-Channel First Convolution Adaptation
Pretrained ImageNet backbones expect 3 RGB input channels. To retain transfer learned spatial filters while absorbing the 8 spectral indices, the initial convolutional layer $W_{\text{conv1}} \in \mathbb{R}^{C_{\text{out}} \times 3 \times k \times k}$ is expanded to $W'_{\text{conv1}} \in \mathbb{R}^{C_{\text{out}} \times 11 \times k \times k}$:
$$
W'_{[:, 0:3, :, :]} = W_{\text{conv1}}
$$
$$
W'_{[:, 3:11, :, :]} = \frac{1}{3} \sum_{c=0}^2 W_{\text{conv1}[:, c, :, :]} \times 0.5
$$

### 4.2 11-Channel U-Net Architecture
- **Encoder:** ResNet18 layers 0–4 with receptive fields downsampling by $32\times$.
- **Decoder:** 4 stages of bilinear upsampling combined with skip connection concatenation and double $3\times3$ Conv-BN-ReLU blocks.
- **Output:** Softmax distribution over 4 classes ($C=4$).

### 4.3 Multi-Task CNN with Valid Crop Gating
To prevent dry soil pixels from biasing crop stress classification, the feature map entering the shared CNN backbone is softly gated by the U-Net crop canopy prediction:
$$
\mathbf{X}_{\text{gated}} = \mathbf{X} \odot \left(0.25 + 0.75 \cdot \mathbb{I}(\hat{y}_{\text{unet}} \in \{1, 2\})\right)
$$

### 4.4 Multi-Task Loss with Inverse Class Frequency Balancing
The joint optimization objective combines weighted cross-entropy losses:
$$
\mathcal{L}_{\text{total}} = \lambda_s \mathcal{L}_{\text{CE}}(\hat{y}_s, y_s; w_s) + \lambda_m \mathcal{L}_{\text{CE}}(\hat{y}_m, y_m; w_m) + \lambda_y \mathcal{L}_{\text{CE}}(\hat{y}_y, y_y; w_y)
$$
where class weights are dynamically computed from training partition frequencies:
$$
w_{k, c} = \frac{1 / N_{k, c}}{\sum_{j=1}^3 1 / N_{k, j}} \times 3.0
$$

---

## 5. Statistical Comparison Engine

To formally test the thesis hypothesis (*"SSPV-irrigated plots exhibit higher, spatially more uniform vegetation index levels and lower temporal variance compared to conventional control plots"*), four statistical tests are executed:

### 5.1 Welch's Two-Sample t-Test (Mean NGRDI)
Tests whether the mean canopy greenness difference is statistically significant without assuming equal variances:
$$
t = \frac{\bar{X}_{\text{sspv}} - \bar{X}_{\text{ctrl}}}{\sqrt{\frac{s_{\text{sspv}}^2}{n_1} + \frac{s_{\text{ctrl}}^2}{n_2}}}, \quad \nu \approx \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2 / n_1)^2}{n_1 - 1} + \frac{(s_2^2 / n_2)^2}{n_2 - 1}}
$$

### 5.2 One-Way ANOVA (Stress Pixel Proportions)
1. **Across Plot Types:** Tests variance between SSPV and Control plots:
   $$F_{\text{plot}} = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}}$$
2. **Across Survey Dates:** Tests stability of crop health over temporal dates.

### 5.3 Cohen's $d$ Effect Size (Practical Significance)
Quantifies whether the magnitude of difference is practically significant beyond $p$-values:
$$
d = \frac{\bar{X}_1 - \bar{X}_2}{s_{\text{pooled}}}, \quad s_{\text{pooled}} = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}
$$
- $|d| \ge 0.8$: Large effect (high agronomic impact).
- $|d| \ge 0.5$: Moderate effect.
- $|d| \ge 0.2$: Small effect.

### 5.4 Temporal Growth Linear Regression & Slope Z-Test
Fits ordinary least squares (OLS) linear trajectories $\text{NGRDI}(t) = \beta t + \alpha$ and calculates the Z-score for slope divergence between SSPV and Control:
$$
Z_{\text{slope}} = \frac{\beta_{\text{sspv}} - \beta_{\text{ctrl}}}{\sqrt{\text{SE}_{\beta_{\text{sspv}}}^2 + \text{SE}_{\beta_{\text{ctrl}}}^2}}
$$

---

## 6. Hybrid Recommendation Engine

The decision support system employs a two-tier hierarchy:

### Tier 1: Determinate Agronomic Rule Triggers
1. **Low NGRDI ($< 0.01$) + Low ExG ($< 0.05$):**  
   $\rightarrow$ `Urgent Irrigation Alert: Immediate SSPV Refill + Check Porous Wall Clogging` (Confidence: 0.94)
2. **Elevated ExB ($> 0.08$) + High Moisture Probability ($P_{\text{wet}} > 0.45$):**  
   $\rightarrow$ `Waterlogging Warning: Suspend Refill & Inspect Subsurface Drainage` (Confidence: 0.91)
3. **High NGRDI ($> 0.09$) + High VARI ($> 0.07$) + $P_{\text{healthy}} > 0.50$:**  
   $\rightarrow$ `Optimal: Maintain Current SSPV Capillary Delivery (No Intervention)` (Confidence: 0.95)
4. **Mound Outer Salt Ratio ($> 0.05$):**  
   $\rightarrow$ Generates secondary `Salinity Management Advisory` for scraping perimeter crusts.

### Tier 2: Decision Tree for Indeterminate Multi-Task States
When rules do not trigger, a 13-dimensional feature vector:
$$
\mathbf{v} = \left[ P(\text{stress}_{0..2}), P(\text{moist}_{0..2}), P(\text{yield}_{0..2}), \text{NGRDI}, \text{ExG}, \text{VARI}, \text{TGI} \right] \in \mathbb{R}^{13}
$$
is evaluated through a calibrated Decision Tree classifier, outputting a complete **ranked list of recommended actions with normalized confidence scores**.

---

## 7. Thesis Deliverables & Output Schema

When the pipeline executes (`python main.py` or `sspv_pipeline.orchestrator`), it exports:

1. `sspv_7date_metrics.csv`: Tabular evaluation metrics per patch across all 7 temporal survey dates.
2. `statistical_report.json`: Formatted JSON of all t-test stats, ANOVA $F$-values, Cohen's $d$, regression slope coefficients, and RuTAG capillary hydrology validation.
3. `unet_11ch_weights.pth`: PyTorch binary state dictionary of the trained 11-channel U-Net.
4. `multitask_cnn_11ch_weights.pth`: PyTorch binary state dictionary of the trained Multi-Task CNN.
5. `thesis_sspv_vs_control_summary.png`: 4-panel high-resolution (200 DPI) publication graphic:
   - Panel (0,0): Temporal NGRDI Growth Trend (SSPV vs. Control)
   - Panel (0,1): Crop Stress Pixel Proportion Boxplot (ANOVA)
   - Panel (1,0): Temporal ExG Stability (Testing Slide 16 hypothesis)
   - Panel (1,1): Temporal VARI Distribution Consistency
6. `MTP_Thesis_Evaluation_Chapter_Report.pdf`: Publication-grade formal academic evaluation chapter PDF formatted with IIT Jodhpur banners, candidate/supervisor details, statistical tables, RuTAG audit, and signature blocks.
7. `MTP_Thesis_Evaluation_Chapter_Report.txt`: Ready-to-use chapter text summarizing all findings in formal academic thesis format.

---

## 8. Academic References

1. Bainbridge, D. A. (2001). *Buried clay pot irrigation: A little known but very efficient traditional method of irrigation.* Agricultural Water Management, 48(2), 79–88.
2. Panigrahi, B., et al. (2012). *FEM modelling of wetting-front geometry around porous vessels in subsurface irrigation.* Irrigation Science.
3. Siyal, A. A., & Skaggs, T. H. (2009). *Measured and simulated soil wetting patterns under porous clay pipe sub-surface irrigation.* Agricultural Water Management, 96(6), 893–904.
4. Torres-Sánchez, J., et al. (2015). *Multi-temporal mapping of the vegetation fraction in early-season wheat using unmanned aerial vehicles.* Computers and Electronics in Agriculture.
5. Hamuda, E., et al. (2016). *A survey of image processing techniques for plant extraction and segmentation in the field.* Computers and Electronics in Agriculture, 125, 184–199.
6. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
7. Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* IEEE ICCV.
8. Zhang, Y., & Yang, Q. (2022). *A Survey on Multi-Task Learning.* IEEE TKDE.
9. Poblete, T., et al. (2020). *Detection of water stress in vineyards using hyperspectral and thermal imagery with machine learning algorithms.* Remote Sensing.
