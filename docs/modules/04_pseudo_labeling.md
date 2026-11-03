# Module Documentation: `pseudo_labeling.py`

**File Path:** [`sspv_pipeline/pseudo_labeling.py`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/pseudo_labeling.py)  
**Component:** Weakly-Supervised Biophysical Pseudo-Labeling Engine  
**Presentation Mapping:** Slide 7 (SSPV Salinity Removal) & Slide 13 (AI Architecture Targets)  

---

## 1. Module Overview & Purpose

In real-world precision agriculture deployments, obtaining manual pixel-by-pixel segmentation masks and synchronized in-situ IoT soil moisture probe readings for thousands of drone flight images is prohibitively labor-intensive and expensive.

`pseudo_labeling.py` solves this unannotated data cold-start problem by applying **biophysical domain rules** to automatically generate:
1. **4-Class Semantic Segmentation Masks:** Pixel-level labels for U-Net training.
2. **Multi-Task Classification Targets:** Patch-level labels for the 3 heads of the Multi-Task CNN (Stress, Moisture, Yield).

---

## 2. Pseudo-Label Generation Mechanics

### 2.1 4-Class Segmentation Mask Generation (`generate_segmentation_mask`)
The algorithm processes the RGB reflectance and computed indices:

1. **White Salt Patch Detection on SSPV Mounds (Class 3):**
   - *Physical Mechanism (Slide 7):* Water seeping through the porous walls of the SSPV moves dissolved minerals and salts radially away from root zones. As evaporation occurs at the dry air interface, a bright, achromatic saline crust forms on the outer mound slope.
   - *Detection Rule:*
     $$\mathcal{M}_{\text{salt}} = \left( \frac{R+G+B}{3} > \text{P}_{92}(\text{Luminance}) \right) \land (\sigma_{RGB} < 0.06) \land (\text{ExG} < 0.05)$$
     Pixels matching this condition are designated `Class 3`.

2. **Vegetation Candidate Thresholding:**
   - Evaluates adaptive ExG threshold:
     $$\tau_{\text{exg}} = \max\left(0.06, \text{Percentile}_{65}(\text{ExG})\right)$$
   - Candidate vegetation: $(\text{ExG} > \tau_{\text{exg}}) \lor (\text{NGRDI} > 0.04)$.

3. **Healthy vs. Stressed Vegetation Partitioning:**
   - **Class 1 (Healthy Canopy):** $\text{Candidate} \land (\text{NGRDI} \ge 0.05) \land (\text{TGI} \ge 0.01) \land \neg \mathcal{M}_{\text{salt}}$
   - **Class 2 (Stressed Crop Zone):** $\text{Candidate} \land \neg \text{Healthy} \land (\text{ExR} > 0.0) \land \neg \mathcal{M}_{\text{salt}}$
   - **Class 0 (Soil / Background):** All remaining non-vegetated substrate pixels.

---

### 2.2 Patch-Level Multi-Task Target Derivation (`generate_multitask_labels`)

Extracts patch-level summary statistics from valid crop pixels:

1. **Crop Stress Severity ($y_s \in \{0, 1, 2\}$):**
   - $y_s = 0$ (`Healthy`): $\overline{\text{NGRDI}} > 0.08 \land \overline{\text{ExR}} < 0.12$
   - $y_s = 1$ (`Mild Stress`): $\overline{\text{NGRDI}} > 0.01$
   - $y_s = 2$ (`Severe Stress`): $\overline{\text{NGRDI}} \le 0.01$ (chlorosis / water starvation)

2. **Soil Moisture Zone ($y_m \in \{0, 1, 2\}$):**
   - Arid Rajasthan sandy soils darken considerably when wetted by SSPV capillary seepage.
   - $y_m = 2$ (`Wet`): $\overline{\text{ExB}} > 0.05 \lor \overline{R}_{\text{soil}} < 0.28$ (surface saturation)
   - $y_m = 1$ (`Optimal`): $\overline{R}_{\text{soil}} < 0.48 \land y_s \le 1$ (stable capillary equilibrium)
   - $y_m = 0$ (`Dry`): Bright desiccated soil surface

3. **Yield Potential ($y_y \in \{0, 1, 2\}$):**
   - Evaluates integrated vigor index:
     $$S_y = 0.5 f_{\text{crop}} + 2.5 \max(0, \overline{\text{NGRDI}}) + 2.0 \max(0, \overline{\text{TGI}})$$
   - $y_y = 2$ (`High`): $S_y > 0.35$
   - $y_y = 1$ (`Medium`): $0.15 < S_y \le 0.35$
   - $y_y = 0$ (`Low`): $S_y \le 0.15$

---

## 3. Example Usage

```python
from sspv_pipeline.feature_engineering import VegetationIndexExtractor
from sspv_pipeline.pseudo_labeling import BiophysicalPseudoLabeler

vi_extractor = VegetationIndexExtractor()
_, indices = vi_extractor.build_11ch_tensor(rgb_image)

seg_mask = BiophysicalPseudoLabeler.generate_segmentation_mask(rgb_image, indices)
stress, moisture, yield_pot = BiophysicalPseudoLabeler.generate_multitask_labels(rgb_image, indices, seg_mask)

print("Salt pixels detected:", (seg_mask == 3).sum())
print(f"Targets -> Stress: {stress}, Moisture: {moisture}, Yield: {yield_pot}")
```
