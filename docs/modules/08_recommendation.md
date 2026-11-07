# Module Documentation: `recommendation.py`

**File Path:** [`sspv_pipeline/recommendation.py`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/recommendation.py)  
**Component:** Hybrid Logic-and-ML Decision Support Recommendation Engine  
**Presentation Mapping:** Slide 9 (Objective 6), Slide 10 (Decision Support), Slide 14 (Hybrid Engine)  

---

## 1. Module Overview & Purpose

Translating raw deep learning model predictions into actionable farming guidance is a cornerstone of this M.Tech thesis. `recommendation.py` implements a **two-tier hybrid recommendation engine**:
- **Tier 1:** Deterministic rule-based logic for clear-cut, extreme agronomic states.
- **Tier 2:** Multi-class Decision Tree classifier for complex, indeterminate multi-factor situations.

Slide 14 specifically requires: *"Ranked recommendations with associated confidence scores"*. This module generates a normalized probability distribution across all candidate actions and returns them sorted by confidence.

---

## 2. Action Taxonomy

The engine evaluates five standard agronomic interventions:
1. `Optimal: Maintain Current SSPV Capillary Delivery (No Intervention)`
2. `Moderate Refill: Replenish SSPV Vessels to 75% Capacity Within 24h`
3. `Urgent Irrigation Alert: Immediate SSPV Refill + Check Porous Wall Clogging`
4. `Waterlogging Warning: Suspend Refill & Inspect Subsurface Drainage`
5. `Salinity Management: Scrape Outer Salt Crust on Mound Perimeter`

---

## 3. Two-Tier Decision Architecture

### Tier 1: Deterministic Rule-Based Logic
Triggered when clear physical thresholds are violated:

1. **Urgent Irrigation Alert (Low NGRDI + Low ExG):**
   - Condition: $\text{NGRDI} < 0.01 \land \text{ExG} < 0.05$
   - Assigned Confidence: $0.94$
   - Meaning: Crops are undergoing active moisture stress and low chlorophyll greenness; immediate refill needed.
2. **Waterlogging Warning (Elevated ExB):**
   - Condition: $\text{ExB} > 0.08 \land P(\text{Wet}) > 0.45$
   - Assigned Confidence: $0.91$
   - Meaning: Excessive water seepage is creating anoxic root conditions; suspend vessel refills.
3. **Optimal Condition (High NGRDI + High VARI):**
   - Condition: $\text{NGRDI} > 0.09 \land \text{VARI} > 0.07 \land P(\text{Healthy}) > 0.50$
   - Assigned Confidence: $0.95$
   - Meaning: Equilibrium capillary delivery is meeting crop transpiration demands perfectly.

### Tier 2: Decision Tree Classifier (Indeterminate Conditions)
When none of the extreme rules trigger, the state is classified via a calibrated Decision Tree (`max_depth=5`):
- **Input Feature Vector ($\mathbf{v} \in \mathbb{R}^{13}$):**
  $$\mathbf{v} = \begin{bmatrix} P(\text{Healthy}), P(\text{Mild}), P(\text{Severe}), P(\text{Dry}), P(\text{Optimal}), P(\text{Wet}), P(\text{Low}), P(\text{Med}), P(\text{High}), \text{NGRDI}, \text{ExG}, \text{VARI}, \text{TGI} \end{bmatrix}$$
- **Pre-calibration:** Trained on 600 synthetic domain patterns encoding agronomic interactions between probability margins and spectral indices.

### Secondary Salinity Advisory (Slide 7)
If the mound white salt crust ratio exceeds 5% ($\text{salt\_ratio} > 0.05$):
- Appends an advisory confirming that SSPV outward percolation is functioning as designed by leaching salts away from the central root zone.

---

## 4. Class Structure: `HybridRecommendationEngine`

### Core Method: `recommend`
```python
def recommend(
    self,
    ngrdi: float,
    exg: float,
    vari: float,
    tgi: float,
    exb: float,
    salt_ratio: float,
    stress_probs: np.ndarray,
    moisture_probs: np.ndarray,
    yield_probs: np.ndarray
) -> Dict[str, Any]
```

### Return Dictionary Structure:
- `"primary_recommendation"`: Top-ranked string action.
- `"confidence"`: Float confidence in $[0, 1]$.
- `"engine_mode"`: Description of whether Tier 1 rule or Tier 2 Decision Tree resolved the state.
- `"triggers"`: List of specific numerical conditions that fired.
- `"secondary_advisory"`: Salt crust management message (if applicable).
- `"ranked_recommendations"`: List of dicts `[{"rank": 1, "action": ..., "confidence": ...}, ...]` sorted in descending order of confidence.
- `"rutag_farmer_advisory"`: Grassroots smallholder action dict:
  - `"action_en"`: Plain-English farmer instruction (e.g. vessel refilling or pot silt scraping).
  - `"action_hi"`: Native Hindi instruction (e.g. *"मटके में पानी भरें और भीतरी सतह साफ करें"*).
  - `"urgency"`: Urgency level (`"Normal"`, `"Medium"`, `"High"`).
  - `"estimated_water_saved_liters_day"`: Groundwater conserved in liters/day compared to surface flooding.

---

## 5. Example Usage

```python
import numpy as np
from sspv_pipeline.recommendation import HybridRecommendationEngine

engine = HybridRecommendationEngine()

advice = engine.recommend(
    ngrdi=0.12, exg=0.18, vari=0.09, tgi=0.03, exb=0.01, salt_ratio=0.02,
    stress_probs=np.array([0.85, 0.10, 0.05]),
    moisture_probs=np.array([0.10, 0.82, 0.08]),
    yield_probs=np.array([0.05, 0.25, 0.70])
)

print("Primary Action:", advice["primary_recommendation"])
print("Engine Mode:", advice["engine_mode"])
for item in advice["ranked_recommendations"]:
    print(f"#{item['rank']}: {item['action']} ({item['confidence']*100:.1f}%)")
```
