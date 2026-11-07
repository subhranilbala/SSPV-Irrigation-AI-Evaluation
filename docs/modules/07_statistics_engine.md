# Module Documentation: `statistics_engine.py`

**File Path:** [`sspv_pipeline/statistics_engine.py`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/statistics_engine.py)  
**Component:** Inferential Statistical Comparison Engine  
**Presentation Mapping:** Slide 9 (Objective 5), Slide 14 (Statistical Comparison Engine), Slide 16 (Hypothesis Testing)  

---

## 1. Module Overview & Purpose

Precision agriculture thesis experiments require rigorous mathematical validation to substantiate claims of technological superiority. `statistics_engine.py` provides the formal inferential statistical testing framework comparing SSPV-irrigated crops against conventional row-irrigated control plots across multi-date drone surveys.

---

## 2. Statistical Tests Implemented

### 2.1 Welch's Two-Sample t-Test (Mean NGRDI per Plot)
Evaluates whether the mean canopy greenness difference between SSPV and Control plots is statistically significant without assuming homoscedasticity (equal variance):
$$
t = \frac{\bar{X}_{\text{sspv}} - \bar{X}_{\text{ctrl}}}{\sqrt{\frac{s_{\text{sspv}}^2}{n_1} + \frac{s_{\text{ctrl}}^2}{n_2}}}
$$
Degrees of freedom calculated via the Welch–Satterthwaite equation:
$$
\nu \approx \frac{\left( \frac{s_1^2}{n_1} + \frac{s_2^2}{n_2} \right)^2}{\frac{(s_1^2 / n_1)^2}{n_1 - 1} + \frac{(s_2^2 / n_2)^2}{n_2 - 1}}
$$
Output: $t$-statistic, $p$-value, and significance flag at $\alpha = 0.05$.

---

### 2.2 One-Way ANOVA (Stress Pixel Proportions)
Tests variance in crop stress pixel proportion across categorical groups:
$$
F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}} = \frac{\sum_{i=1}^k n_i (\bar{Y}_i - \bar{Y})^2 / (k - 1)}{\sum_{i=1}^k \sum_{j=1}^{n_i} (Y_{ij} - \bar{Y}_i)^2 / (N - k)}
$$
Executed in two dimensions:
1. **Across Plot Types ($F_{\text{plot}}$):** Quantifies variance between SSPV vs. Control.
2. **Across Multi-Date Surveys ($F_{\text{dates}}$):** Quantifies whether stress levels significantly fluctuate across flight dates.

---

### 2.3 Cohen's $d$ Effect Size (Practical Significance)
While $p$-values indicate whether a difference is non-zero, Cohen's $d$ quantifies the practical agronomic magnitude of the effect:
$$
d = \frac{\bar{X}_1 - \bar{X}_2}{s_{\text{pooled}}}, \quad s_{\text{pooled}} = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}
$$
Evaluated on:
1. **Canopy Greenness ($d_{\text{ngrdi}}$)**
2. **Crop Stress Reduction ($d_{\text{stress}}$)**
3. **Soil Moisture Zone Variance ($d_{\text{moisture}}$)** (Slide 14)

Qualitative interpretation:
- $|d| \ge 0.8$: Large Effect (High Practical Agronomic Significance)
- $|d| \ge 0.5$: Medium Effect
- $|d| \ge 0.2$: Small Effect
- $|d| < 0.2$: Negligible Effect

---

### 2.4 Temporal Linear Growth Regression & Inter-Plot Slope Z-Test
Fits ordinary least squares (OLS) linear trajectories over survey days:
$$\text{NGRDI}_{\text{sspv}}(t) = \beta_1 t + \alpha_1, \quad \text{NGRDI}_{\text{ctrl}}(t) = \beta_2 t + \alpha_2$$
Tests if the growth rate divergence $(\beta_1 - \beta_2)$ is statistically significant:
$$
Z = \frac{\beta_1 - \beta_2}{\sqrt{\text{SE}_{\beta_1}^2 + \text{SE}_{\beta_2}^2}}, \quad p = 2 \cdot (1 - \Phi(|Z|))
$$
Also computes temporal standard deviation ($\sigma_{\text{exg}}$) across dates to validate the Slide 16 hypothesis:
$$\sigma_{\text{exg, sspv}} < \sigma_{\text{exg, ctrl}}$$

---

## 3. Class Structure: `StatisticalComparisonEngine`

### Core Method: `run_full_comparison`
```python
def run_full_comparison(self, df_metrics: pd.DataFrame) -> Dict[str, Any]
```
Expects a metrics DataFrame containing columns:
`['date', 'day_num', 'plot_type', 'mean_ngrdi', 'mean_exg', 'mean_vari', 'mean_tgi', 'mean_exb', 'stress_pixel_ratio', 'salt_patch_ratio', 'moisture_zone_variance']`

Returns a dictionary structured as:
```json
{
  "t_test": { "t_statistic": ..., "p_value": ..., "significant_at_005": true },
  "anova": { "f_statistic": ..., "p_value": ..., "f_statistic_across_dates": ... },
  "effect_size": { "cohens_d_ngrdi": ..., "interpretation_ngrdi": ... },
  "temporal_regression": { "sspv_slope_per_day": ..., "inter_plot_slope_z_stat": ... },
  "rutag_hydrology_validation": {
    "capillary_stability_ratio": ...,
    "capillary_stability_interpretation": "SSPV maintains steady matric buffer (Gangotri low-oscillation percolation)",
    "sspv_perimeter_salt_ratio": ...,
    "desalination_protection_active": true,
    "estimated_water_conservation_pct": 38.5
  }
}
```

---

## 4. Example Usage

```python
import pandas as pd
from sspv_pipeline.statistics_engine import StatisticalComparisonEngine

engine = StatisticalComparisonEngine()
df = pd.read_csv("sspv_7date_metrics.csv")
results = engine.run_full_comparison(df)

print(f"t-test p-value: {results['t_test']['p_value']:.4e}")
print(f"Cohen's d (NGRDI): {results['effect_size']['cohens_d_ngrdi']:.2f}")
print(f"Slope Z-score: {results['temporal_regression']['inter_plot_slope_z_stat']:.2f}")
```
