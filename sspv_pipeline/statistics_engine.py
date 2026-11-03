"""
MODULE 6: Statistical Comparison Engine (t-Test, ANOVA, Cohen's d, Regression).

Implements Slide 14 Statistical Comparison Engine:
1. Two-sample Welch's t-test (SSPV vs. Conventional Control mean NGRDI difference)
2. One-way ANOVA across multiple plots AND across multiple survey dates
3. Cohen's d effect size (NGRDI, Stress Proportion, and Soil Moisture Zone Variance)
4. Time-series linear regression per plot with inter-plot slope comparison Z-test
"""

import math
from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy import stats


class StatisticalComparisonEngine:
    """Full statistical comparison pipeline for SSPV vs. Control plots."""

    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Computes pooled Cohen's d effect size between SSPV and Control samples."""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 0.0
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_se = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2) + 1e-9)
        return float((np.mean(group1) - np.mean(group2)) / pooled_se)

    @staticmethod
    def interpret_cohens_d(d: float) -> str:
        abs_d = abs(d)
        if abs_d >= 0.8:
            return "Large Effect (High Practical Significance)"
        elif abs_d >= 0.5:
            return "Medium Effect"
        elif abs_d >= 0.2:
            return "Small Effect"
        return "Negligible Effect"

    def run_full_comparison(self, df_metrics: pd.DataFrame) -> Dict[str, Any]:
        """
        Expects a DataFrame with columns:
        ['date', 'day_num', 'plot_type', 'mean_ngrdi', 'mean_exg', 'mean_vari',
         'mean_tgi', 'mean_exb', 'stress_pixel_ratio', 'salt_patch_ratio',
         'moisture_zone_variance']
        where `plot_type` is in ['SSPV', 'Control'].
        """
        sspv = df_metrics[df_metrics["plot_type"] == "SSPV"]
        ctrl = df_metrics[df_metrics["plot_type"] == "Control"]

        # 1. Two-Sample t-test on Mean NGRDI
        t_stat, p_val_ttest = stats.ttest_ind(
            sspv["mean_ngrdi"].values,
            ctrl["mean_ngrdi"].values,
            equal_var=False
        )

        # 2. Cohen's d Effect Size on NGRDI, Stress Proportion, and Moisture Zone Variance (Slide 14)
        d_ngrdi = self.cohens_d(sspv["mean_ngrdi"].values, ctrl["mean_ngrdi"].values)
        d_stress = self.cohens_d(
            ctrl["stress_pixel_ratio"].values, sspv["stress_pixel_ratio"].values
        )
        d_moisture_var = (
            self.cohens_d(
                ctrl["moisture_zone_variance"].values,
                sspv["moisture_zone_variance"].values
            )
            if "moisture_zone_variance" in df_metrics.columns
            else 0.0
        )

        # 3a. One-way ANOVA across plot types on Stress Pixel Proportion
        plot_groups = [
            group["stress_pixel_ratio"].values
            for _, group in df_metrics.groupby("plot_type")
            if len(group) > 1
        ]
        if len(plot_groups) >= 2:
            f_stat_plot, p_val_anova_plot = stats.f_oneway(*plot_groups)
        else:
            f_stat_plot, p_val_anova_plot = 0.0, 1.0

        # 3b. One-way ANOVA across multiple survey dates on Stress Pixel Proportion (Slide 14)
        date_groups = [
            group["stress_pixel_ratio"].values
            for _, group in df_metrics.groupby("date")
            if len(group) > 1
        ]
        if len(date_groups) >= 2:
            f_stat_dates, p_val_anova_dates = stats.f_oneway(*date_groups)
        else:
            f_stat_dates, p_val_anova_dates = 0.0, 1.0

        # 4. Time-Series Linear Regression on NGRDI Growth Trend + Inter-Plot Slope Comparison (Slide 14 & 16)
        if len(np.unique(sspv["day_num"].values)) >= 2 and len(np.unique(ctrl["day_num"].values)) >= 2:
            sspv_reg = stats.linregress(sspv["day_num"].values, sspv["mean_ngrdi"].values)
            ctrl_reg = stats.linregress(ctrl["day_num"].values, ctrl["mean_ngrdi"].values)
            slope_diff = float(sspv_reg.slope - ctrl_reg.slope)
            se_diff = math.sqrt((sspv_reg.stderr or 0.0) ** 2 + (ctrl_reg.stderr or 0.0) ** 2 + 1e-12)
            slope_z_stat = float(slope_diff / se_diff)
            slope_diff_p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(slope_z_stat))))
        else:
            sspv_reg = stats.linregress([0, 1], [0, 0])
            ctrl_reg = stats.linregress([0, 1], [0, 0])
            slope_diff, slope_z_stat, slope_diff_p_val = 0.0, 0.0, 1.0

        # 5. Temporal Variance Analysis (Testing Slide 16 Hypothesis)
        sspv_exg_temporal_std = float(np.std(sspv.groupby("date")["mean_exg"].mean()))
        ctrl_exg_temporal_std = float(np.std(ctrl.groupby("date")["mean_exg"].mean()))

        return {
            "t_test": {
                "t_statistic": float(t_stat),
                "p_value": float(p_val_ttest),
                "significant_at_005": bool(p_val_ttest < 0.05),
                "sspv_mean_ngrdi": float(sspv["mean_ngrdi"].mean()),
                "control_mean_ngrdi": float(ctrl["mean_ngrdi"].mean()),
            },
            "anova": {
                "f_statistic": float(f_stat_plot),
                "p_value": float(p_val_anova_plot),
                "significant_at_005": bool(p_val_anova_plot < 0.05),
                "f_statistic_across_dates": float(f_stat_dates),
                "p_value_across_dates": float(p_val_anova_dates),
            },
            "effect_size": {
                "cohens_d_ngrdi": float(d_ngrdi),
                "interpretation_ngrdi": self.interpret_cohens_d(d_ngrdi),
                "cohens_d_stress_reduction": float(d_stress),
                "cohens_d_moisture_zone_variance": float(d_moisture_var),
                "interpretation_moisture_variance": self.interpret_cohens_d(d_moisture_var),
            },
            "temporal_regression": {
                "sspv_slope_per_day": float(sspv_reg.slope),
                "sspv_r_value": float(sspv_reg.rvalue),
                "sspv_p_value": float(sspv_reg.pvalue),
                "control_slope_per_day": float(ctrl_reg.slope),
                "control_r_value": float(ctrl_reg.rvalue),
                "control_p_value": float(ctrl_reg.pvalue),
                "inter_plot_slope_difference": slope_diff,
                "inter_plot_slope_z_stat": slope_z_stat,
                "inter_plot_slope_p_value": slope_diff_p_val,
                "sspv_exg_temporal_std": sspv_exg_temporal_std,
                "control_exg_temporal_std": ctrl_exg_temporal_std,
            }
        }
