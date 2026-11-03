"""
MODULE 3: Weakly-Supervised Biophysical Pseudo-Label Generator.

Addresses the lack of manual pixel-level ground-truth masks and IoT sensor labels
by generating physically grounded weak labels from spectral signatures.
"""

from typing import Dict, Tuple

import numpy as np


class BiophysicalPseudoLabeler:
    """
    Generates:
    - Segmentation Mask (0: Soil, 1: Healthy Crop, 2: Stressed Crop, 3: White Salt Patch)
    - Multi-Task Targets (Stress Level, Soil Moisture Zone, Yield Potential)
    """

    @staticmethod
    def generate_segmentation_mask(
        rgb: np.ndarray, indices: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Generates a 4-class segmentation mask (H, W):
        0 = Bare Soil / Background
        1 = Healthy Crop Canopy (High ExG & positive NGRDI)
        2 = Stressed / Chlorotic Crop Zone (Moderate ExG but low NGRDI / high ExR)
        3 = White Salt Patch around SSPV Mound (High luminance, low saturation, elevated ExB)
        """
        H, W, _ = rgb.shape
        mask = np.zeros((H, W), dtype=np.int64)

        ngrdi = indices["NGRDI"]
        exg = indices["ExG"]
        exr = indices["ExR"]
        tgi = indices["TGI"]

        # Luminance and color saturation for white salt crust detection (Slide 7)
        luminance = np.mean(rgb, axis=2)
        rgb_std = np.std(rgb, axis=2)

        # Class 3: White Salt Patch (bright, achromatic crust on soil mounds)
        salt_mask = (luminance > np.percentile(luminance, 92)) & (rgb_std < 0.06) & (exg < 0.05)
        mask[salt_mask] = 3

        # Vegetation mask via adaptive ExG + TGI threshold
        exg_thresh = max(0.06, float(np.percentile(exg, 65)))
        veg_candidates = (exg > exg_thresh) | (ngrdi > 0.04)

        # Separate Healthy Canopy (Class 1) vs Stressed Crop (Class 2)
        healthy_veg = veg_candidates & (ngrdi >= 0.05) & (tgi >= 0.01) & (~salt_mask)
        stressed_veg = veg_candidates & (~healthy_veg) & (~salt_mask) & (exr > 0.0)

        mask[stressed_veg] = 2
        mask[healthy_veg] = 1

        return mask

    @staticmethod
    def generate_multitask_labels(
        rgb: np.ndarray,
        indices: Dict[str, np.ndarray],
        seg_mask: np.ndarray
    ) -> Tuple[int, int, int]:
        """
        Derives patch-level (stress_label, moisture_label, yield_label) in {0, 1, 2}.
        """
        crop_pixels = (seg_mask == 1) | (seg_mask == 2)
        crop_fraction = float(np.mean(crop_pixels))

        if np.any(crop_pixels):
            mean_ngrdi = float(np.mean(indices["NGRDI"][crop_pixels]))
            mean_exr = float(np.mean(indices["ExR"][crop_pixels]))
            mean_tgi = float(np.mean(indices["TGI"][crop_pixels]))
        else:
            mean_ngrdi = float(np.mean(indices["NGRDI"]))
            mean_exr = float(np.mean(indices["ExR"]))
            mean_tgi = float(np.mean(indices["TGI"]))

        # 1. Crop Stress: 0=Healthy, 1=Mild, 2=Severe
        if mean_ngrdi > 0.08 and mean_exr < 0.12:
            stress_label = 0
        elif mean_ngrdi > 0.01:
            stress_label = 1
        else:
            stress_label = 2

        # 2. Soil Moisture Zone: 0=Dry, 1=Optimal, 2=Wet
        # Darker soil surrounding SSPV & lower ExR / elevated ExB indicates higher moisture
        soil_pixels = seg_mask == 0
        soil_lum = float(np.mean(rgb[:, :, 0][soil_pixels])) if np.any(soil_pixels) else 0.45
        mean_exb = float(np.mean(indices["ExB"]))

        if mean_exb > 0.05 or soil_lum < 0.28:
            moisture_label = 2  # Wet
        elif soil_lum < 0.48 and stress_label <= 1:
            moisture_label = 1  # Optimal (SSPV capillary balance)
        else:
            moisture_label = 0  # Dry

        # 3. Yield Potential: 0=Low, 1=Medium, 2=High
        yield_score = 0.5 * crop_fraction + 2.5 * max(0.0, mean_ngrdi) + 2.0 * max(0.0, mean_tgi)
        if yield_score > 0.35:
            yield_label = 2  # High
        elif yield_score > 0.15:
            yield_label = 1  # Medium
        else:
            yield_label = 0  # Low

        return stress_label, moisture_label, yield_label
