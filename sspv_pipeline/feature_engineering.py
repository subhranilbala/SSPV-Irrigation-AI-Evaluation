"""
MODULE 03: 11-Channel Spectral Index & Feature Engineering.

Implements Slide 12 & 13:
Computes 8 RGB Vegetation Indices pixel-wise and stacks them with R, G, B
into an 11-channel tensor of shape (11, H, W).
"""

from typing import Dict, Tuple

import numpy as np


class VegetationIndexExtractor:
    """
    Exact Formulas from Slide 12:
    1. NGRDI = (G - R) / (G + R)
    2. ExG   = (2*G - B - R) / (B + G + R)
    3. ExR   = (1.4*R - G) / (B + G + R)
    4. VARI  = (G - R) / (G + R - B)
    5. MGRVI = (G^2 - R^2) / (G^2 + R^2)
    6. NPCI  = (B - R) / (B + R)
    7. TGI   = G - 0.39*R - 0.61*B
    8. ExB   = 1.4*B - G
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def compute_indices(self, rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Input: rgb array of shape (H, W, 3) with float32 reflectance values in [0, 1].
        Returns dictionary of 8 vegetation indices.
        """
        R = rgb[:, :, 0].astype(np.float32)
        G = rgb[:, :, 1].astype(np.float32)
        B = rgb[:, :, 2].astype(np.float32)

        sum_rgb = R + G + B + self.eps
        sum_gr = G + R + self.eps
        sum_br = B + R + self.eps

        # Optimized: Single-pass direct floating multiplication instead of redundant pow() calls
        G2 = G * G
        R2 = R * R
        sum_gr2 = G2 + R2 + self.eps

        ngrdi = (G - R) / sum_gr
        exg = (2.0 * G - B - R) / sum_rgb
        exr = (1.4 * R - G) / sum_rgb

        # VARI denominator can approach zero when G + R ~= B; stabilize denominator
        vari_denom = G + R - B
        vari_denom = np.where(
            np.abs(vari_denom) < 1e-3,
            np.sign(vari_denom + 1e-7) * 1e-3,
            vari_denom
        )
        vari = np.clip((G - R) / vari_denom, -2.0, 2.0)

        mgrvi = (G2 - R2) / sum_gr2
        npci = (B - R) / sum_br
        tgi = G - 0.39 * R - 0.61 * B
        exb = 1.4 * B - G

        return {
            "NGRDI": ngrdi,
            "ExG": exg,
            "ExR": exr,
            "VARI": vari,
            "MGRVI": mgrvi,
            "NPCI": npci,
            "TGI": tgi,
            "ExB": exb
        }

    def build_11ch_tensor(self, rgb: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Returns:
        - tensor_11ch: numpy array of shape (11, H, W)
        - indices_dict: dictionary of the 8 individual index maps (H, W)
        """
        indices = self.compute_indices(rgb)
        channels = [
            rgb[:, :, 0],
            rgb[:, :, 1],
            rgb[:, :, 2],
            indices["NGRDI"],
            indices["ExG"],
            indices["ExR"],
            indices["VARI"],
            indices["MGRVI"],
            indices["NPCI"],
            indices["TGI"],
            indices["ExB"],
        ]
        tensor_11ch = np.stack(channels, axis=0).astype(np.float32)
        return tensor_11ch, indices

    def compute_canopy_thermal_buffering_proxy(
        self, indices: Dict[str, np.ndarray], seg_mask: np.ndarray
    ) -> float:
        """
        RuTAG / Indigenous Hydrology Principle: 'Cooling down the structure'.
        Measures the thermal and chlorotic buffering provided by the SSPV subsurface
        evaporative cooling in the rhizosphere.
        Uses the inverse of NPCI (Normalized Pigment Chlorophyll Index) weighted by VARI.
        Higher value indicates greater canopy cooling & chlorophyll stability under heat.
        """
        crop_pixels = (seg_mask == 1) | (seg_mask == 2)
        if not np.any(crop_pixels):
            return 0.0

        # In heat-stressed crops, NPCI rises (carotenoid/chlorophyll ratio increases)
        # while VARI drops. The buffering metric measures resistance to this divergence.
        npci_crop = indices["NPCI"][crop_pixels]
        vari_crop = indices["VARI"][crop_pixels]

        # Normalized cooling score in [0, 1]
        buffering = np.mean(np.clip((vari_crop - npci_crop + 1.0) / 2.0, 0.0, 1.0))
        return float(buffering)

    def compute_perimeter_salt_leaching_index(
        self, seg_mask: np.ndarray, patch_center: Tuple[int, int] = None
    ) -> Dict[str, float]:
        """
        Parashurama / Mannarasala Principle: Saline Soil Reclamation.
        Quantifies the radial displacement of salts away from the rhizosphere to the
        mound periphery, confirming that the porous vessel is pushing salts outward.
        """
        H, W = seg_mask.shape
        salt_pixels = (seg_mask == 3)
        salt_ratio = float(np.mean(salt_pixels))

        if not np.any(salt_pixels):
            return {
                "salt_ratio": 0.0,
                "mean_radial_distance_px": 0.0,
                "normalized_dispersion": 0.0,
                "root_zone_protected": True
            }

        cy, cx = patch_center if patch_center is not None else (H // 2, W // 2)
        ys, xs = np.where(salt_pixels)
        distances = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)

        max_possible_r = np.sqrt((H / 2) ** 2 + (W / 2) ** 2)
        mean_r = float(np.mean(distances))
        norm_dispersion = float(mean_r / (max_possible_r + self.eps))

        # If salt is displaced toward the outer perimeter (norm_dispersion > 0.40),
        # the central root zone is actively protected from salinity toxicity.
        root_zone_protected = bool(norm_dispersion > 0.40 and salt_ratio > 0.01)

        return {
            "salt_ratio": salt_ratio,
            "mean_radial_distance_px": mean_r,
            "normalized_dispersion": norm_dispersion,
            "root_zone_protected": root_zone_protected
        }

