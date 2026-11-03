"""
MODULE 2: 11-Channel Spectral Index & Feature Engineering.

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
        sum_gr2 = (G ** 2) + (R ** 2) + self.eps

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

        mgrvi = ((G ** 2) - (R ** 2)) / sum_gr2
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
