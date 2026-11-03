"""
MODULE 1: Preprocessing Pipeline (Radiometric, DOS, Geometric, ROI).

Implements Slide 11 Preprocessing Pipeline:
1. DN -> Reflectance Conversion (sRGB Inverse Gamma Linearization + Panel Calibration)
2. Atmospheric Correction via Dark Object Subtraction (DOS)
3. Geometric Correction (SIFT + RANSAC Affine Multi-Date Co-registration)
4. ROI Clipping & Median Filtering
"""

import os
from typing import Dict, Tuple, Optional, Any

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

from .config import HAS_CV2, HAS_RASTERIO

if HAS_CV2:
    import cv2
if HAS_RASTERIO:
    import rasterio


class UAVPreprocessor:
    """
    Full 4-step preprocessing pipeline for DJI Zenmuse P1 (8192x5460 JPGs)
    and GeoTIFF drone imagery.
    """

    def __init__(self, dos_percentile: float = 1.0, median_kernel: int = 3):
        self.dos_percentile = dos_percentile
        self.median_kernel = median_kernel
        self.Eps = 1e-6

    def load_image(self, image_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Loads either a high-res DJI Zenmuse P1 .JPG or a Georeferenced .tif."""
        ext = os.path.splitext(image_path)[1].lower()
        metadata = {"path": image_path, "crs": None, "transform": None}

        if ext in [".tif", ".tiff"] and HAS_RASTERIO:
            with rasterio.open(image_path) as src:
                img = src.read([1, 2, 3])  # (3, H, W)
                img = np.transpose(img, (1, 2, 0))  # (H, W, 3) in RGB
                metadata["crs"] = src.crs
                metadata["transform"] = src.transform
        elif HAS_CV2:
            bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"Could not read image at {image_path}")
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            with Image.open(image_path) as pil_img:
                img = np.array(pil_img.convert("RGB"))

        return img, metadata

    def dn_to_reflectance(
        self, img_dn: np.ndarray, panel_factor: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Step 1: Converts 8-bit/16-bit Digital Numbers (DN) to normalized linear surface
        reflectance [0, 1]. Removes camera sRGB gamma curve for DJI Zenmuse P1 JPEGs.
        """
        if img_dn.dtype == np.uint16:
            norm = img_dn.astype(np.float32) / 65535.0
        else:
            norm = img_dn.astype(np.float32) / 255.0

        # Inverse sRGB gamma correction to recover linear physical radiance proxy
        linear_rgb = np.where(
            norm <= 0.04045,
            norm / 12.92,
            np.power((norm + 0.055) / 1.055, 2.4)
        ).astype(np.float32)

        if panel_factor is not None:
            linear_rgb = linear_rgb * panel_factor.reshape(1, 1, 3)

        return np.clip(linear_rgb, 0.0, 1.0)

    def dark_object_subtraction(self, reflectance: np.ndarray) -> np.ndarray:
        """
        Step 2: Dark Object Subtraction (DOS) for low-altitude UAV atmospheric path
        radiance removal. Subtracts the 1st percentile dark object value per band.
        """
        corrected = np.zeros_like(reflectance, dtype=np.float32)
        for c in range(3):
            band = reflectance[:, :, c]
            valid_pixels = band[band > 0.001]
            dark_val = np.percentile(valid_pixels, self.dos_percentile) if valid_pixels.size > 0 else 0.0
            band_corr = np.clip(band - dark_val, 0.0, None)
            max_val = np.percentile(band_corr, 99.9) if band_corr.size > 0 else 1.0
            corrected[:, :, c] = np.clip(band_corr / (max_val + self.Eps), 0.0, 1.0)
        return corrected

    def geometric_alignment(
        self, target_img: np.ndarray, reference_img: np.ndarray
    ) -> np.ndarray:
        """
        Step 3: Multi-date geometric co-registration using SIFT feature matching
        and RANSAC Affine transformation.
        """
        if target_img.shape[:2] != reference_img.shape[:2]:
            if HAS_CV2:
                target_img = cv2.resize(
                    target_img, (reference_img.shape[1], reference_img.shape[0])
                )
            else:
                pil_t = Image.fromarray((np.clip(target_img, 0, 1) * 255).astype(np.uint8))
                pil_t = pil_t.resize((reference_img.shape[1], reference_img.shape[0]), Image.BILINEAR)
                target_img = np.asarray(pil_t, dtype=np.float32) / 255.0

        if not HAS_CV2:
            return target_img

        target_gray = (np.mean(target_img, axis=2) * 255).astype(np.uint8)
        ref_gray = (np.mean(reference_img, axis=2) * 255).astype(np.uint8)

        sift = cv2.SIFT_create(nfeatures=2500)
        kp1, des1 = sift.detectAndCompute(target_gray, None)
        kp2, des2 = sift.detectAndCompute(ref_gray, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return target_img

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) >= 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
            if M is not None:
                h, w = reference_img.shape[:2]
                aligned = cv2.warpAffine(
                    target_img, M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101
                )
                return aligned
        return target_img

    def clip_and_filter(
        self, img: np.ndarray, roi_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Step 4: Isolates plot boundary ROI (y1, y2, x1, x2) and applies spatial
        median filtering to suppress high-frequency soil/sensor noise.
        """
        if roi_bbox is not None:
            y1, y2, x1, x2 = roi_bbox
            img = img[y1:y2, x1:x2, :]

        if self.median_kernel > 1:
            filtered = np.zeros_like(img)
            for c in range(3):
                filtered[:, :, c] = median_filter(img[:, :, c], size=self.median_kernel)
            return filtered
        return img

    def process(
        self,
        img_dn: np.ndarray,
        reference_img: Optional[np.ndarray] = None,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """Executes Steps 1 to 4 end-to-end."""
        refl = self.dn_to_reflectance(img_dn)
        dos_refl = self.dark_object_subtraction(refl)
        if reference_img is not None:
            dos_refl = self.geometric_alignment(dos_refl, reference_img)
        clean_refl = self.clip_and_filter(dos_refl, roi_bbox=roi_bbox)
        return clean_refl
