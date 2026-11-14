"""
MODULE 02: Preprocessing Pipeline (Radiometric, DOS, Geometric, ROI & Boundary Detection).

Implements Slide 11 Preprocessing Pipeline:
1. DN -> Reflectance Conversion (sRGB Inverse Gamma Linearization + Panel Calibration)
2. Atmospheric Correction via Dark Object Subtraction (DOS)
3. Geometric Correction (SIFT + RANSAC Affine Multi-Date Co-registration)
4. ROI Clipping & Median Filtering
"""

import os
import math
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
from PIL import Image, ExifTags
from scipy.ndimage import median_filter

from .config import (
    HAS_CV2, HAS_RASTERIO, GROUND_TRUTH_ROIS, REFERENCE_PLOT_GPS,
    GROUND_TRUTH_GEO_POLYS, DJI_ZENMUSE_P1_SPECS
)

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

    def load_image(
        self, image_path: str, target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Loads either a high-res DJI Zenmuse P1 .JPG or a Georeferenced .tif.
        If target_size is provided (e.g. (1024, 682)), uses hardware-accelerated
        downscaled JPEG decompression (IMREAD_REDUCED_COLOR_8 / PIL draft) for ~5x-10x
        faster loading without large RAM allocations.
        """
        ext = os.path.splitext(image_path)[1].lower()
        metadata = {"path": image_path, "crs": None, "transform": None}

        if ext in [".tif", ".tiff"] and HAS_RASTERIO:
            with rasterio.open(image_path) as src:
                img = src.read([1, 2, 3])  # (3, H, W)
                img = np.transpose(img, (1, 2, 0))  # (H, W, 3) in RGB
                metadata["crs"] = src.crs
                metadata["transform"] = src.transform
                if target_size is not None and HAS_CV2:
                    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        elif HAS_CV2:
            if target_size is not None and ext in [".jpg", ".jpeg"]:
                bgr = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_8)
                if bgr is not None:
                    bgr = cv2.resize(bgr, target_size, interpolation=cv2.INTER_AREA)
                else:
                    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            else:
                bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if bgr is None:
                raise FileNotFoundError(f"Could not read image at {image_path}")
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            with Image.open(image_path) as pil_img:
                if target_size is not None and ext in [".jpg", ".jpeg"]:
                    pil_img.draft("RGB", target_size)
                    pil_resized = pil_img.resize(target_size, Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR)
                    img = np.array(pil_resized.convert("RGB"))
                else:
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


def extract_exif_gps(img_path: str) -> Optional[Tuple[float, float, float]]:
    """
    Extracts (latitude, longitude, altitude) from DJI Zenmuse P1 EXIF metadata
    without decoding the full image pixels. Returns None if EXIF or GPS is absent.
    """
    try:
        with Image.open(img_path) as pil_img:
            exif = pil_img._getexif()
            if not exif:
                return None
            gps_data = None
            for tag_id, val in exif.items():
                if ExifTags.TAGS.get(tag_id) == "GPSInfo":
                    gps_data = val
                    break
            if not gps_data or 2 not in gps_data or 4 not in gps_data:
                return None
            # Latitude
            lat_deg, lat_min, lat_sec = gps_data[2]
            lat = float(lat_deg) + float(lat_min) / 60.0 + float(lat_sec) / 3600.0
            if gps_data.get(1) == "S":
                lat = -lat
            # Longitude
            lon_deg, lon_min, lon_sec = gps_data[4]
            lon = float(lon_deg) + float(lon_min) / 60.0 + float(lon_sec) / 3600.0
            if gps_data.get(3) == "W":
                lon = -lon
            # Altitude
            alt = float(gps_data.get(6, 0.0))
            return lat, lon, alt
    except Exception:
        return None


def compute_gps_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Computes ground distance in meters between two GPS coordinates."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    dlat = (lat2 - lat1) * 111139.0
    mean_lat_rad = np.radians((lat1 + lat2) / 2.0)
    dlon = (lon2 - lon1) * (111410.0 * np.cos(mean_lat_rad))
    return float(np.sqrt(dlat ** 2 + dlon ** 2))


def extract_dji_flight_telemetry(img_path: str) -> Optional[Dict[str, float]]:
    """
    Extracts high-precision RTK GNSS and gimbal orientation telemetry from DJI Zenmuse P1:
    - lat, lon, alt (AMSL)
    - yaw: GimbalYawDegree (clockwise from North)
    - pitch: GimbalPitchDegree (nadir is -90.0)
    - roll: GimbalRollDegree (+180.0 when flown in reverse pass)
    - rel_alt: RelativeAltitude (above takeoff / ground in meters)
    """
    gps = extract_exif_gps(img_path)
    if gps is None:
        return None
    lat, lon, alt = gps
    yaw, pitch, roll, rel_alt = 0.0, -90.0, 0.0, 40.0
    try:
        with open(img_path, "rb") as f:
            data = f.read(65536)
            def parse_val(tag: str) -> Optional[float]:
                idx = data.find(tag.encode())
                if idx != -1:
                    start = data.find(b"\"", idx) + 1
                    end = data.find(b"\"", start)
                    return float(data[start:end].decode("utf-8", errors="ignore"))
                return None
            y = parse_val("GimbalYawDegree")
            if y is not None: yaw = y
            p = parse_val("GimbalPitchDegree")
            if p is not None: pitch = p
            r = parse_val("GimbalRollDegree")
            if r is not None: roll = r
            ra = parse_val("RelativeAltitude")
            if ra is not None: rel_alt = ra
    except Exception:
        pass
    return {
        "lat": lat, "lon": lon, "alt": alt,
        "yaw": yaw, "pitch": pitch, "roll": roll,
        "rel_alt": rel_alt
    }


def project_geo_polygon_to_image(
    geo_poly: List[Tuple[float, float]],
    telem: Dict[str, float],
    out_w: int = 1024,
    out_h: int = 682,
    specs: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Transforms physical ground GPS coordinates into image pixel plane:
    Earth East-North -> Camera Sensor Frame (accounting for Yaw & Roll)
    -> Perspective sensor coordinates (using focal length & GSD).
    Returns integer polygon coordinates (N, 2).
    """
    if specs is None:
        specs = DJI_ZENMUSE_P1_SPECS
    sensor_w = specs["sensor_w_mm"]
    sensor_h = specs["sensor_h_mm"]
    focal = specs["focal_length_mm"]
    native_w = specs["native_width_px"]
    native_h = specs["native_height_px"]

    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(telem["lat"]))
    rad = math.radians(telem["yaw"])
    cos_yaw = math.cos(rad)
    sin_yaw = math.sin(rad)

    is_reversed = abs(telem.get("roll", 0.0) - 180.0) < 30.0
    rel_alt = max(5.0, telem.get("rel_alt", 40.0))
    gsd_x = (rel_alt * sensor_w) / (focal * native_w)
    gsd_y = (rel_alt * sensor_h) / (focal * native_h)

    px_coords = []
    for pt_lat, pt_lon in geo_poly:
        dn = (pt_lat - telem["lat"]) * m_per_deg_lat
        de = (pt_lon - telem["lon"]) * m_per_deg_lon
        x_cam = de * cos_yaw - dn * sin_yaw
        y_cam = de * sin_yaw + dn * cos_yaw
        if is_reversed:
            x_cam, y_cam = -x_cam, -y_cam
        dx_px = x_cam / gsd_x
        dy_px = -y_cam / gsd_y
        xn = (native_w * 0.5 + dx_px) / native_w
        yn = (native_h * 0.5 + dy_px) / native_h
        px_coords.append((int(round(xn * out_w)), int(round(yn * out_h))))
    return np.array(px_coords, dtype=np.int32)


def compute_polygon_visibility(poly_pts: np.ndarray, out_w: int = 1024, out_h: int = 682) -> float:
    """Computes the visible area proportion [0.0 - 1.0] of a polygon inside the image canvas."""
    if not HAS_CV2:
        return 1.0
    pts = poly_pts.reshape((-1, 1, 2))
    raw_area = float(cv2.contourArea(pts))
    if raw_area <= 0:
        return 0.0
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    vis_px = float(np.count_nonzero(mask))
    return float(vis_px / raw_area)


def detect_realtime_plot_boundaries(
    img: np.ndarray,
    img_path: Optional[str] = None,
    telemetry: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Dynamically identifies the 3 experimental boundaries per image in real-time:
    1. Study Area (Red)
    2. SSPV Treatment Zone (Blue) - tightly bounding the grid of pitcher mounds
    3. Control Plot (Green) - tightly bounding the flood irrigation check basins
    """
    H, W = img.shape[:2]

    # Mode 1: RTK GNSS & Gimbal Photogrammetric Projection (Exact per-image geometry)
    if img_path and os.path.exists(img_path) and telemetry is None:
        telemetry = extract_dji_flight_telemetry(img_path)

    if telemetry is not None:
        study_poly = project_geo_polygon_to_image(GROUND_TRUTH_GEO_POLYS["study_area"], telemetry, W, H)
        sspv_poly = project_geo_polygon_to_image(GROUND_TRUTH_GEO_POLYS["sspv_zone"], telemetry, W, H)
        ctrl_poly = project_geo_polygon_to_image(GROUND_TRUTH_GEO_POLYS["control_zone"], telemetry, W, H)

        vis_sspv = compute_polygon_visibility(sspv_poly, W, H)
        vis_ctrl = compute_polygon_visibility(ctrl_poly, W, H)
        vis_study = compute_polygon_visibility(study_poly, W, H)

        is_valid = (vis_sspv >= 0.15 and vis_ctrl >= 0.15)

        return {
            "study_area": study_poly,
            "sspv_zone": sspv_poly,
            "control_zone": ctrl_poly,
            "visibility_sspv": vis_sspv,
            "visibility_control": vis_ctrl,
            "visibility_study": vis_study,
            "is_valid_dual_presence": is_valid,
            "yaw_deg": telemetry.get("yaw", 0.0),
            "altitude_m": telemetry.get("rel_alt", 40.0),
            "method": "telemetry_projection"
        }

    # Mode 2: Computer Vision Mound & Grid Cluster Detection (For images without RTK telemetry)
    if HAS_CV2:
        if np.issubdtype(img.dtype, np.floating):
            img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img_u8 = img.astype(np.uint8)
        gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY) if img_u8.ndim == 3 else img_u8
        g1 = cv2.GaussianBlur(gray, (5, 5), 1.0)
        g2 = cv2.GaussianBlur(gray, (15, 15), 3.0)
        dog = cv2.subtract(g1, g2)
        _, thresh = cv2.threshold(dog, 12, 255, cv2.THRESH_BINARY)
        thresh_u8 = thresh.astype(np.uint8)
        contours, _ = cv2.findContours(thresh_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        mound_centers = []
        for c in contours:
            area = cv2.contourArea(c)
            if 8 < area < 400:
                perimeter = cv2.arcLength(c, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if circularity > 0.35:
                        M = cv2.moments(c)
                        if M["m00"] > 0:
                            mound_centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

        if len(mound_centers) >= 8:
            pts = np.array(mound_centers)
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect).astype(np.int32)
            box_w, box_h = rect[1]
            center = np.mean(box, axis=0)
            ctrl_box = box.copy()
            ctrl_box = (ctrl_box - center) * 0.4 + (center - np.array([box_w * 0.4, 0]))
            ctrl_box = ctrl_box.astype(np.int32)

            all_pts = np.vstack([box, ctrl_box])
            study_hull = cv2.convexHull(all_pts).reshape(-1, 2)

            return {
                "study_area": study_hull,
                "sspv_zone": box,
                "control_zone": ctrl_box,
                "visibility_sspv": 1.0,
                "visibility_control": 1.0,
                "visibility_study": 1.0,
                "is_valid_dual_presence": True,
                "yaw_deg": float(rect[2]),
                "altitude_m": 40.0,
                "method": "computer_vision_clustering"
            }

    # Mode 3: Calibrated Static ROI Fallback
    sa = GROUND_TRUTH_ROIS["study_area"]
    sz = GROUND_TRUTH_ROIS["sspv_zone"]
    cz = GROUND_TRUTH_ROIS["control_zone"]
    return {
        "study_area": np.array([[int(sa["xmin"]*W), int(sa["ymin"]*H)], [int(sa["xmax"]*W), int(sa["ymin"]*H)], [int(sa["xmax"]*W), int(sa["ymax"]*H)], [int(sa["xmin"]*W), int(sa["ymax"]*H)]], dtype=np.int32),
        "sspv_zone": np.array([[int(sz["xmin"]*W), int(sz["ymin"]*H)], [int(sz["xmax"]*W), int(sz["ymin"]*H)], [int(sz["xmax"]*W), int(sz["ymax"]*H)], [int(sz["xmin"]*W), int(sz["ymax"]*H)]], dtype=np.int32),
        "control_zone": np.array([[int(cz["xmin"]*W), int(cz["ymin"]*H)], [int(cz["xmax"]*W), int(cz["ymin"]*H)], [int(cz["xmax"]*W), int(cz["ymax"]*H)], [int(cz["xmin"]*W), int(cz["ymax"]*H)]], dtype=np.int32),
        "visibility_sspv": 1.0,
        "visibility_control": 1.0,
        "visibility_study": 1.0,
        "is_valid_dual_presence": True,
        "yaw_deg": 0.0,
        "altitude_m": 40.0,
        "method": "calibrated_fallback"
    }


def draw_ground_truth_boundaries(
    img: np.ndarray,
    rois: Optional[Dict[str, Any]] = None,
    img_path: Optional[str] = None,
    alpha: float = 0.18,
    show_study_area: bool = True,
    show_sspv: bool = True,
    show_control: bool = True
) -> np.ndarray:
    """
    Renders the 3 real-time dynamically detected experimental zones on an RGB image:
    1. Red: Total Experimental Study Area (Rotated Polygon)
    2. Blue: SSPV Treatment Zone (Pitcher Mounds Grid Polygon)
    3. Green: Control Plot (Partitioned Flood Basins Strip Polygon)
    """
    is_float = np.issubdtype(img.dtype, np.floating)
    if is_float:
        display_img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        display_img = img.copy()

    H, W = display_img.shape[:2]

    # Detect real-time dynamic per-image boundaries
    det = detect_realtime_plot_boundaries(display_img, img_path=img_path)

    if HAS_CV2:
        overlay = display_img.copy()

        poly_sa = det["study_area"].reshape((-1, 1, 2))
        poly_sspv = det["sspv_zone"].reshape((-1, 1, 2))
        poly_ctrl = det["control_zone"].reshape((-1, 1, 2))

        red_rgb = (255, 0, 0)
        blue_rgb = (30, 144, 255)
        green_rgb = (0, 230, 0)

        # 1. Translucent Fills
        if show_study_area:
            cv2.fillPoly(overlay, [poly_sa], red_rgb)
        if show_sspv:
            cv2.fillPoly(overlay, [poly_sspv], blue_rgb)
        if show_control:
            cv2.fillPoly(overlay, [poly_ctrl], green_rgb)

        cv2.addWeighted(overlay, alpha, display_img, 1.0 - alpha, 0, display_img)

        # 2. Crisp Solid Polygon Outlines
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.42, min(0.85, W / 1400.0))
        thick = 2 if scale > 0.55 else 1

        def _get_label_anchor(pts: np.ndarray) -> Tuple[int, int]:
            # Anchor near the top-left vertex
            pts_2d = pts.reshape((-1, 2))
            top_pt = pts_2d[np.argmin(pts_2d[:, 1] + pts_2d[:, 0] * 0.5)]
            x, y = int(top_pt[0]), int(top_pt[1])
            return max(10, min(W - 220, x + 4)), max(24, min(H - 10, y + 20))

        if show_study_area:
            cv2.polylines(display_img, [poly_sa], isClosed=True, color=red_rgb, thickness=3, lineType=cv2.LINE_AA)
            lx, ly = _get_label_anchor(poly_sa)
            cv2.putText(display_img, "STUDY AREA (SSPV + CONTROL)", (lx, ly), font, scale, red_rgb, thick, cv2.LINE_AA)

        if show_sspv:
            cv2.polylines(display_img, [poly_sspv], isClosed=True, color=blue_rgb, thickness=2, lineType=cv2.LINE_AA)
            lx, ly = _get_label_anchor(poly_sspv)
            cv2.putText(display_img, "SSPV FIELD (POROUS VESSELS)", (lx, ly), font, scale * 0.9, blue_rgb, thick, cv2.LINE_AA)

        if show_control:
            cv2.polylines(display_img, [poly_ctrl], isClosed=True, color=green_rgb, thickness=2, lineType=cv2.LINE_AA)
            lx, ly = _get_label_anchor(poly_ctrl)
            cv2.putText(display_img, "CONTROL (FLOOD BASINS)", (lx, ly), font, scale * 0.9, green_rgb, thick, cv2.LINE_AA)

        # If incomplete dual presence detected, display notice
        if not det.get("is_valid_dual_presence", True):
            cv2.putText(
                display_img,
                "NOTICE: Partial / Off-Center Frame (Treatment boundaries partially extend beyond sensor edge)",
                (20, H - 20), font, scale * 0.85, (0, 165, 255), thick, cv2.LINE_AA
            )
    else:
        pil_img = Image.fromarray(display_img)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(pil_img)
        if show_study_area:
            draw.polygon([tuple(p) for p in det["study_area"].reshape((-1, 2))], outline=(255, 0, 0), width=3)
        if show_sspv:
            draw.polygon([tuple(p) for p in det["sspv_zone"].reshape((-1, 2))], outline=(30, 144, 255), width=2)
        if show_control:
            draw.polygon([tuple(p) for p in det["control_zone"].reshape((-1, 2))], outline=(0, 230, 0), width=2)
        display_img = np.array(pil_img)

    if is_float:
        return display_img.astype(np.float32) / 255.0
    return display_img
