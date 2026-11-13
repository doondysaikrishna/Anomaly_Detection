"""
Industrial Visual Anomaly Detection - OpenCV Deep Defect Localization & Segmentation
Generates pixel-accurate heatmaps, vector polygon contours, convex hulls, morphometric analytics,
oriented bounding boxes, ROI patch crops, multi-colormap saliency overlays, and ground-truth pixel IoU validation.
"""

import base64
import math
from io import BytesIO
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import numpy as np
import cv2
from PIL import Image


COLORMAP_MAP = {
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "plasma": cv2.COLORMAP_PLASMA,
    "hot": cv2.COLORMAP_HOT,
}


@dataclass
class DefectMorphometrics:
    centroid_x: int
    centroid_y: int
    aspect_ratio: float
    circularity: float
    orientation_deg: float
    major_axis_px: float
    minor_axis_px: float
    defect_class: str  # "Crack / Scratch", "Pinhole / Void", "Surface Stain", "Structural Damage", "General Defect"
    peak_intensity: float
    mean_intensity: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BoundingBox:
    id: int
    x: int
    y: int
    width: int
    height: int
    area_px: int
    area_percent: float
    confidence: float
    polygon_points: List[List[int]]
    convex_hull: List[List[int]]
    morphometrics: DefectMorphometrics
    crop_base64: Optional[str] = None
    heatmap_crop_base64: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        if isinstance(self.morphometrics, DefectMorphometrics):
            res["morphometrics"] = self.morphometrics.to_dict()
        return res


@dataclass
class GroundTruthMetrics:
    has_ground_truth: bool
    pixel_iou: float
    dice_score: float
    pixel_precision: float
    pixel_recall: float
    specificity: float
    tp_pixels: int
    fp_pixels: int
    fn_pixels: int
    tn_pixels: int
    diff_map_base64: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DefectResult:
    is_anomalous: bool
    anomaly_score: float
    threshold: float
    severity: str  # "Normal", "Minor", "Moderate", "Critical"
    defect_area_percent: float
    bounding_boxes: List[BoundingBox]
    heatmap_overlay_base64: Optional[str] = None
    defect_mask_base64: Optional[str] = None
    annotated_image_base64: Optional[str] = None
    contour_mesh_base64: Optional[str] = None
    ground_truth_metrics: Optional[GroundTruthMetrics] = None
    model_name: str = "PatchCore"
    status: str = "NORMAL"  # "NORMAL" or "DEFECT"
    reconstructed_image_base64: Optional[str] = None
    reconstruction_error_base64: Optional[str] = None
    pipeline_steps: Optional[List[Dict[str, Any]]] = None
    comparison_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["bounding_boxes"] = [
            b.to_dict() if isinstance(b, BoundingBox) else b for b in self.bounding_boxes
        ]
        if self.ground_truth_metrics and isinstance(self.ground_truth_metrics, GroundTruthMetrics):
            res["ground_truth_metrics"] = self.ground_truth_metrics.to_dict()
        return res


class DefectLocalizer:
    """
    Advanced OpenCV-based defect localization, contour extraction, morphometrics analysis,
    and ground-truth verification engine.
    """
    def __init__(
        self,
        colormap: Union[int, str] = cv2.COLORMAP_JET,
        heatmap_alpha: float = 0.50,
        min_defect_area_px: int = 12,
    ):
        if isinstance(colormap, str):
            self.colormap = COLORMAP_MAP.get(colormap.lower(), cv2.COLORMAP_JET)
        else:
            self.colormap = colormap
        self.heatmap_alpha = heatmap_alpha
        self.min_defect_area_px = min_defect_area_px

    def set_colormap(self, colormap_name_or_id: Union[int, str]) -> None:
        """Dynamically update active colormap."""
        if isinstance(colormap_name_or_id, str):
            self.colormap = COLORMAP_MAP.get(colormap_name_or_id.lower(), cv2.COLORMAP_JET)
        else:
            self.colormap = colormap_name_or_id

    def generate_heatmap_overlay(
        self,
        original_rgb: np.ndarray,
        anomaly_map: np.ndarray,
        colormap_override: Optional[Union[str, int]] = None,
        alpha: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates colored heatmap and blends it with original image.
        Returns:
            heatmap_colored (np.ndarray): Pure RGB heatmap
            overlay (np.ndarray): Blended original + heatmap RGB image
        """
        if original_rgb.dtype != np.uint8:
            if original_rgb.max() <= 1.0:
                original_rgb = (original_rgb * 255).astype(np.uint8)
            else:
                original_rgb = original_rgb.astype(np.uint8)
                
        H, W = original_rgb.shape[:2]
        if anomaly_map.shape[:2] != (H, W):
            anomaly_map = cv2.resize(anomaly_map, (W, H), interpolation=cv2.INTER_LINEAR)
            
        norm_map = (np.clip(anomaly_map, 0.0, 1.0) * 255).astype(np.uint8)
        
        cmap = self.colormap
        if colormap_override is not None:
            if isinstance(colormap_override, str):
                cmap = COLORMAP_MAP.get(colormap_override.lower(), self.colormap)
            else:
                cmap = colormap_override
                
        heatmap_bgr = cv2.applyColorMap(norm_map, cmap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        
        a = alpha if alpha is not None else self.heatmap_alpha
        overlay = cv2.addWeighted(heatmap_rgb, a, original_rgb, 1.0 - a, 0)
        return heatmap_rgb, overlay

    def segment_defect_mask(
        self, anomaly_map: np.ndarray, threshold: float = 0.50, use_otsu: bool = False
    ) -> np.ndarray:
        """
        Generates binary defect mask via thresholding and morphological filtering.
        Returns uint8 binary mask (0 or 255).
        """
        norm_map = (np.clip(anomaly_map, 0.0, 1.0) * 255).astype(np.uint8)
        
        if use_otsu:
            _, binary_mask = cv2.threshold(norm_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            int_th = int(np.clip(threshold, 0.0, 1.0) * 255)
            _, binary_mask = cv2.threshold(norm_map, int_th, 255, cv2.THRESH_BINARY)
            
        # Morphological noise suppression
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        return binary_mask

    def extract_defect_analytics(
        self,
        binary_mask: np.ndarray,
        anomaly_map: np.ndarray,
        original_rgb: np.ndarray,
        heatmap_rgb: np.ndarray,
    ) -> List[BoundingBox]:
        """
        Extracts deep morphometric properties, polygon vertices, convex hulls,
        and high-resolution cropped defect ROI patches.
        """
        H, W = binary_mask.shape[:2]
        total_area = float(H * W)
        
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []
        
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < self.min_defect_area_px:
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            perimeter = cv2.arcLength(cnt, True)
            area_percent = (area / total_area) * 100.0
            
            # Polygon approximation (simplifies vertices while retaining geometric fidelity)
            epsilon = 0.012 * perimeter
            approx_poly = cv2.approxPolyDP(cnt, epsilon, True)
            poly_points = [[int(pt[0][0]), int(pt[0][1])] for pt in approx_poly]
            
            # Convex Hull
            hull = cv2.convexHull(cnt)
            hull_points = [[int(pt[0][0]), int(pt[0][1])] for pt in hull]
            
            # Moments for Centroid
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
                
            # Circularity / Compactness: 4 * pi * area / perimeter^2
            circularity = (4.0 * math.pi * area) / (perimeter * perimeter + 1e-6)
            circularity = float(np.clip(circularity, 0.0, 1.0))
            
            # Oriented Bounding Box (minAreaRect)
            if len(cnt) >= 5:
                rect = cv2.minAreaRect(cnt)
                (rx, ry), (rw, rh), angle = rect
                major_axis = max(rw, rh)
                minor_axis = min(rw, rh)
                aspect_ratio = float(major_axis / (minor_axis + 1e-6))
                orientation_deg = float(round(angle, 1))
            else:
                major_axis = float(max(w, h))
                minor_axis = float(min(w, h))
                aspect_ratio = float(major_axis / (minor_axis + 1e-6))
                orientation_deg = 0.0
                
            # Regional Anomaly Scores
            region_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(region_mask, [cnt], -1, 255, -1)
            defect_pixels = anomaly_map[region_mask > 0]
            peak_intensity = float(defect_pixels.max()) if len(defect_pixels) > 0 else float(anomaly_map[y:y+h, x:x+w].max())
            mean_intensity = float(defect_pixels.mean()) if len(defect_pixels) > 0 else float(anomaly_map[y:y+h, x:x+w].mean())
            
            # Morphological Classification
            if aspect_ratio >= 2.75 and circularity < 0.55:
                defect_class = "Crack / Linear Scratch"
            elif circularity >= 0.65 and area_percent < 2.0:
                defect_class = "Pinhole / Micro-Void"
            elif area_percent >= 8.0:
                defect_class = "Structural Damage / Void"
            elif mean_intensity > 0.70:
                defect_class = "Severe Surface Defect"
            else:
                defect_class = "Surface Contamination / Stain"
                
            morphometrics = DefectMorphometrics(
                centroid_x=int(cx),
                centroid_y=int(cy),
                aspect_ratio=float(round(aspect_ratio, 2)),
                circularity=float(round(circularity, 3)),
                orientation_deg=orientation_deg,
                major_axis_px=float(round(major_axis, 1)),
                minor_axis_px=float(round(minor_axis, 1)),
                defect_class=defect_class,
                peak_intensity=float(round(peak_intensity, 4)),
                mean_intensity=float(round(mean_intensity, 4)),
            )
            
            # Generate ROI Crops (padded)
            pad = 12
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(W, x + w + pad)
            y2 = min(H, y + h + pad)
            
            crop_orig = original_rgb[y1:y2, x1:x2]
            crop_heat = heatmap_rgb[y1:y2, x1:x2]
            
            crop_b64 = self.encode_base64(crop_orig, quality=85) if crop_orig.size > 0 else None
            crop_heat_b64 = self.encode_base64(crop_heat, quality=85) if crop_heat.size > 0 else None
            
            bounding_boxes.append(
                BoundingBox(
                    id=idx + 1,
                    x=int(x),
                    y=int(y),
                    width=int(w),
                    height=int(h),
                    area_px=int(area),
                    area_percent=float(round(area_percent, 2)),
                    confidence=float(round(peak_intensity, 4)),
                    polygon_points=poly_points,
                    convex_hull=hull_points,
                    morphometrics=morphometrics,
                    crop_base64=crop_b64,
                    heatmap_crop_base64=crop_heat_b64,
                )
            )
            
        bounding_boxes.sort(key=lambda b: b.area_px, reverse=True)
        return bounding_boxes

    def draw_annotated_image(
        self,
        original_rgb: np.ndarray,
        bounding_boxes: List[BoundingBox],
        severity: str,
    ) -> np.ndarray:
        """
        Draws bounding box annotations and defect badges directly onto the image.
        """
        annotated = original_rgb.copy()
        
        color_map = {
            "Normal": (34, 197, 94),     # Green
            "Minor": (234, 179, 8),      # Yellow
            "Moderate": (249, 115, 22),  # Orange
            "Critical": (239, 68, 68),   # Red
        }
        box_color = color_map.get(severity, (239, 68, 68))
        
        for i, box in enumerate(bounding_boxes):
            x, y, w, h = box.x, box.y, box.width, box.height
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
            
            # Label
            label = f"#{box.id} {box.morphometrics.defect_class.split('/')[0].strip()} ({box.confidence:.2f})"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(annotated, (x, max(0, y - label_h - 6)), (x + label_w + 6, max(0, y)), box_color, -1)
            cv2.putText(
                annotated,
                label,
                (x + 3, max(label_h + 2, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            
        return annotated

    def draw_contour_mesh(
        self,
        original_rgb: np.ndarray,
        bounding_boxes: List[BoundingBox],
    ) -> np.ndarray:
        """
        Draws high-precision vector polygon contours, convex hulls, and centroid pins
        over the original image.
        """
        mesh_img = original_rgb.copy()
        
        for box in bounding_boxes:
            # 1. Draw Convex Hull in dashed/cyan outline
            if box.convex_hull and len(box.convex_hull) >= 3:
                hull_np = np.array([box.convex_hull], dtype=np.int32)
                cv2.polylines(mesh_img, hull_np, True, (6, 182, 212), 1, cv2.LINE_AA)
                
            # 2. Draw Exact Polygon Contour in vivid neon
            if box.polygon_points and len(box.polygon_points) >= 3:
                poly_np = np.array([box.polygon_points], dtype=np.int32)
                # Fill semi-transparent polygon
                overlay = mesh_img.copy()
                cv2.fillPoly(overlay, poly_np, (239, 68, 68))
                cv2.addWeighted(overlay, 0.35, mesh_img, 0.65, 0, mesh_img)
                cv2.polylines(mesh_img, poly_np, True, (244, 63, 94), 2, cv2.LINE_AA)
                
            # 3. Draw Centroid Crosshair
            cx = box.morphometrics.centroid_x
            cy = box.morphometrics.centroid_y
            cv2.circle(mesh_img, (cx, cy), 3, (255, 255, 255), -1)
            cv2.circle(mesh_img, (cx, cy), 5, (239, 68, 68), 1)
            
        return mesh_img

    def evaluate_ground_truth(
        self,
        binary_mask: np.ndarray,
        gt_mask_input: Union[str, np.ndarray, Image.Image],
        original_rgb: np.ndarray,
    ) -> GroundTruthMetrics:
        """
        Computes pixel-level localization metrics (IoU, Dice/F1, Precision, Recall)
        and generates an RGB Pixel Difference Map (Green=TP, Yellow=FP, Red=FN).
        """
        if isinstance(gt_mask_input, str):
            gt_img = cv2.imread(gt_mask_input, cv2.IMREAD_GRAYSCALE)
        elif isinstance(gt_mask_input, Image.Image):
            gt_img = np.array(gt_mask_input.convert("L"))
        elif isinstance(gt_mask_input, np.ndarray):
            gt_img = cv2.cvtColor(gt_mask_input, cv2.COLOR_RGB2GRAY) if gt_mask_input.ndim == 3 else gt_mask_input
        else:
            raise ValueError("Unsupported ground-truth mask format")
            
        H, W = binary_mask.shape[:2]
        if gt_img.shape[:2] != (H, W):
            gt_img = cv2.resize(gt_img, (W, H), interpolation=cv2.INTER_NEAREST)
            
        gt_binary = (gt_img > 0).astype(np.uint8)
        pred_binary = (binary_mask > 0).astype(np.uint8)
        
        tp = int(np.sum((pred_binary == 1) & (gt_binary == 1)))
        fp = int(np.sum((pred_binary == 1) & (gt_binary == 0)))
        fn = int(np.sum((pred_binary == 0) & (gt_binary == 1)))
        tn = int(np.sum((pred_binary == 0) & (gt_binary == 0)))
        
        # Pixel-IoU (Jaccard)
        intersection = tp
        union = tp + fp + fn
        iou = float(intersection / (union + 1e-8))
        
        # Dice Score (Localization F1)
        dice = float((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8))
        
        precision = float(tp / (tp + fp + 1e-8)) if (tp + fp) > 0 else 1.0
        recall = float(tp / (tp + fn + 1e-8)) if (tp + fn) > 0 else 1.0
        specificity = float(tn / (tn + fp + 1e-8)) if (tn + fp) > 0 else 1.0
        
        # Generate RGB Diff Map: Green (TP), Yellow (FP), Red (FN)
        diff_rgb = original_rgb.copy()
        
        # Blend overlay
        overlay = diff_rgb.copy()
        overlay[(pred_binary == 1) & (gt_binary == 1)] = [34, 197, 94]    # TP = Green
        overlay[(pred_binary == 1) & (gt_binary == 0)] = [234, 179, 8]    # FP = Yellow
        overlay[(pred_binary == 0) & (gt_binary == 1)] = [239, 68, 68]    # FN = Red (Missed defect)
        
        cv2.addWeighted(overlay, 0.55, diff_rgb, 0.45, 0, diff_rgb)
        diff_b64 = self.encode_base64(diff_rgb, quality=85)
        
        return GroundTruthMetrics(
            has_ground_truth=True,
            pixel_iou=float(round(iou, 4)),
            dice_score=float(round(dice, 4)),
            pixel_precision=float(round(precision, 4)),
            pixel_recall=float(round(recall, 4)),
            specificity=float(round(specificity, 4)),
            tp_pixels=tp,
            fp_pixels=fp,
            fn_pixels=fn,
            tn_pixels=tn,
            diff_map_base64=diff_b64,
        )

    def determine_severity(self, anomaly_score: float, threshold: float) -> str:
        """Categorizes anomaly score into industrial quality severity levels."""
        if anomaly_score < threshold:
            return "Normal"
        delta = anomaly_score - threshold
        if delta < 0.15:
            return "Minor"
        elif delta < 0.35:
            return "Moderate"
        else:
            return "Critical"

    def process(
        self,
        original_rgb: np.ndarray,
        anomaly_score: float,
        anomaly_map: np.ndarray,
        threshold: float = 0.50,
        colormap: Optional[Union[str, int]] = None,
        ground_truth_mask: Optional[Union[str, np.ndarray, Image.Image]] = None,
        model_name: str = "PatchCore",
        reconstructed_image_base64: Optional[str] = None,
        reconstruction_error_base64: Optional[str] = None,
        pipeline_steps: Optional[List[Dict[str, Any]]] = None,
        comparison_data: Optional[Dict[str, Any]] = None,
    ) -> DefectResult:
        """
        Complete localization and segmentation workflow.
        Follows the Anomalib orchestration pipeline:
        Preprocessing -> Model Prediction -> Thresholding -> Heatmap & Binary Mask -> Contours / ROI -> Final Output.
        """
        if colormap is not None:
            self.set_colormap(colormap)
            
        is_anomalous = bool(anomaly_score >= threshold)
        severity = self.determine_severity(anomaly_score, threshold)
        status = "DEFECT" if is_anomalous else "NORMAL"
        
        # 1. Heatmap overlay (Step 6)
        heatmap_rgb, overlay_rgb = self.generate_heatmap_overlay(original_rgb, anomaly_map)
        
        # 2. Binary segmentation mask (Step 6)
        binary_mask = self.segment_defect_mask(anomaly_map, threshold=threshold)
        
        # 3. Defect analytics & bounding boxes / Contours / ROI (Step 7)
        bounding_boxes = (
            self.extract_defect_analytics(binary_mask, anomaly_map, original_rgb, heatmap_rgb)
            if is_anomalous
            else []
        )
        
        # 4. Annotated image & Vector Contour Mesh
        annotated_rgb = self.draw_annotated_image(original_rgb, bounding_boxes, severity)
        contour_mesh_rgb = self.draw_contour_mesh(original_rgb, bounding_boxes)
        
        # 5. Defect area
        total_defect_px = int(np.sum(binary_mask > 0))
        total_px = binary_mask.shape[0] * binary_mask.shape[1]
        defect_area_percent = float(round((total_defect_px / total_px) * 100.0, 2)) if is_anomalous else 0.0
        
        # 6. Ground-Truth verification if available
        gt_metrics = None
        if ground_truth_mask is not None:
            try:
                gt_metrics = self.evaluate_ground_truth(binary_mask, ground_truth_mask, original_rgb)
            except Exception as e:
                print(f"[!] Ground truth evaluation error: {e}")
        
        return DefectResult(
            is_anomalous=is_anomalous,
            anomaly_score=float(round(anomaly_score, 4)),
            threshold=float(round(threshold, 4)),
            severity=severity,
            defect_area_percent=defect_area_percent,
            bounding_boxes=bounding_boxes,
            heatmap_overlay_base64=self.encode_base64(overlay_rgb),
            defect_mask_base64=self.encode_base64(cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)),
            annotated_image_base64=self.encode_base64(annotated_rgb),
            contour_mesh_base64=self.encode_base64(contour_mesh_rgb),
            ground_truth_metrics=gt_metrics,
            model_name=model_name,
            status=status,
            reconstructed_image_base64=reconstructed_image_base64,
            reconstruction_error_base64=reconstruction_error_base64,
            pipeline_steps=pipeline_steps,
            comparison_data=comparison_data,
        )

    @staticmethod
    def encode_base64(image_rgb: np.ndarray, format: str = "JPEG", quality: int = 88) -> str:
        """Encodes an RGB numpy image into a base64 Data URL string."""
        pil_img = Image.fromarray(image_rgb)
        buffered = BytesIO()
        pil_img.save(buffered, format=format, quality=quality)
        b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{b64_str}"
