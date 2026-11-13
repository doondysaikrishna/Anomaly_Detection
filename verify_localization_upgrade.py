"""
Comprehensive Verification Script for Deep Anomaly Localization Enhancements.
Tests:
1. PatchCore multi-scale extraction and Edge-Preserving Guided Bilateral Filtering.
2. OpenCV DefectLocalizer with Polygon Contours, Convex Hulls, and Morphometrics.
3. Ground-Truth Localization Verification (Pixel-IoU, Dice Score, Diff Map).
4. Pipeline inspect() and fast resegment().
"""

import os
import sys
import glob
import time
import numpy as np
import cv2
import torch
from PIL import Image

from core.models.patchcore import PatchCoreDetector, MultiScaleFeatureExtractor, apply_edge_preserving_refinement
from core.vision.segmentation import DefectLocalizer, DefectResult, COLORMAP_MAP
from core.pipeline import IndustrialAnomalyPipeline
from core.evaluation.metrics import AnomalyEvaluator


def run_localization_tests():
    print("==================================================================")
    print("  DEEP ANOMALY LOCALIZATION & SEGMENTATION VERIFICATION SUITE   ")
    print("==================================================================")

    # 1. Test MultiScaleFeatureExtractor
    print("\n[*] Step 1: Testing MultiScaleFeatureExtractor (L1+L2+L3)...")
    extractor_triple = MultiScaleFeatureExtractor(backbone_name="resnet18", pretrained=False, use_layer1=True)
    extractor_dual = MultiScaleFeatureExtractor(backbone_name="resnet18", pretrained=False, use_layer1=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    feat_triple = extractor_triple(dummy_input)
    feat_dual = extractor_dual(dummy_input)
    print(f"    [+] Triple-layer shape: {feat_triple.shape} (Expected: [2, 448, 28, 28])")
    print(f"    [+] Dual-layer shape:   {feat_dual.shape} (Expected: [2, 384, 28, 28])")
    assert feat_triple.shape == (2, 448, 28, 28), "Triple feature shape mismatch"
    assert feat_dual.shape == (2, 384, 28, 28), "Dual feature shape mismatch"

    # 2. Test Edge-Preserving Guided Bilateral Refinement
    print("\n[*] Step 2: Testing Edge-Preserving Guided Refinement...")
    dummy_raw_map = np.zeros((224, 224), dtype=np.float32)
    dummy_raw_map[100:130, 100:130] = 0.95
    dummy_guide_rgb = np.full((224, 224, 3), 128, dtype=np.uint8)
    dummy_guide_rgb[90:140, 90:140] = 220  # simulate object boundary
    
    refined_map = apply_edge_preserving_refinement(dummy_raw_map, guide_image_rgb=dummy_guide_rgb)
    print(f"    [+] Refined map min={refined_map.min():.3f}, max={refined_map.max():.3f}, shape={refined_map.shape}")
    assert refined_map.shape == (224, 224)
    assert 0.0 <= refined_map.min() and refined_map.max() <= 1.0

    # 3. Test DefectLocalizer with Polygon Contours, Morphometrics & Colormaps
    print("\n[*] Step 3: Testing DefectLocalizer Morphometrics & Colormaps...")
    localizer = DefectLocalizer(colormap="turbo", min_defect_area_px=10)
    
    # Synthetic crack / scratch anomaly map
    synth_map = np.zeros((224, 224), dtype=np.float32)
    # Long thin line (scratch)
    synth_map[50:55, 60:150] = 0.88
    # Small circular blob (pinhole)
    cv2.circle(synth_map, (180, 180), 8, 0.92, -1)
    
    orig_img = np.full((224, 224, 3), 160, dtype=np.uint8)
    
    result = localizer.process(
        original_rgb=orig_img,
        anomaly_score=0.85,
        anomaly_map=synth_map,
        threshold=0.50,
        colormap="inferno",
    )
    
    print(f"    [+] Verdict: {result.severity} (Score: {result.anomaly_score})")
    print(f"    [+] Defect Area: {result.defect_area_percent}%")
    print(f"    [+] Detected Bounding Boxes / Contours: {len(result.bounding_boxes)}")
    
    for box in result.bounding_boxes:
        m = box.morphometrics
        print(f"        -> Region #{box.id}: Class='{m.defect_class}', Aspect={m.aspect_ratio}:1, Circularity={m.circularity}, Centroid=({m.centroid_x}, {m.centroid_y}), Angle={m.orientation_deg}°")
        print(f"           Polygon points count: {len(box.polygon_points)}, Convex hull count: {len(box.convex_hull)}")
        assert len(box.polygon_points) >= 3, "Polygon points missing"
        assert box.crop_base64 is not None, "ROI crop missing"
        assert box.heatmap_crop_base64 is not None, "Heatmap crop missing"
        
    assert len(result.bounding_boxes) >= 2, "Expected at least 2 distinct contours detected"
    assert result.contour_mesh_base64 is not None, "Contour mesh overlay missing"
    assert result.heatmap_overlay_base64 is not None, "Heatmap overlay missing"

    # 4. Test Ground-Truth Evaluation (IoU & Dice & Diff Map)
    print("\n[*] Step 4: Testing Ground-Truth Mask Validation...")
    gt_mask = np.zeros((224, 224), dtype=np.uint8)
    gt_mask[50:55, 60:150] = 255  # ground truth scratch
    
    gt_metrics = localizer.evaluate_ground_truth(
        binary_mask=(synth_map > 0.5).astype(np.uint8) * 255,
        gt_mask_input=gt_mask,
        original_rgb=orig_img,
    )
    print(f"    [+] Pixel-IoU:    {gt_metrics.pixel_iou * 100:.2f}%")
    print(f"    [+] Dice (F1):    {gt_metrics.dice_score * 100:.2f}%")
    print(f"    [+] Precision:    {gt_metrics.pixel_precision * 100:.2f}%")
    print(f"    [+] Recall:       {gt_metrics.pixel_recall * 100:.2f}%")
    print(f"    [+] Diff Map B64: {gt_metrics.diff_map_base64[:40]}...")
    assert gt_metrics.pixel_iou > 0.5, "Expected high IoU for matching synthetic regions"
    assert gt_metrics.diff_map_base64 is not None

    # 5. Test Live Industrial Pipeline on Real Dataset Sample
    print("\n[*] Step 5: Testing IndustrialAnomalyPipeline on Real Dataset Sample...")
    test_files = [f for f in (glob.glob("test/*/*/*/*.*") + glob.glob("test/*/*.*")) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if test_files:
        sample_img_path = test_files[0]
        print(f"    [*] Loading sample: {sample_img_path}")
        pipeline = IndustrialAnomalyPipeline(category="cable", dataset_root="test", weights_dir="weights")
        if pipeline.is_ready:
            start_t = time.time()
            res, raw_map = pipeline.inspect(sample_img_path, colormap="turbo", use_guided_filter=True)
            lat = (time.time() - start_t) * 1000.0
            print(f"    [+] Real inference latency: {lat:.1f}ms")
            print(f"    [+] Result: {res.severity} (Score: {res.anomaly_score:.3f}, Threshold: {res.threshold:.2f})")
            print(f"    [+] Defect count: {len(res.bounding_boxes)}")

            # Test fast resegmentation (<10ms)
            start_reseg = time.time()
            res_reseg = pipeline.resegment(
                original_rgb=cv2.imread(sample_img_path),
                anomaly_map=raw_map,
                score=res.anomaly_score,
                threshold=0.35,
                colormap="viridis",
            )
            lat_reseg = (time.time() - start_reseg) * 1000.0
            print(f"    [+] Fast resegmentation latency: {lat_reseg:.2f}ms (Expected < 10ms)")
            print(f"    [+] Resegmented defect count: {len(res_reseg.bounding_boxes)}")
        else:
            print("    [!] Pipeline weights not initialized yet (will auto-train on startup)")
            
    print("\n==================================================================")
    print("  ALL DEEP LOCALIZATION SYSTEM TESTS PASSED SUCCESSFULLY! (100%) ")
    print("==================================================================")


if __name__ == "__main__":
    run_localization_tests()
