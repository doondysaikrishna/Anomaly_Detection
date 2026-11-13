"""
Industrial Visual Anomaly Detection - Single Image Inspection CLI
Inspects an image, localizes defects, extracts bounding boxes, and generates heatmap artifacts.
"""

import argparse
import os
import json
import cv2
from PIL import Image
from core.pipeline import IndustrialAnomalyPipeline


def main():
    parser = argparse.ArgumentParser(description="Inspect an Industrial Image for Visual Anomalies")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--category", type=str, required=True, help="Category name (e.g. cable, capsule, circuit_board, transistor)")
    parser.add_argument("--dataset_root", type=str, default="test", help="Root dataset directory")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Directory containing model weights")
    parser.add_argument("--threshold", type=float, default=None, help="Custom threshold override")
    parser.add_argument("--output_dir", type=str, default="inspection_results", help="Directory to save visual inspection outputs")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n=======================================================")
    print(f"  INDUSTRIAL VISUAL INSPECTION - SINGLE SAMPLE         ")
    print(f"=======================================================")
    print(f"Input Image: {args.image}")
    print(f"Category:    {args.category}")
    print(f"=======================================================\n")
    
    pipeline = IndustrialAnomalyPipeline(
        category=args.category,
        dataset_root=args.dataset_root,
        weights_dir=args.weights_dir,
    )
    
    if not pipeline.is_ready:
        print(f"[*] Pretrained weights not found for '{args.category}'. Fitting model...")
        pipeline.train(calibrate=True)
        
    result = pipeline.inspect(args.image, threshold_override=args.threshold)
    
    # Save visual outputs
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    
    # Decode and save annotated image
    annotated_path = os.path.join(args.output_dir, f"{base_name}_annotated.jpg")
    heatmap_path = os.path.join(args.output_dir, f"{base_name}_heatmap.jpg")
    mask_path = os.path.join(args.output_dir, f"{base_name}_mask.png")
    json_path = os.path.join(args.output_dir, f"{base_name}_report.json")
    
    # Write JSON report
    report_dict = result.to_dict()
    report_dict["image_path"] = os.path.abspath(args.image)
    report_dict["category"] = args.category
    
    # Remove base64 from JSON report file to keep it clean
    clean_report = {k: v for k, v in report_dict.items() if not k.endswith("_base64")}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_report, f, indent=2)
        
    status_str = "[FAIL] ANOMALY DETECTED" if result.is_anomalous else "[PASS] NORMAL"
    
    print(f"Status:             {status_str}", flush=True)
    print(f"Anomaly Score:      {result.anomaly_score:.4f} (Threshold: {result.threshold:.4f})", flush=True)
    print(f"Severity:           {result.severity}", flush=True)
    print(f"Defect Area:        {result.defect_area_percent:.2f}%", flush=True)
    print(f"Defect Count:       {len(result.bounding_boxes)}", flush=True)
    
    if result.bounding_boxes:
        print("\nDetected Defect Regions:", flush=True)
        for i, box in enumerate(result.bounding_boxes):
            print(f"  - Defect #{i+1}: BBox (x={box.x}, y={box.y}, w={box.width}, h={box.height}) | Area: {box.area_px}px ({box.area_percent}%) | Confidence: {box.confidence:.4f}", flush=True)
            
    print(f"\n[+] Diagnostic Report saved to: {os.path.abspath(json_path)}\n", flush=True)


if __name__ == "__main__":
    main()
