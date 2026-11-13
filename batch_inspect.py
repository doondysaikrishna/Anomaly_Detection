"""
Industrial Visual Anomaly Detection - Batch Inspection CLI
Inspects multiple images or folders in batch and generates quality control CSV reports.
"""

import argparse
import os
import glob
import csv
import time
from core.pipeline import IndustrialAnomalyPipeline


def main():
    parser = argparse.ArgumentParser(description="Batch Industrial Visual Anomaly Inspection")
    parser.add_argument("--folder", type=str, required=True, help="Folder containing images to inspect")
    parser.add_argument("--category", type=str, required=True, help="Object category (cable, capsule, circuit_board, transistor)")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Model weights directory")
    parser.add_argument("--output_csv", type=str, default="batch_inspection_results.csv", help="CSV report path")
    
    args = parser.parse_args()
    
    img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_paths = sorted([
        p for p in glob.glob(os.path.join(args.folder, "**", "*.*"), recursive=True)
        if p.lower().endswith(img_exts)
    ])
    
    if not image_paths:
        print(f"[!] No image files found in '{args.folder}'")
        return
        
    print(f"\n=======================================================")
    print(f"  INDUSTRIAL BATCH QUALITY INSPECTION                  ")
    print(f"=======================================================")
    print(f"Target Directory: {args.folder}")
    print(f"Category:         {args.category}")
    print(f"Found Images:     {len(image_paths)}")
    print(f"=======================================================\n")
    
    pipeline = IndustrialAnomalyPipeline(category=args.category, weights_dir=args.weights_dir)
    if not pipeline.is_ready:
        print(f"[*] Training model for '{args.category}'...")
        pipeline.train(calibrate=True)
        
    results = []
    passed = 0
    failed = 0
    start_time = time.time()
    
    for idx, img_path in enumerate(image_paths):
        res = pipeline.inspect(img_path)
        status = "FAIL" if res.is_anomalous else "PASS"
        if res.is_anomalous:
            failed += 1
        else:
            passed += 1
            
        results.append({
            "filename": os.path.basename(img_path),
            "filepath": img_path,
            "category": args.category,
            "status": status,
            "anomaly_score": res.anomaly_score,
            "threshold": res.threshold,
            "severity": res.severity,
            "defect_area_percent": res.defect_area_percent,
            "defect_count": len(res.bounding_boxes),
        })
        
        if (idx + 1) % 20 == 0 or (idx + 1) == len(image_paths):
            print(f"  Processed {idx + 1}/{len(image_paths)} images... (Pass: {passed}, Fail: {failed})")
            
    total_time = time.time() - start_time
    
    # Write CSV
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "filepath", "category", "status", "anomaly_score", "threshold", "severity", "defect_area_percent", "defect_count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\n=======================================================")
    print(f"               BATCH INSPECTION SUMMARY                ")
    print(f"=======================================================")
    print(f"Total Inspected:  {len(image_paths)}")
    print(f"Passed (Normal):  {passed} ({passed/len(image_paths)*100:.1f}%)")
    print(f"Failed (Defects): {failed} ({failed/len(image_paths)*100:.1f}%)")
    print(f"Throughput:       {len(image_paths)/total_time:.2f} images/sec")
    print(f"\n[+] CSV Quality Report saved to: {os.path.abspath(args.output_csv)}\n")


if __name__ == "__main__":
    main()
