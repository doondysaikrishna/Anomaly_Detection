"""
Industrial Visual Anomaly Detection - Model Training & Calibration CLI
Trains PatchCore Multi-scale CNN Anomaly Detection models on industrial objects:
Cable, Capsule, Circuit Board / PCB, Transistor, and custom datasets.
"""

import argparse
import os
import json
import time
from core.pipeline import IndustrialAnomalyPipeline
from core.dataset.industrial_dataset import get_available_categories


def main():
    parser = argparse.ArgumentParser(description="Train Industrial Visual Anomaly Detection Models")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["cable", "capsule", "circuit_board", "transistor"],
        help="List of industrial categories to train",
    )
    parser.add_argument("--dataset_root", type=str, default="test", help="Root directory containing dataset")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Directory to save model weights")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Feature extractor backbone (resnet18, resnet50)")
    parser.add_argument("--coreset_ratio", type=float, default=0.10, help="Memory bank coreset subsampling ratio")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for feature extraction")
    
    args = parser.parse_args()
    
    os.makedirs(args.weights_dir, exist_ok=True)
    
    available = get_available_categories(args.dataset_root)
    print(f"\n=======================================================")
    print(f"  INDUSTRIAL VISUAL ANOMALY DETECTION - MODEL TRAINING  ")
    print(f"=======================================================")
    print(f"Available Dataset Categories: {available}", flush=True)
    print(f"Target Categories to Train:   {args.categories}", flush=True)
    print(f"CNN Backbone:                 {args.backbone}", flush=True)
    print(f"Coreset Ratio:                {args.coreset_ratio}", flush=True)
    print(f"Weights Output Directory:     {args.weights_dir}", flush=True)
    print(f"=======================================================\n", flush=True)
    
    summary = {}
    total_start = time.time()
    
    for category in args.categories:
        try:
            pipeline = IndustrialAnomalyPipeline(
                category=category,
                dataset_root=args.dataset_root,
                weights_dir=args.weights_dir,
                backbone=args.backbone,
                coreset_ratio=args.coreset_ratio,
            )
            
            res = pipeline.train(batch_size=args.batch_size, calibrate=True)
            summary[category] = res
            
        except Exception as e:
            print(f"[!] Error training category '{category}': {e}")
            summary[category] = {"status": "failed", "error": str(e)}
            
    total_duration = time.time() - total_start
    
    # Save training report summary
    summary_path = os.path.join(args.weights_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n=======================================================")
    print(f"              TRAINING SUMMARY REPORT                  ")
    print(f"=======================================================")
    print(f"{'Category':<16} | {'Samples':<8} | {'AUROC':<8} | {'F1-Score':<8} | {'Threshold':<10} | {'Status'}")
    print("-" * 68)
    for cat, data in summary.items():
        if "image_auroc" in data:
            print(f"{cat:<16} | {data.get('total_samples', 'N/A'):<8} | {data.get('image_auroc', 0.0):<8.4f} | {data.get('f1_score', 0.0):<8.4f} | {data.get('threshold', 0.0):<10.4f} | [PASSED]")
        else:
            print(f"{cat:<16} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<10} | [FAILED]")
            
    print(f"\n[+] All models saved to: {os.path.abspath(args.weights_dir)}", flush=True)
    print(f"[+] Summary saved to: {summary_path}", flush=True)
    print(f"[+] Total training time: {total_duration:.2f} seconds\n", flush=True)


if __name__ == "__main__":
    main()
