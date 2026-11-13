"""
Industrial Visual Anomaly Detection - Evaluation & Benchmarking CLI
Computes Image-level and Pixel-level AUROC, Precision, Recall, F1, and Confusion Matrices.
"""

import argparse
import os
import json
from core.pipeline import IndustrialAnomalyPipeline
from core.dataset.industrial_dataset import get_available_categories


def main():
    parser = argparse.ArgumentParser(description="Evaluate Industrial Visual Anomaly Detection Models")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["cable", "capsule", "circuit_board", "transistor"],
        help="Categories to evaluate",
    )
    parser.add_argument("--dataset_root", type=str, default="test", help="Root dataset directory")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Directory containing trained weights")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="Path to save results JSON")
    
    args = parser.parse_args()
    
    print(f"\n=======================================================")
    print(f"  INDUSTRIAL VISUAL ANOMALY DETECTION - EVALUATION     ")
    print(f"=======================================================\n")
    
    results = {}
    
    for category in args.categories:
        print(f"[*] Evaluating category: '{category}'...")
        try:
            pipeline = IndustrialAnomalyPipeline(
                category=category,
                dataset_root=args.dataset_root,
                weights_dir=args.weights_dir,
            )
            
            if not pipeline.is_ready:
                print(f"[!] No weights found for '{category}'. Training on the fly...")
                pipeline.train(calibrate=True)
                
            metrics = pipeline.calibrate_and_evaluate()
            results[category] = metrics
            
        except Exception as e:
            print(f"[!] Error evaluating category '{category}': {e}")
            results[category] = {"error": str(e)}
            
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n==========================================================================================================")
    print(f"                                       BENCHMARK EVALUATION RESULTS                                       ")
    print(f"==========================================================================================================")
    print(f"{'Category':<15} | {'Samples':<7} | {'Image AUROC':<12} | {'Pixel AUROC':<12} | {'Image F1':<10} | {'Pixel F1':<10} | {'Pixel IoU':<10} | {'Recall':<8}")
    print("-" * 106)
    for cat, data in results.items():
        if "image_auroc" in data:
            pix_auroc = f"{data['pixel_auroc']:.4f}" if data.get("pixel_auroc") is not None else "N/A"
            pix_f1 = f"{data['pixel_f1']:.4f}" if data.get("pixel_f1") is not None else (f"{data['pixel_dice']:.4f}" if data.get("pixel_dice") is not None else "N/A")
            pix_iou = f"{data['pixel_iou']:.4f}" if data.get("pixel_iou") is not None else "N/A"
            print(f"{cat:<15} | {data.get('total_samples', 0):<7} | {data.get('image_auroc', 0.0):<12.4f} | {pix_auroc:<12} | {data.get('f1_score', 0.0):<10.4f} | {pix_f1:<10} | {pix_iou:<10} | {data.get('recall', 0.0):<8.4f}")
        else:
            print(f"{cat:<15} | {'ERROR':<7} | {'ERROR':<12} | {'ERROR':<12} | {'ERROR':<10} | {'ERROR':<10} | {'ERROR':<10} | {'ERROR':<8}")
            
    print(f"\n[+] Results exported to: {os.path.abspath(args.output_file)}\n")


if __name__ == "__main__":
    main()
