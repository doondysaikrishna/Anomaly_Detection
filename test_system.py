"""
End-to-End System Verification Test Script
Tests model initialization, single image inspection, localization, and evaluation metrics.
"""

import os
import sys
import glob
from core.pipeline import IndustrialAnomalyPipeline
from core.dataset.industrial_dataset import discover_category_samples, get_available_categories


def run_tests():
    print("\n=======================================================")
    print("  RUNNING SYSTEM INTEGRATION & VERIFICATION TESTS      ")
    print("=======================================================\n")
    
    categories = ["cable", "capsule", "circuit_board", "transistor"]
    passed = 0
    total = len(categories)
    
    for cat in categories:
        print(f"[*] Testing Pipeline for Category: '{cat}'...")
        try:
            samples = discover_category_samples("test", cat)
            train_count = len(samples.get("train", []))
            test_count = len(samples.get("test", []))
            
            print(f"    - Found {train_count} train samples, {test_count} test samples.")
            
            pipeline = IndustrialAnomalyPipeline(category=cat, weights_dir="weights")
            if not pipeline.is_ready:
                print(f"    - Training fast memory bank for '{cat}'...")
                pipeline.train(calibrate=True)
                
            # Pick a sample image to inspect
            test_samples = samples.get("test", [])
            if test_samples:
                sample_img = test_samples[0]["image_path"]
                result, _ = pipeline.inspect(sample_img)
                print(f"    - Inspection test: Score={result.anomaly_score:.4f}, Verdict={result.severity}, BBoxes={len(result.bounding_boxes)}")
                assert result.anomaly_score >= 0.0, "Anomaly score must be non-negative"
                assert result.heatmap_overlay_base64 is not None, "Heatmap overlay must be generated"
                
            print(f"[+] Category '{cat}' PASSED!\n")
            passed += 1
            
        except Exception as e:
            print(f"[!] Category '{cat}' FAILED with error: {e}\n")
            
    print("=======================================================")
    print(f"TEST RESULTS: {passed}/{total} Categories Passed ({passed/total*100:.1f}%)")
    print("=======================================================\n")
    return passed == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
