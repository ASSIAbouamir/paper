"""
Test & Validation Script for Unified Dataset Integration Pipeline.
Validates loading and preprocessing for all registered datasets:
  - ml1m (MovieLens-1M)
  - amazon_beauty (Amazon Beauty)
  - amazon_sports (Amazon Sports & Outdoors)
  - amazon (Amazon Reviews)
  - steam (Steam Games)
  - yelp (Yelp Academic)
  - kuairec (KuaiRec Micro-Video)
"""

import os
import sys
import time
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset_registry import DATASET_CONFIGS, get_dataset, list_datasets


REQUIRED_KEYS = [
    "num_users", "num_items", "train_user_seq", "val_user_seq",
    "test_user_seq", "cold_start_users", "item_metadata",
    "item_frequencies", "user_sequences"
]


def test_all_datasets():
    print("=" * 80)
    print(" TESTING UNIFIED DATASET INTEGRATION PIPELINE")
    print(" Registered datasets:", list(DATASET_CONFIGS.keys()))
    print("=" * 80)

    list_datasets()

    results = {}

    for key, cfg in DATASET_CONFIGS.items():
        print(f"\n--- Testing Dataset: '{key}' ({cfg.display_name}) ---")
        t0 = time.time()
        try:
            data = get_dataset(key)
            elapsed = time.time() - t0

            # Check required keys
            missing_keys = [k for k in REQUIRED_KEYS if k not in data]
            if missing_keys:
                print(f"  FAILED: Missing keys {missing_keys}")
                results[key] = {"status": "FAILED", "error": f"Missing keys: {missing_keys}"}
                continue

            num_users = data["num_users"]
            num_items = data["num_items"]
            num_train = len(data["train_user_seq"])
            num_val = len(data["val_user_seq"])
            num_test = len(data["test_user_seq"])
            num_cold = len(data["cold_start_users"])

            # Basic assertions
            assert num_users > 0, "num_users must be > 0"
            assert num_items > 0, "num_items must be > 0"
            assert num_test > 0, "num_test_seq must be > 0"

            print(f"  [SUCCESS] Loaded in {elapsed:.2f}s")
            print(f"  Stats: {num_users:,} Users | {num_items:,} Items | {num_test:,} Test Instances | {num_cold:,} Cold-Start Users")

            results[key] = {
                "status": "OK",
                "display_name": cfg.display_name,
                "num_users": num_users,
                "num_items": num_items,
                "num_test": num_test,
                "num_cold": num_cold,
                "time_sec": round(elapsed, 2)
            }

        except Exception as e:
            print(f"  FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results[key] = {"status": "FAILED", "error": str(e)}

    print("\n" + "=" * 90)
    print(f"{'DATASET INTEGRATION PIPELINE VERIFICATION SUMMARY':^90}")
    print("=" * 90)
    header = f"| {'Key':<15} | {'Display Name':<25} | {'Status':<7} | {'Users':>7} | {'Items':>7} | {'Cold':>6} | {'Time (s)':>8} |"
    print(header)
    print("|" + "-"*17 + "|" + "-"*27 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*8 + "|" + "-"*10 + "|")

    all_passed = True
    for key, r in results.items():
        if r["status"] == "OK":
            row = f"| {key:<15} | {r['display_name']:<25} | {r['status']:<7} | {r['num_users']:>7,} | {r['num_items']:>7,} | {r['num_cold']:>6,} | {r['time_sec']:>8.2f} |"
        else:
            row = f"| {key:<15} | {DATASET_CONFIGS[key].display_name:<25} | {'FAILED':<7} | {'—':>7} | {'—':>7} | {'—':>6} | {'—':>8} |"
            all_passed = False
        print(row)

    print("=" * 90)

    if all_passed:
        print("\nALL DATASETS PASSED INTEGRATION PIPELINE VERIFICATION!")
    else:
        print("\nSOME DATASETS FAILED INTEGRATION VERIFICATION. CHECK ERRORS ABOVE.")

    return all_passed


if __name__ == "__main__":
    test_all_datasets()
