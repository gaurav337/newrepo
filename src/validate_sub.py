import pandas as pd
import numpy as np

def validate_submission_integrity():
    print("loading target test splits and generation frames...")
    test_df = pd.read_csv("data/test.csv")
    submission_df = pd.read_csv("submissions/ensemble_submission.csv")
    
    # check 1: row count symmetry bounds
    assert len(submission_df) == len(test_df), f"length mismatch error: submission has {len(submission_df)} rows, test has {len(test_df)} rows."
    print(f"row alignment count verified: {len(submission_df)} records match.")
    
    # check 2: operational schema header checks
    expected_headers = ["Index", "demand"]
    assert list(submission_df.columns) == expected_headers, f"malformed schema error: found headers {list(submission_df.columns)}"
    print("schema structural columns check passed: target outputs match ['Index', 'demand'].")
    
    # check 3: tracking index alignment sequence matching
    assert (submission_df["Index"] == test_df["Index"]).all(), "index alignment mismatch: submission indices diverge from baseline requirements."
    print("spatio-temporal index alignment sequence verified.")
    
    # check 4: missing field validations
    assert not submission_df.isnull().any().any(), "data integrity violation: matrix houses null/nan elements."
    print("empty elements/nan presence scan passed.")
    
    # check 5: demand index value boundaries
    assert (submission_df["demand"] >= 0).all() and (submission_df["demand"] <= 1).all(), "mathematical violation: predictions stray outside raw [0, 1] thresholds."
    print("boundary index compliance checks passed: records are contained within [0, 1].")
    
    # check 6: baseline statistical distribution diagnostics
    print("\n=== SYSTEM STANDINGS: DEMAND MATRIX SNAPSHOT ===")
    print(submission_df["demand"].describe())
    
    # check 7: output distribution variance checks (catches structural collapsing anomalies)
    unique_prediction_count = submission_df["demand"].nunique()
    print(f"tracked functional unique value mutations: {unique_prediction_count}")
    assert unique_prediction_count > 100, "degeneration warning: too few unique estimates found. output array might be constant."
    
    print("\nALL SYSTEM DATA INTEGRITY VALIDATIONS PASSED SUCCESSFULY!")

if __name__ == "__main__":
    validate_submission_integrity()