import pandas as pd
import numpy as np
from src.utils import create_simple_structure

def verify_covariance(data_path, n_factors, vars_per_factor, loading_strength):
    print(f"Verifying {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded data shape: {df.shape}")
    
    # Calculate Sample Covariance
    S = df.cov().values
    
    # Expected approximate covariance for items on same factor: loading^2
    # For items on different factors (orthogonal): 0
    # For diagonal: 1 (if standardized and uniqueness = 1 - comm)
    
    expected_same_factor_cov = loading_strength ** 2
    
    # Check first two items (assumed on same factor)
    cov_0_1 = S[0, 1]
    print(f"Cov(V1, V2) [Same Factor] = {cov_0_1:.4f} (Expected ~{expected_same_factor_cov:.4f})")
    
    # Check item 0 (Factor 1) and item N (Factor 2)
    # Assuming orthogonal for simple_1f or checking cross-cov otherwise
    if n_factors > 1:
        idx_f1 = 0
        idx_f2 = vars_per_factor # First item of second factor
        cov_cross = S[idx_f1, idx_f2]
        print(f"Cov(V1, V{idx_f2+1}) [Diff Factor] = {cov_cross:.4f}")

    # Check Variance
    var_0 = S[0, 0]
    print(f"Var(V1) = {var_0:.4f} (Expected ~1.0)")

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1]
    verify_covariance(data_path, 1, 10, 0.7)
