
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.getcwd())
from src.model import FactorModel, PopulationGenerator
from src.utils import create_simple_structure

def test_fix_all_5s():
    print("Testing Fix for All 5s Issue...")
    
    # Setup Model
    n_factors = 1
    items_per_factor = [10]
    loadings = create_simple_structure(n_factors, items_per_factor, reliability=0.8)
    model = FactorModel(loadings=loadings)
    generator = PopulationGenerator(model)
    
    # 1. Test High Mean with Likert Disabled (likert_points=None)
    # This was failing (producing 5s) because user couldn't disable Likert.
    target_mean = 100.0
    target_std = 15.0
    
    df_continuous = generator.generate(n_samples=1000, mean=target_mean, std=target_std, likert_points=None)
    
    mean_val = df_continuous.values.mean()
    max_val = df_continuous.values.max()
    
    print(f"Continuous Mode (Mean=100): Avg={mean_val:.2f}, Max={max_val:.2f}")
    
    if mean_val > 99.0 and max_val > 100.0:
        print("PASS: Continuous generation works with high mean (no ceiling effect).")
    else:
        print("FAIL: Continuous generation seems constrained.")

    # 2. Test High Latent Mean with Likert Enabled (Expected Ceiling)
    # This confirms WHY the user saw all 5s.
    df_likert = generator.generate(n_samples=1000, mean=100.0, std=15.0, likert_points=5)
    likert_counts = df_likert.apply(pd.Series.value_counts).fillna(0).sum(axis=1)
    
    print(f"Likert Mode (Mean=100, Z-score space): Counts = {likert_counts.to_dict()}")
    
    if 5 in likert_counts and likert_counts[5] > 9000: # 10 items * 1000 samples = 10000
        print("PASS (Expected Behavior): High Latent Mean causes ceiling effect in Likert mode.")
    
    # 3. Test Normal Likert (Zero Latent Mean)
    df_normal = generator.generate(n_samples=1000, mean=0.0, std=1.0, likert_points=5)
    mean_normal = df_normal.values.mean()
    print(f"Likert Mode (Mean=0): Avg={mean_normal:.2f}")
    
    if 2.5 < mean_normal < 3.5:
        print("PASS: Normal Likert generation works.")

if __name__ == "__main__":
    test_fix_all_5s()
