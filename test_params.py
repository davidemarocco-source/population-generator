
import sys
import os
import pandas as pd
import numpy as np

# Setup
sys.path.append(os.getcwd())
from src.model import FactorModel, PopulationGenerator
from src.utils import create_simple_structure

def test_population_params():
    print("Testing Population Parameters...")
    
    # Setup Model
    n_factors = 1
    items_per_factor = [10]
    loadings = create_simple_structure(n_factors, items_per_factor, reliability=0.8)
    model = FactorModel(loadings=loadings)
    generator = PopulationGenerator(model)
    
    # 1. Test Continuous Data with Shift
    target_mean = 100.0
    target_std = 15.0
    df = generator.generate(n_samples=5000, mean=target_mean, std=target_std, likert_points=None)
    
    actual_means = df.mean().mean()
    actual_stds = df.std().mean()
    
    print(f"Continuous Target Mean: {target_mean}, Actual (Mean of Means): {actual_means:.2f}")
    print(f"Continuous Target SD: {target_std}, Actual (Mean of SDs): {actual_stds:.2f}")
    
    if abs(actual_means - target_mean) < 1.0 and abs(actual_stds - target_std) < 1.0:
        print("PASS: Continuous parameters within range.")
    else:
        print("FAIL: Continuous parameters off.")
        
    # 2. Test Likert Data with Shift (Extreme Mean)
    # If we shift mean by +1.0 (1 SD), we expect higher scores
    # Baseline (Mean 0)
    df_base = generator.generate(n_samples=1000, mean=0.0, std=1.0, likert_points=5)
    mean_base = df_base.values.mean()
    
    # Shifted (Mean 1.0)
    df_high = generator.generate(n_samples=1000, mean=1.0, std=1.0, likert_points=5)
    mean_high = df_high.values.mean()
    
    print(f"Likert Base Mean: {mean_base:.2f}")
    print(f"Likert High Mean: {mean_high:.2f}")
    
    if mean_high > mean_base + 0.5:
        print("PASS: Shifted Likert population has significantly higher mean.")
    else:
        print("FAIL: Shifted Likert population not significantly higher.")

if __name__ == "__main__":
    test_population_params()
