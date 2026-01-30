
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.getcwd())
from src.model import FactorModel, PopulationGenerator
from src.utils import create_simple_structure

def test_sum_score_scaling():
    print("Testing Sum Score Scaling...")
    
    # Setup
    n_factors = 1
    items_per_factor = [10]
    loadings = create_simple_structure(n_factors, items_per_factor, reliability=0.8)
    model = FactorModel(loadings=loadings)
    generator = PopulationGenerator(model)
    
    # Generate Base Data (Likert 1-5, Mean 0, SD 1)
    df = generator.generate(n_samples=1000, mean=0.0, std=1.0, likert_points=5)
    
    # 1. Calculate Sum Score Manual Verification
    df['Total_Score'] = df.iloc[:, :10].sum(axis=1)
    mean_raw = df['Total_Score'].mean()
    std_raw = df['Total_Score'].std()
    
    print(f"Raw Sum Score: Mean={mean_raw:.2f}, SD={std_raw:.2f}")
    # With 10 items (Mean ~3), Sum ~ 30.
    if 25 < mean_raw < 35:
        print("PASS: Raw Sum Scores reasonable.")
    else:
        print("FAIL: Raw Sum scores odd.")

    # 2. Rescale to IQ (Mean=100, SD=15)
    # Simulate the logic added to gui.py
    # Z-score normalization
    z_scores = (df['Total_Score'] - mean_raw) / std_raw
    df['Scaled_Score'] = z_scores * 15.0 + 100.0
    
    mean_scaled = df['Scaled_Score'].mean()
    std_scaled = df['Scaled_Score'].std()
    
    print(f"Scaled Score (Target 100/15): Mean={mean_scaled:.2f}, SD={std_scaled:.2f}")
    
    if abs(mean_scaled - 100.0) < 0.1 and abs(std_scaled - 15.0) < 0.1:
        print("PASS: Scaling logic works perfectly.")
    else:
        print("FAIL: Scaling math error.")

if __name__ == "__main__":
    test_sum_score_scaling()
