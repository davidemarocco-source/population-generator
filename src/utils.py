import json
import numpy as np
from typing import Dict, Any, Union, List

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load model configuration from a JSON file.
    Expected format:
    {
        "loadings": [[0.8, 0], ...],
        "correlations": [[1.0, 0.3], ...],  (optional)
        "uniqueness": [0.2, 0.3, ...],      (optional)
        "n_samples": 1000                   (optional default)
    }
    """
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config

def create_simple_structure(
    n_factors: int, 
    n_items_per_factor: Union[int, List[int]], 
    reliability: float = 0.8,
    cross_loading: float = 0.0
) -> np.ndarray:
    """
    Helper to create a simple structure loading matrix.
    
    Args:
        n_factors (int): Number of latent factors.
        n_items_per_factor (int or List[int]): Number of items per factor. 
                                               If int, same for all factors.
        reliability (float): Target reliability for items (squared loading).
                             loading = sqrt(reliability).
        cross_loading (float): Value for cross-loadings (default 0.0).
        
    Returns:
        np.ndarray: Loadings matrix (total_items x n_factors).
    """
    if isinstance(n_items_per_factor, int):
        items_counts = [n_items_per_factor] * n_factors
    else:
        items_counts = n_items_per_factor
        
    if len(items_counts) != n_factors:
        raise ValueError("Length of n_items_per_factor list must match n_factors")
        
    n_vars = sum(items_counts)
    loadings = np.full((n_vars, n_factors), cross_loading)
    
    # Calculate main loading from reliability
    # Var(Item) = Loading^2 * Var(Factor) + Uniqueness
    # If standardized, Var(Item)=1, Var(Factor)=1 (diag Phi), Uniqueness = 1 - Loading^2
    # So Loading = sqrt(Reliability)
    main_loading = np.sqrt(reliability)
    
    current_idx = 0
    for f, count in enumerate(items_counts):
        end_idx = current_idx + count
        loadings[current_idx:end_idx, f] = main_loading
        current_idx = end_idx
        
    return loadings
