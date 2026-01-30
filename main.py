import argparse
import sys
import numpy as np
import pandas as pd
from typing import List
from src.model import FactorModel, PopulationGenerator
from src.utils import load_config, create_simple_structure

def parse_structure_string(s: str) -> List[int]:
    """Parse '5,10,5' into [5, 10, 5]"""
    try:
        return [int(x) for x in s.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("Structure must be comma-separated integers (e.g., '10,10,5')")

def main():
    parser = argparse.ArgumentParser(description="Psychometric Population Generator")
    parser.add_argument('--output', type=str, required=True, help="Path to output CSV file")
    parser.add_argument('--samples', type=int, default=1000, help="Number of subjects (N)")
    
    # Configuration Group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', type=str, help="Path to JSON configuration file")
    group.add_argument('--preset', type=str, choices=['simple_1f', 'simple_3f'], help="Use a built-in preset")
    group.add_argument('--structure', type=parse_structure_string, help="Define structure directly: items per factor (e.g., '10,10,5' for 3 factors)")
    
    # Model Parameters (for Structure/Preset)
    parser.add_argument('--reliability', type=float, default=0.8, help="Target reliability (0.0-1.0). Determines loadings.")
    parser.add_argument('--noise', type=float, help="Alternative to reliability. reliability = 1 - noise.")
    parser.add_argument('--likert', type=int, help="Discretize output to Likert scale (2-10).")
    parser.add_argument('--loading_noise', type=float, default=0.0, help="Random noise to add to loading matrix (0.0-1.0).")
    
    args = parser.parse_args()
    
    # Resolve reliability
    reliability = args.reliability
    if args.noise is not None:
        reliability = 1.0 - args.noise
        
    loadings = None
    phi = None
    psi = None
    
    if args.config:
        try:
            config = load_config(args.config)
            loadings = config.get('loadings')
            phi = config.get('correlations')
            psi = config.get('uniqueness')
            # If config is used, other CLI params like reliability are ignored
            # unless we decided to merge them, but simplest is Config Authoritative.
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)
            
    elif args.structure:
        # User defined structure via CLI
        n_factors = len(args.structure)
        loadings = create_simple_structure(
            n_factors=n_factors,
            n_items_per_factor=args.structure,
            reliability=reliability
        )
        # Default orthogonal, can add --correlations later if needed or default to orthogonal
        
    elif args.preset:
        if args.preset == 'simple_1f':
            loadings = create_simple_structure(
                n_factors=1, 
                n_items_per_factor=10, 
                reliability=reliability
            )
        elif args.preset == 'simple_3f':
            loadings = create_simple_structure(
                n_factors=3, 
                n_items_per_factor=5, 
                reliability=reliability
            )
            # Add some correlation
            phi = np.array([
                [1.0, 0.3, 0.3],
                [0.3, 1.0, 0.3],
                [0.3, 0.3, 1.0]
            ])

    # Apply Loading Noise
    if loadings is not None and args.loading_noise > 0:
        noise = np.random.uniform(-args.loading_noise, args.loading_noise, size=loadings.shape)
        loadings = loadings + noise
        loadings = np.clip(loadings, -1.0, 1.0)

    # Initialize Model
    try:
        model = FactorModel(loadings=loadings, factor_correlations=phi, uniqueness=psi)
        print(f"Model Initialized: {model.n_vars} Variables, {model.n_factors} Factors")
        print(f"Target Reliability: {reliability}")
    except Exception as e:
        print(f"Error initializing FactorModel: {e}")
        sys.exit(1)

    # Generate Data
    generator = PopulationGenerator(model)
    print(f"Generating {args.samples} samples...")
    try:
        df = generator.generate(n_samples=args.samples, likert_points=args.likert)
        df.to_csv(args.output, index=False)
        print(f"Data saved to {args.output}")
    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
