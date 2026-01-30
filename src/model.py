import numpy as np
import pandas as pd
from typing import Optional, List, Union

class FactorModel:
    """
    Represents a psychometric Factor Analysis model.
    Model: Sigma = Lambda * Phi * Lambda^T + Psi
    """
    def __init__(
        self,
        loadings: np.ndarray,
        factor_correlations: Optional[np.ndarray] = None,
        uniqueness: Optional[np.ndarray] = None,
        variable_names: Optional[List[str]] = None,
        factor_names: Optional[List[str]] = None
    ):
        """
        Initialize the Factor Model.

        Args:
            loadings (np.ndarray): Lambda matrix (n_variables x n_factors).
            factor_correlations (np.ndarray, optional): Phi matrix (n_factors x n_factors). 
                                                        Defaults to Identity (orthogonal factors).
            uniqueness (np.ndarray, optional): Psi diagonal vector/matrix (n_variables). 
                                               If None, calculated as 1 - diag(Lambda * Phi * Lambda^T) (assuming standardized).
            variable_names (List[str], optional): Names for the observed variables.
            factor_names (List[str], optional): Names for the latent factors.
        """
        self.loadings = np.array(loadings)
        self.n_vars, self.n_factors = self.loadings.shape

        # Default Factor Correlations (Identity -> Orthogonal)
        if factor_correlations is None:
            self.phi = np.eye(self.n_factors)
        else:
            self.phi = np.array(factor_correlations)
            if self.phi.shape != (self.n_factors, self.n_factors):
                raise ValueError(f"Factor correlations must be {self.n_factors}x{self.n_factors}")

        # Default Uniqueness (1 - communality)
        # Communality = diag(Lambda * Phi * Lambda^T)
        implied_communalities = np.diag(self.loadings @ self.phi @ self.loadings.T)
        
        if uniqueness is None:
            # Assume standardized variables (variance=1)
            # If communality > 1, this will result in negative uniqueness (Heywood case), which is technically possible but problematic for generation unless intended.
            # We'll allow it but warn if we had a logger.
            self.psi = np.diag(1.0 - implied_communalities)
            # Clip negative variances to a small epsilon for stability if needed? 
            # For now, let's keep it raw math. If user provides bad loadings, they get bad sigma.
        else:
            if uniqueness.ndim == 1:
                self.psi = np.diag(uniqueness)
            else:
                self.psi = uniqueness
            
            if self.psi.shape != (self.n_vars, self.n_vars):
                 raise ValueError(f"Uniqueness matrix must be {self.n_vars}x{self.n_vars}")

        self.variable_names = variable_names if variable_names else [f"V{i+1}" for i in range(self.n_vars)]
        self.factor_names = factor_names if factor_names else [f"F{i+1}" for i in range(self.n_factors)]

    def get_covariance_matrix(self) -> np.ndarray:
        """
        Calculate the model-implied covariance matrix Sigma.
        Sigma = Lambda * Phi * Lambda^T + Psi
        """
        common_part = self.loadings @ self.phi @ self.loadings.T
        sigma = common_part + self.psi
        return sigma

class PopulationGenerator:
    """
    Generates data samples based on a FactorModel.
    """
    def __init__(self, model: FactorModel):
        self.model = model
        self.sigma = self.model.get_covariance_matrix()
        self.mu = np.zeros(self.model.n_vars) # Assume mean 0 for simplicity

    def generate(self, n_samples: int, seed: Optional[int] = None, likert_points: Optional[int] = None, 
                 mean: float = 0.0, std: float = 1.0) -> pd.DataFrame:
        """
        Generate N samples from the multivariate normal distribution defined by the model.
        
        Args:
            n_samples (int): Number of subjects.
            seed (int, optional): Random seed.
            likert_points (int, optional): If set (e.g., 5, 7), output will be discretized 
                                           to integers in range [1, likert_points].
            mean (float): Population mean (shift). Default 0.0.
            std (float): Population standard deviation (scale). Default 1.0.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Check for positive definiteness
        try:
            data = np.random.multivariate_normal(self.mu, self.sigma, n_samples)
        except np.linalg.LinAlgError as e:
            # Fallback for non-positive definite matrices (sometimes happens with high correlations)
            # Try finding nearest PD matrix or just raise?
            # For this simple tool, let's warn and add epsilon diagonal
            print("Warning: Matrix not positive definite. Adding ridge.")
            self.sigma += np.eye(self.model.n_vars) * 1e-4
            data = np.random.multivariate_normal(self.mu, self.sigma, n_samples)

        # Apply Population Mean and SD scaling
        # data is currently N(0, Sigma). 
        # We transform to N(Mean, SD^2 * Sigma)
        data = data * std + mean

        if likert_points is not None and likert_points > 1:
            # Discretize data to 1..K
            # Underlying assumption: Data is N(0, 1) (if standardized)
            # If user shifted the mean/std, this will result in ceiling/floor effects or shifted distributions
            # which is likely the intended behavior (e.g., "high ability population").
            
            # Range [-3, 3] coverage of standard unit normal is mapped to the scale.
            # Width in sigmas: approx 6
            # Target width: K - 1
            scale_factor = (likert_points - 1) / 6.0
            shift = (likert_points + 1) / 2.0
            
            scaled = data * scale_factor + shift
            discretized = np.round(scaled)
            
            # Clip to ensure valid range (handle outliers beyond 3 sigma)
            discretized = np.clip(discretized, 1, likert_points)
            data = discretized.astype(int)

        df = pd.DataFrame(data, columns=self.model.variable_names)
        return df
