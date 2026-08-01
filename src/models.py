import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import warnings

# Suppress hmmlearn convergence warnings for clean terminal output
warnings.filterwarnings("ignore")

def fit_hmm_regime(returns_series, macro_df=None, n_components=2, macro_weight=0.5):
    """
    Fits a Gaussian Hidden Markov Model to identify market regimes (Bull/Bear).
    Supports multivariate features (Equity Returns + Macro Data) with feature weighting.
    """
    series = returns_series.dropna()
    
    # Require minimum data for convergence
    if len(series) < 30:
        return pd.Series("bull", index=returns_series.index)
        
    # Merge Macro Data if provided
    if macro_df is not None:
        X_df = pd.concat([series, macro_df], axis=1).dropna()
    else:
        X_df = series.to_frame()
        
    # Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)
    
    # Apply Feature Weighting 
    # This forces the HMM to prioritize Equity Returns over Macro 'Noise'
    if macro_df is not None:
        weights = np.ones(X_scaled.shape[1])
        weights[1:] = macro_weight  # Dampen macro indicators
        X_scaled = X_scaled * weights
        
    model = GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=200,
        tol=1e-3,
        random_state=42
    )
    
    model.fit(X_scaled)
    states = model.predict(X_scaled)
    
    # Identify Bull State based on the first feature (Equity Returns)
    means = model.means_[:, 0]
    bull_state = means.argmax()
    
    regime = pd.Series(states, index=X_df.index)
    regime = regime.map(lambda s: "bull" if s == bull_state else "bear")
    
    full_regime = pd.Series(index=returns_series.index, dtype=object)
    full_regime.loc[regime.index] = regime
    full_regime = full_regime.ffill().bfill()
    
    return full_regime
