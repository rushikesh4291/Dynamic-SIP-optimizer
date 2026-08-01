import numpy as np
import pandas as pd
import cvxpy as cp
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

def rebalance_industry_grade(current_w, target_w, max_turnover):
    """
    Uses convex optimization to minimize tracking error to target weights 
    while strictly enforcing a maximum turnover constraint.
    """
    n_assets = len(current_w)
    w_exec = cp.Variable(n_assets)
    
    # Minimize the distance between execution weights and ideal target weights
    objective = cp.Minimize(cp.sum_squares(w_exec - target_w))
    
    constraints = [
        w_exec >= 0,
        cp.sum(w_exec) == 1.0,
        # L1 Norm (sum of absolute deltas) <= max_turnover * 2 (since every buy has a sell)
        cp.norm(w_exec - current_w, 1) <= (max_turnover * 2)
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return w_exec.value

class HRPAllocator:
    """
    Hierarchical Risk Parity (HRP) Allocator.
    Allocates inverse to cluster variance, handling highly correlated assets automatically.
    """
    @staticmethod
    def get_cluster_var(cov, c_items):
        # Convert string labels to integer positions for iloc
        indices = cov.columns.get_indexer(c_items)
        cov_slice = cov.iloc[indices, indices]
        ivp = 1. / np.diag(cov_slice)
        ivp /= ivp.sum()
        return np.dot(ivp, np.dot(cov_slice, ivp))

    def get_rec_bipart(self, cov, sort_ix):
        w = pd.Series(1.0, index=sort_ix)
        c_items = [sort_ix]
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(c_items), 2):
                c_1 = c_items[i]
                c_2 = c_items[i+1]
                c_1_var = self.get_cluster_var(cov, c_1)
                c_2_var = self.get_cluster_var(cov, c_2)
                alpha = 1 - c_1_var / (c_1_var + c_2_var)
                w[c_1] *= alpha
                w[c_2] *= (1 - alpha)
        return w

    def execute_hrp(self, returns_df):
        cov = returns_df.cov()
        corr = returns_df.corr()
        dist = np.sqrt(0.5 * (1 - corr).clip(0, 2))
        link = sch.linkage(squareform(dist), 'ward')
        sort_ix = sch.leaves_list(link)
        ordered_tickers = corr.index[sort_ix].tolist()
        return self.get_rec_bipart(cov, ordered_tickers)
