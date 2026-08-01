import pandas as pd
import numpy as np
import logging
import warnings
import re

# Import our custom modules
from src.data_ingestion import LocalMFPipeline
from src.models import fit_hmm_regime
from src.portfolio import HRPAllocator, rebalance_industry_grade
from src.execution import SIPExecutionEngine

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ==========================================
# 1. Preprocessing Helpers
# ==========================================
def clean_name(name):
    """Strips variants to group the same fund across different plans."""
    name = str(name).lower()
    name = re.sub(r'[-_/()]+', ' ', name)
    name = re.sub(
        r'\b(direct|regular|growth|idcw|dividend|payout|reinvestment|'
        r'income distribution|capital withdrawal|cum|bonus|plan|option)\b',
        ' ',
        name
    )
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def pick_best(cols, nav_df):
    """Prioritizes Direct Growth, falls back to Regular Growth, then least missing data."""
    preferred = [c for c in cols if 'direct' in c.lower() and 'growth' in c.lower()]
    if preferred: return preferred[0]
    
    secondary = [c for c in cols if 'growth' in c.lower() and 'idcw' not in c.lower()]
    if secondary: return secondary[0]
    
    return min(cols, key=lambda c: nav_df[c].isna().mean())

# ==========================================
# 2. Walk-Forward Engine
# ==========================================
class WalkForwardEngine:
    """Handles chronological train/test splits for the HMM and HRP models."""
    def __init__(self, data, train_window_days, test_window_days):
        self.data = data.copy()
        self.train_window = train_window_days
        self.test_window = test_window_days

    def generate_splits(self):
        total_days = len(self.data)
        current_start = 0
        while current_start + self.train_window < total_days:
            train_end = current_start + self.train_window
            test_end = min(train_end + self.test_window, total_days)
            yield (
                self.data.iloc[current_start:train_end].copy(),
                self.data.iloc[train_end:test_end].copy()
            )
            current_start += self.test_window

    def run(self, max_turnover=0.20, dust_limit=0.05):
        weights_history = {}
        month_end_dates = self.data.resample('M').last().index
        current_executed_weights = None
        
        for train_data, test_data in self.generate_splits():
            for date in test_data.index:
                if date not in month_end_dates:
                    continue
                
                historical_data = self.data.loc[:date].copy()
                fund_returns = historical_data.pct_change().dropna()
                fund_returns = fund_returns.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(fund_returns) < 30:
                    continue
                    
                # 1. Regime Detection
                regime_series = fit_hmm_regime(fund_returns.mean(axis=1))
                current_regime = regime_series.loc[date] if date in regime_series.index else "bear"
                
                # 2. Generate Weights
                if current_regime == "bear":
                    target_weights = pd.Series(dtype=float)
                else:
                    allocator = HRPAllocator()
                    lookback_data = fund_returns.tail(126) # ~6 months of trading days
                    if len(lookback_data) < 30:
                        target_weights = pd.Series(dtype=float)
                    else:
                        target_weights = allocator.execute_hrp(lookback_data)
                        
                if target_weights.empty:
                    continue
                    
                # 3. Apply Turnover Constraints
                target_sum = target_weights.sum()
                if current_executed_weights is None or target_sum < 0.99:
                    final_weights = target_weights
                else:
                    all_funds = list(set(target_weights.index).union(current_executed_weights.index))
                    curr_array = np.array([current_executed_weights.get(f, 0.0) for f in all_funds])
                    tgt_array = np.array([target_weights.get(f, 0.0) for f in all_funds])
                    try:
                        exec_array = rebalance_industry_grade(curr_array, tgt_array, max_turnover)
                        final_weights = pd.Series(exec_array, index=all_funds)
                        final_weights = final_weights[final_weights >= dust_limit]
                        if final_weights.sum() > 0:
                            final_weights = (final_weights / final_weights.sum()) * target_sum
                        else:
                            final_weights = target_weights
                    except Exception:
                        final_weights = target_weights
                        
                current_executed_weights = final_weights
                weights_history[date] = final_weights
                
        return weights_history


def main():
    logging.info("Starting Dynamic SIP Optimizer Pipeline...")
    
    ZIP_FILE_PATH = "data/mutual_funds_dataset.zip" 
    SIP_AMOUNT = 50000.0
    
    # The complete list of 50 base funds
    USER_FUNDS = [
        "Nippon India Large Cap Fund", "ICICI Prudential Bluechip Fund", "JM Large Cap Fund",
        "IDBI India Top 100 Equity Fund", "HDFC Top 100 Fund", "Quant Mid Cap Fund – Direct Plan – Growth",
        "Motilal Oswal Midcap Fund", "Edelweiss Mid Cap Fund – Direct Plan – Growth",
        "PGIM India Midcap Opportunities Fund – Direct Plan – Growth", "Nippon India Growth Fund – Direct Plan – Growth",
        "Quant Small Cap Fund", "Bank of India Small Cap Fund", "Nippon India Small Cap Fund",
        "Canara Robeco Small Cap Fund", "Tata Small Cap Fund", "Parag Parikh Flexi Cap Fund",
        "HDFC Flexi Cap Fund", "Kotak Flexicap Fund", "UTI Flexi Cap Fund",
        "Aditya Birla Sun Life Flexi Cap Fund", "ICICI Prudential Infrastructure Fund",
        "Nippon India Power & Infra Fund", "HDFC Infrastructure Fund", "Tata Infrastructure Fund",
        "L&T Infrastructure Fund", "ICICI Prudential Banking and Financial Services Fund",
        "SBI Banking & Financial Services Fund", "Nippon India Banking Fund",
        "UTI Banking and Financial Services Fund", "Aditya Birla Sun Life Banking & Financial Services Fund",
        "SBI PSU Fund", "CPSE ETF", "ICICI Prudential Bharat 22 FOF", "Kotak PSU Bank ETF",
        "Nippon India ETF PSU Bank BeES", "ICICI Prudential Technology Fund",
        "Aditya Birla Sun Life Digital India Fund", "Franklin India Technology Fund",
        "Tata Digital India Fund", "SBI Technology Opportunities Fund", "HDFC Defence Fund",
        "Mirae Asset Healthcare Fund", "SBI Consumption Opportunities Fund", "ICICI Prudential FMCG Fund",
        "Aditya Birla Sun Life Global Commodities Fund", "HDFC Balanced Advantage Fund",
        "ICICI Prudential Equity & Debt Fund", "SBI Equity Hybrid Fund", "Mirae Asset Hybrid Equity Fund",
        "Kotak Equity Hybrid Fund"
    ]
    
    try:
        # --- PHASE 1: Data Ingestion & Preprocessing ---
        logging.info("Initiating Data Ingestion...")
        pipeline = LocalMFPipeline(ZIP_FILE_PATH)
        raw_data_dict = pipeline.load_user_funds(USER_FUNDS)
        
        logging.info("Merging and mapping fund data...")
        selected_codes = list(raw_data_dict.keys())
        nav_df = pd.concat(
            {code: raw_data_dict[code].set_index('date').sort_index()['nav'] for code in selected_codes},
            axis=1
        )
        nav_df.columns = [raw_data_dict[code]['Fund_Name'].iloc[0] for code in selected_codes]
        nav_df = nav_df.sort_index().ffill()
        nav_df = nav_df.dropna(axis=1, thresh=max(12, int(0.6 * len(nav_df))))
        
        # Deduplicate to finding the best 38 funds
        clean_map = {col: clean_name(col) for col in nav_df.columns}
        groups = {}
        for original, clean in clean_map.items():
            groups.setdefault(clean, []).append(original)
            
        selected_cols = [pick_best(cols, nav_df) for cols in groups.values()]
        nav_df = nav_df[selected_cols].copy()
        logging.info(f"Final number of unique funds mapped for processing: {len(nav_df.columns)}")
        
        # --- PHASE 2: Walk-Forward Engine ---
        logging.info("Running Walk-Forward Engine (HMM Regime + HRP Allocation)...")
        wf_engine = WalkForwardEngine(nav_df, train_window_days=756, test_window_days=252)
        weights_history = wf_engine.run()
        
        # --- PHASE 3: Execution Engine ---
        logging.info("Executing Dynamic SIP with Taxation & FIFO Queue...")
        exec_engine = SIPExecutionEngine(nav_df, weights_history, sip_amount=SIP_AMOUNT)
        portfolio_value = exec_engine.run_simulation()
        
        # --- PHASE 4: Export ---
        logging.info(f"Simulation Complete. Final Portfolio Gross Value: ₹{portfolio_value.iloc[-1]:,.2f}")
        
        snapshots_df = pd.DataFrame(exec_engine.monthly_snapshots).T
        snapshots_df.to_excel("Dynamic_Portfolio_Results.xlsx", index_label="Month")
        logging.info("Detailed monthly tracker saved to 'Dynamic_Portfolio_Results.xlsx'")
        
    except FileNotFoundError:
        logging.warning(f"Dataset not found at '{ZIP_FILE_PATH}'.")
        logging.info("Please place the historical CSV zip file in the 'data/' folder to run the full simulation.")

if __name__ == "__main__":
    main()
