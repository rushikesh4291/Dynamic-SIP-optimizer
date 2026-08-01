import pandas as pd
import numpy as np
from dataclasses import dataclass

# --- CONSTANTS ---
EXIT_LOAD_PCT = 0.01
TXN_COST_BPS = 0.0005
STT_BPS = 0.001

STCG_RATE = 0.20
LTCG_RATE = 0.125
LTCG_EXEMPTION_LIMIT = 125000.0

@dataclass
class Lot:
    buy_date: pd.Timestamp
    nav: float
    units: float

class TaxLedger:
    """Tracks and dynamically calculates STCG and LTCG across financial years."""
    def __init__(self):
        self.current_fy = None
        self.fy_accumulated_ltcg = 0.0
        self.total_taxes_paid = 0.0

    def _get_financial_year(self, date: pd.Timestamp):
        if date.month >= 4:
            return f"{date.year}-{date.year + 1}"
        return f"{date.year - 1}-{date.year}"

    def process_sale_for_taxes(self, sold_lots, current_date, current_nav):
        fy = self._get_financial_year(current_date)
        
        # Reset exemption limit for new Financial Year
        if fy != self.current_fy:
            self.current_fy = fy
            self.fy_accumulated_ltcg = 0.0
            
        tax_liability = 0.0
        for lot in sold_lots:
            days_held = (current_date - lot.buy_date).days
            profit = (current_nav - lot.nav) * lot.units
            
            if profit > 0:
                if days_held < 365:
                    tax_liability += profit * STCG_RATE
                else:
                    self.fy_accumulated_ltcg += profit
                    
        # Apply the LTCG Exemption Threshold
        if self.fy_accumulated_ltcg > LTCG_EXEMPTION_LIMIT:
            taxable_ltcg = self.fy_accumulated_ltcg - LTCG_EXEMPTION_LIMIT
            tax_liability += taxable_ltcg * LTCG_RATE
            self.fy_accumulated_ltcg = LTCG_EXEMPTION_LIMIT
            
        self.total_taxes_paid += tax_liability
        return tax_liability


def calculate_net_sell_proceeds(sold_lots_df, current_date, current_nav):
    """Vectorized calculation of exit loads, STT, and transaction costs."""
    days_held = (current_date - sold_lots_df['buy_date']).dt.days
    exit_load_mask = days_held < 365
    gross_value = sold_lots_df['units'] * current_nav
    
    exit_load_fees = np.where(exit_load_mask, gross_value * EXIT_LOAD_PCT, 0.0)
    stt_fees = gross_value * STT_BPS
    txn_fees = gross_value * TXN_COST_BPS
    
    net_proceeds = np.sum(gross_value - exit_load_fees - stt_fees - txn_fees)
    total_exit_load = np.sum(exit_load_fees)
    
    return net_proceeds, total_exit_load


class SIPExecutionEngine:
    """Point-in-Time execution engine handling rebalancing, FIFO queueing, and taxation."""
    def __init__(self, nav_df, weights_history, sip_amount=50000.0, liquid_return=0.065):
        self.nav_df = nav_df.copy()
        self.weights_history = weights_history
        self.sip_amount = sip_amount
        
        self.liquid_return = liquid_return 
        self.daily_liquid_rate = (1 + self.liquid_return)**(1/252) - 1
        
        self.tax_ledger = TaxLedger()
        self.cash = 0.0
        self.holdings = {}
        
        # State tracking
        self.portfolio_value_history = []
        self.tax_history = []
        self.exit_load_history = []
        self.monthly_snapshots = {}
        self.current_month_tax = 0.0
        self.current_month_exit_load = 0.0

    def _sell_fifo(self, fund, units_to_sell):
        sold_record = []
        epsilon = 1e-6
        lots = self.holdings.get(fund, [])
        
        while units_to_sell > epsilon and lots:
            oldest_lot = lots[0]
            if oldest_lot.units <= units_to_sell + epsilon:
                units_to_sell -= oldest_lot.units
                sold_record.append(lots.pop(0))
            else:
                oldest_lot.units -= units_to_sell
                if oldest_lot.units < epsilon:
                    lots.pop(0)
                sold_record.append(Lot(oldest_lot.buy_date, oldest_lot.nav, units_to_sell))
                units_to_sell = 0.0
                
        self.holdings[fund] = lots
        return sold_record

    def _execute_rebalance(self, target_weights, current_navs, current_date):
        total_value = self.cash + sum(sum(l.units for l in lots) * current_navs.get(f, 0) 
                                      for f, lots in self.holdings.items())
        if total_value == 0:
            return
            
        target_values = target_weights * total_value
        current_values = {f: sum(l.units for l in lots) * current_navs.get(f, 0) 
                          for f, lots in self.holdings.items()}
                          
        sells, buys = {}, {}
        all_funds = set(target_weights.index).union(self.holdings.keys())
        tolerance = 0.02 
        
        for f in all_funds:
            tv = target_values.get(f, 0.0)
            cv = current_values.get(f, 0.0)
            diff = tv - cv
            
            if abs(diff) / total_value > tolerance:
                if diff < 0:
                    sells[f] = abs(diff) / current_navs[f]
                else:
                    buys[f] = diff
                    
        # 1. Execute Sells First 
        for f, units_to_sell in sells.items():
            sold_lots = self._sell_fifo(f, units_to_sell)
            if sold_lots:
                sold_df = pd.DataFrame([{'buy_date': l.buy_date, 'units': l.units} for l in sold_lots])
                net_proceeds, exit_load = calculate_net_sell_proceeds(sold_df, current_date, current_navs[f])
                tax_liability = self.tax_ledger.process_sale_for_taxes(sold_lots, current_date, current_navs[f])
                
                self.exit_load_history.append(exit_load)
                self.tax_history.append(tax_liability)
                self.cash += (net_proceeds - tax_liability)
                self.current_month_exit_load += exit_load
                self.current_month_tax += tax_liability
                
        # 2. Execute Buys
        total_buy_value = sum(buys.values())
        if total_buy_value > self.cash:
            scale_factor = self.cash / total_buy_value
            buys = {f: v * scale_factor for f, v in buys.items()}
            
        for f, buy_val in buys.items():
            if buy_val > 0:
                units_bought = buy_val / current_navs[f]
                if f not in self.holdings:
                    self.holdings[f] = []
                self.holdings[f].append(Lot(current_date, current_navs[f], units_bought))
                self.cash -= buy_val

    def run_simulation(self):
        sip_dates = pd.date_range(start=self.nav_df.index[0], end=self.nav_df.index[-1], freq='BMS')
        
        for current_date, current_navs in self.nav_df.iterrows():
            if self.cash > 0.01:
                self.cash *= (1 + self.daily_liquid_rate)
                
            if current_date in sip_dates:
                self.cash += self.sip_amount
                
            if current_date in self.weights_history:
                target_weights = self.weights_history[current_date]
                self._execute_rebalance(target_weights, current_navs, current_date)
                
            holdings_value = {f: sum(l.units for l in lots) * current_navs.get(f, 0) 
                              for f, lots in self.holdings.items() if sum(l.units for l in lots) > 0.001}
            pv = self.cash + sum(holdings_value.values())
            self.portfolio_value_history.append(pv)
            
            if current_date in sip_dates:
                snapshot = {
                    'Total_Portfolio_Value': round(pv, 2),
                    'Uninvested_Cash': round(self.cash, 2),
                    'Cash_Weight_Pct': round((self.cash / pv) * 100, 2) if pv > 0 else 100.0,
                    'Taxes_Paid_This_Month': round(self.current_month_tax, 2),
                    'Exit_Loads_Paid_This_Month': round(self.current_month_exit_load, 2)
                }
                for fund_name, fund_value in holdings_value.items():
                    snapshot[fund_name + '_Pct'] = round((fund_value / pv) * 100, 2)
                self.monthly_snapshots[current_date.strftime('%Y-%b')] = snapshot
                
                self.current_month_tax = 0.0
                self.current_month_exit_load = 0.0
                
        return pd.Series(self.portfolio_value_history, index=self.nav_df.index)
