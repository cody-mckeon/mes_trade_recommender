# === live_recommender.py ===
import numpy as np
import pandas as pd

class LiveTradeRecommender:
    def __init__(self, model, features, params, initial_capital=1000.0,
                 risk_fraction=0.01, atr_mult_low=1.2, vix_data=None):
        self.model = model
        self.features = features
        self.params = params
        self.initial_capital = initial_capital
        self.risk_fraction = risk_fraction
        self.atr_mult_low = atr_mult_low
        self.vix_data = vix_data

    def generate(self, row, verbose=True):
        # Extract inputs
        x = row[self.features].values.reshape(1, -1)
        proba = self.model.predict_proba(x)[0][1]
        confidence_threshold = self.params['confidence_threshold']
        confidence_margin = self.params.get('confidence_margin', 0.05)
        rr_ratio = self.params['rr_ratio']
        atr_mult = self.params['atr_mult']
        atr_threshold = self.params.get('atr_threshold', None)

        close = row['Close']
        atr = row['atr']
        rsi = row.get('rsi', 50)
        vix = self.vix_data.loc[row.name] if self.vix_data is not None and row.name in self.vix_data.index else None

        # Risk-adjusted ATR multiplier
        adj_atr_mult = self.atr_mult_low if atr_threshold and atr > atr_threshold else atr_mult

        # Skip trades too close to neutral
        if abs(proba - 0.5) < confidence_margin:
            return {"reason": "confidence_margin_rejection", "confidence": proba}

        if proba > confidence_threshold and rsi < self.params.get('rsi_filter', 70):
            direction = 'long'
            signal = 1
        elif proba < (1 - confidence_threshold) and self.params.get('short_signals', True):
            direction = 'short'
            signal = -1
        else:
            return {"reason": "threshold_not_crossed", "confidence": proba}

        # Reject long trades on high ATR if threshold hit
        if signal == 1 and atr_threshold and atr > atr_threshold:
            return {"reason": "high_atr_rejection", "atr": atr, "confidence": proba}

        # Risk and sizing
        stop_loss = close - signal * atr * adj_atr_mult * -1
        take_profit = close + signal * atr * adj_atr_mult * rr_ratio
        risk_per_point = abs(close - stop_loss)
        capital_at_risk = self.initial_capital * self.risk_fraction
        contracts = capital_at_risk / risk_per_point if risk_per_point > 0 else 0

        result = {
            'date': row.name,
            'direction': direction,
            'entry_price': close,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': proba,
            'expected_rr': rr_ratio,
            'contracts': contracts,
            'reason': 'Live signal with filtered confidence and ATR',
        }

        if verbose:
            print("\n📈 Live Trade Signal")
            for k, v in result.items():
                print(f"{k:>15}: {v}")

        return result
