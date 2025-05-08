# === run_wfv_with_params.py (signal timing + adaptive ATR + R:R + hold filter) ===
def run_wfv_with_params(df, features, model_cls, best_params,
                        train_size=60, test_size=10, confidence_threshold=0.6,
                        atr_mult=1.5, rr_ratio=2.0, trailing_stop=False, short_signals=False,
                        initial_capital=1000.0, risk_fraction=0.01, max_hold_days=10,
                        rsi_filter=70, atr_threshold=None, atr_mult_low=1.2,
                        confidence_margin=0.05, plot=True, commission_per_trade=0.0, slippage_points=0.0,
                        vix_series=None, vix_scaling=False, rr_tuning=False):
    import matplotlib.pyplot as plt
    returns_all, dates_all, probas_all, trade_log = [], [], [], []
    equity = initial_capital

    # Precompute VIX stats if available
    if vix_series is not None and vix_scaling:
        vix_min, vix_max = vix_series.min(), vix_series.max()

    for start in range(0, len(df) - train_size - test_size, test_size):
        train = df.iloc[start:start + train_size].copy()
        test = df.iloc[start + train_size:start + train_size + test_size].copy()

        train['target'] = (np.log(train['Close'].shift(-1) / train['Close']) > 0).astype(int)
        train.dropna(inplace=True)

        X_train, y_train = train[features], train['target']
        model = model_cls(**best_params)
        model.fit(X_train, y_train)

        X_test = test[features]
        probas = model.predict_proba(X_test)[:, 1]
        signals = []

        for i, prob in enumerate(probas):
            row = test.iloc[i]
            rsi = row.get('rsi', 50)
            atr = row['atr']

            # Skip long trades when ATR > 100
            if prob > confidence_threshold and atr_threshold and atr > atr_threshold:
                signals.append(0)
                continue

            if rsi > rsi_filter:
                signals.append(0)
                continue

            # Skip trades too close to neutral confidence
            if abs(prob - 0.5) < confidence_margin:
                signals.append(0)
            elif prob > confidence_threshold:
                signals.append(1)
            elif short_signals and prob < (1 - confidence_threshold):
                signals.append(-1)
            else:
                signals.append(0)

        test['target_return'] = np.log(test['Close'].shift(-1) / test['Close'])
        test['position'] = signals
        test['strategy_return'] = 0.0

        for i, signal in enumerate(signals):
            if signal == 0:
                continue

            row = test.iloc[i]
            entry_date = test.index[i]
            entry_price, atr = row['Close'], row['atr']

            vix_value = vix_series.loc[entry_date] if vix_series is not None and entry_date in vix_series.index else None
            rf, rr, am = risk_fraction, rr_ratio, atr_mult

            if vix_value and vix_scaling:
                norm_vix = (vix_value - vix_min) / (vix_max - vix_min + 1e-8)
                rf = risk_fraction / (1 + norm_vix)

            if vix_value and rr_tuning and vix_value > 25:
                rr *= 0.8
                am *= 0.8

            adj_atr_mult = atr_mult_low if atr_threshold and atr > atr_threshold else atr_mult

            stop_loss = entry_price - signal * atr * adj_atr_mult * -1
            take_profit = entry_price + signal * atr * adj_atr_mult * rr_ratio
            max_fav_price = entry_price

            capital_at_risk = equity * risk_fraction
            risk_per_point = abs(entry_price - stop_loss)
            contracts = capital_at_risk / risk_per_point if risk_per_point > 0 else 0

            exit_price, reason, j = None, None, i
            for j in range(i + 1, min(i + max_hold_days + 1, len(test))):
                row_j = test.iloc[j]
                next_high, next_low = row_j['High'], row_j['Low']
                exit_date = test.index[j]

                if trailing_stop:
                    if signal == 1:
                        max_fav_price = max(max_fav_price, next_high)
                        trail_price = max_fav_price - atr * adj_atr_mult
                        if next_low <= trail_price:
                            exit_price, reason = trail_price, 'trailing_stop'
                    else:
                        max_fav_price = min(max_fav_price, next_low)
                        trail_price = max_fav_price + atr * adj_atr_mult
                        if next_high >= trail_price:
                            exit_price, reason = trail_price, 'trailing_stop'

                if not exit_price:
                    if signal == 1 and next_low <= stop_loss:
                        exit_price, reason = stop_loss, 'stop_loss'
                    elif signal == 1 and next_high >= take_profit:
                        exit_price, reason = take_profit, 'take_profit'
                    elif signal == -1 and next_high >= stop_loss:
                        exit_price, reason = stop_loss, 'stop_loss'
                    elif signal == -1 and next_low <= take_profit:
                        exit_price, reason = take_profit, 'take_profit'

                if exit_price:
                    break

            if not exit_price:
                exit_price = test.iloc[j]['Close']
                exit_date = test.index[j]
                reason = 'time_exit'

            # Apply slippage and commission
            slippage = slippage_points * contracts
            gross_pnl = signal * (exit_price - entry_price)
            net_dollar_pnl = (gross_pnl * contracts) - commission_per_trade - slippage
            scaled_return = net_dollar_pnl / equity if equity > 0 else 0
            equity *= (1 + scaled_return)

            test.at[entry_date, 'strategy_return'] = scaled_return
            trade_log.append({
                'entry_date': entry_date, 'exit_date': exit_date,
                'entry_price': entry_price, 'exit_price': exit_price,
                'atr': atr, 'reason': reason, 'held_days': (exit_date - entry_date).days,
                'contracts': contracts, 'dollar_pnl': net_dollar_pnl,
                'equity': equity, 'direction': 'long' if signal == 1 else 'short',
                'position': signal, 'confidence': probas[i]
            })

        returns_all.extend(test['strategy_return'].values)
        probas_all.extend(probas)
        dates_all.extend(test.index)

    df_results = pd.DataFrame({'date': dates_all, 'strategy_return': returns_all, 'proba': probas_all}).set_index('date')
    df_results['cum_return'] = df_results['strategy_return'].cumsum()
    df_results['cum_pct'] = np.exp(df_results['cum_return']) - 1

    stats = {
        "sharpe": df_results['strategy_return'].mean() / df_results['strategy_return'].std() * np.sqrt(252),
        "hit_rate": (df_results['strategy_return'] > 0).mean(),
        "max_drawdown": (df_results['cum_return'] - df_results['cum_return'].cummax()).min(),
        "trade_count": len(trade_log),
        "score": df_results['strategy_return'].mean() / df_results['strategy_return'].std() * np.sqrt(252)
                - abs((df_results['cum_return'] - df_results['cum_return'].cummax()).min()) * 0.5
                + (df_results['strategy_return'] > 0).mean() * 0.3
    }

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(df_results['cum_pct'], label="Strategy")
        plt.title("Cumulative Return")
        plt.ylabel("Return")
        plt.legend()
        plt.grid(True)
        plt.show()

    return df_results, pd.DataFrame(trade_log), stats
