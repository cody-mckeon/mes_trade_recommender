# Engineer the features
def add_features(df, fred_api_key, verbose=True):
    print("Adding features")

    if len(df) < 20:
        raise ValueError(f"🚫 Not enough rows to compute indicators safely. Need at least 20 rows, got {len(df)}.")

    close = df['Close']
    high = df['High']
    low = df['Low']
    print("Adding indicators")
    df['log_return'] = np.log(close / close.shift(1))
    df['rsi'] = RSIIndicator(close=df['Close'], window=14).rsi()
    df['macd'] = MACD(close=df['Close']).macd_diff()
    bb = BollingerBands(close=df['Close'])
    df['bb_mavg'] = bb.bollinger_mavg()
    df['bb_width'] = bb.bollinger_hband() - bb.bollinger_lband()
    df['atr'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()

    # Lagged and momentum
    print("Lagged and Momentum")
    df['rsi_lag1'] = df['rsi'].shift(1)
    df['macd_lag1'] = df['macd'].shift(1)
    df['momentum_3'] = df['Close'] / df['Close'].shift(3) - 1
    df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['vol_5'] = df['log_return'].rolling(window=5).std()

    # Ensure index is datetime for merging with FRED
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df.set_index('Date', inplace=True)
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or a 'Date' column to align with FRED data.")

    print("🕰 df.index sample:", df.index[:5])
    print("📐 Index dtype:", df.index.dtype)

    # === FRED API ===
    print(f"Fred api key:{fred_api_key}")
    # === Optional: Add FRED Macros ===
    if fred_api_key:
        print("Adding FRED macros")
        fred_api_url = "https://api.stlouisfed.org/fred/series/observations"
        fred_indicators = {
            "T10Y2Y": "yield_spread_10y_2y",
            "DFF": "fed_funds_rate",
            "CPIAUCSL": "cpi_inflation",
            "VIXCLS": "vix_level"
        }
        fred_data = {}
        for code, col_name in fred_indicators.items():
            if verbose:
                print(f"📦 Downloading FRED series: {code} → {col_name}")
            params = {
                "series_id": code,
                "api_key": fred_api_key,
                "file_type": "json",
                "observation_start": "2000-01-01",
            }
            response = requests.get(fred_api_url, params=params)
            if response.ok:
                observations = response.json().get("observations", [])
                fred_df = pd.DataFrame(observations)

                # ✅ Force-clean columns before proceeding
                fred_df.columns = [str(c) for c in fred_df.columns]

                # 🔍 Print column types
                print("🧪 FRED raw columns:", fred_df.columns.tolist())

                fred_df['date'] = pd.to_datetime(fred_df['date'])
                fred_df.set_index('date', inplace=True)
                fred_df['value'] = pd.to_numeric(fred_df['value'], errors='coerce')
                df[col_name] = fred_df['value'].reindex(df.index, method='ffill')
                fred_data[col_name] = fred_df['value']
                print(f"✅ Merged {col_name}, final dtype: {df[col_name].dtype}")

            else:
                print(f"⚠️ Failed to fetch {code}: {response.status_code}")

        print("📅 Data index range:", df.index.min(), "to", df.index.max())
         # === ES-VIX Correlation ===
        if 'vix_level' in fred_data:
            print("🧠 Calculating ES-VIX correlation")
            df_returns = pd.DataFrame(index=df.index)
            df_returns['ES_ret'] = df['log_return']
            df_returns['VIX_ret'] = np.log(fred_data['vix_level'] / fred_data['vix_level'].shift(1))
            df_returns['es_vix_corr'] = df_returns['ES_ret'].rolling(10).corr(df_returns['VIX_ret'])
            df['es_vix_corr'] = df_returns['es_vix_corr']
            print("✅ ES-VIX correlation added")
        else:
            print("⚠️ VIX data not found for correlation computation")

    # Restrict final DataFrame to rows where all required features are present
    final_features = [
        'log_return', 'rsi', 'macd', 'bb_mavg', 'bb_width', 'atr',
        'rsi_lag1', 'macd_lag1', 'momentum_3', 'momentum_5', 'vol_5',
        'yield_spread_10y_2y', 'fed_funds_rate', 'cpi_inflation', 'vix_level', 'es_vix_corr'
    ]

    # === Drop Rows with Missing Values ===
    print("\n🚨 NaN summary (before drop):")
    print(df[final_features].isnull().sum().sort_values(ascending=False))

    # Preserve latest row before drop
    latest_date = df.index.max()
    latest_row = df.loc[[latest_date]]

    df = df.dropna(subset=final_features)

    # === Patch: Restore latest row if not fully NaN ===
    if latest_date not in df.index and not latest_row[final_features].isnull().all(axis=1).any():
        print(f"🩹 Re-adding {latest_date.date()} row for live prediction")
        df = pd.concat([df, latest_row])

    print(f"✅ Final shape after drop/fill: {df.shape}")
    print(f"📅 Final index range: {df.index.min().date()} → {df.index.max().date()}")
    return df
