import requests
import pandas as pd
from datetime import datetime

def fetch_mes_data(access_token: str, symbol: str = "MESM5", interval: str = "1min", lookback_days: int = 5) -> pd.DataFrame:
    """
    Fetches historical MES futures data from Schwab Trader API.
    
    Args:
        access_token: OAuth2 token.
        symbol: Schwab-compatible symbol for MES.
        interval: Interval like '1min', '5min', '1day'.
        lookback_days: How many days back to fetch.

    Returns:
        DataFrame with OHLCV data.
    """
    url = f"https://api.schwabapi.com/marketdata/v1/pricehistory"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "symbol": symbol,
        "frequencyType": "minute",  # or 'daily'
        "frequency": 1,             # every 1 minute
        "periodType": "day",
        "period": lookback_days,
        "needExtendedHoursData": "false"
    }

    print(f"📡 Requesting data for: {symbol}")
    response = requests.get(url, headers=headers, params=params)
    print(response.status_code)

    if response.status_code != 200:
        raise Exception(f"Schwab API error: {response.status_code} {response.text}")

    try:
        data = response.json()
    except Exception as e:
        print("❌ Failed to parse JSON:", e)
        print("Response content:", response.text[:300])
        return None

    if "candles" not in data or not isinstance(data["candles"], list) or len(data["candles"]) == 0:
        print("⚠️ No candles found in data.")
        print("Raw data keys:", data.keys())
        return None

    print(f"✅ Received {len(data['candles'])} candles")
    print("🕰 Sample candle:", data["candles"][0])

    df = pd.DataFrame(data["candles"])

    if "datetime" not in df.columns:
        print("❌ 'datetime' field not found in candles")
        print("🧾 DataFrame columns:", df.columns.tolist())
        return None

    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
    df.set_index("datetime", inplace=True)

    return df[["open", "high", "low", "close", "volume"]].rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
    })
