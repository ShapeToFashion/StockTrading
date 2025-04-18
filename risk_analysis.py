# utils/features.py
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

smaUsed = [50, 200]
emaUsed = [21]
lookback_period = 30
prediction_period = 20

def calculate_features(ticker, include_target=True):
    try:
        df = yf.download(ticker, start="2022-01-01", auto_adjust=True)
        if df.empty: return None
        close = df['Close']

        df['Returns'] = close.pct_change()
        df['Return_Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)

        for x in smaUsed:
            sma = close.rolling(window=x).mean()
            df[f"SMA_{x}_Dist"] = (close - sma) / sma * 100

        for x in emaUsed:
            ema = close.ewm(span=x, adjust=False).mean()
            df[f"EMA_{x}_Dist"] = (close - ema) / ema * 100

        df['20_MA'] = close.rolling(window=20).mean()
        df['20_SD'] = close.rolling(window=20).std()
        df['Upper_Band'] = df['20_MA'] + 2 * df['20_SD']
        df['Lower_Band'] = df['20_MA'] - 2 * df['20_SD']
        df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['20_MA']

        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df['MACD'] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']

        if include_target:
            df['Future_Return'] = close.pct_change(periods=prediction_period).shift(-prediction_period)
            df['Risk_Label'] = 1
            df.loc[df['Future_Return'] < -0.05, 'Risk_Label'] = 3
            df.loc[(df['Future_Return'] < -0.02) & (df['Future_Return'] >= -0.05), 'Risk_Label'] = 2

        df['Max_Drawdown'] = close.rolling(window=lookback_period).apply(lambda x: (x[-1] - x.min()) / x.min() * 100)
        df['Sharpe_Ratio'] = df['Returns'].rolling(window=lookback_period).mean() / df['Returns'].rolling(window=lookback_period).std() * np.sqrt(252)

        df = df.dropna()
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def predict_risk(ticker, model):
    df = calculate_features(ticker, include_target=False)
    if df is None: return None

    features = [
        'Return_Volatility', 'SMA_50_Dist', 'SMA_200_Dist', 'EMA_21_Dist',
        'BB_Width', 'RSI', 'MACD', 'MACD_Hist', 'Max_Drawdown', 'Sharpe_Ratio'
    ]
    X = df[features].fillna(0)
    df['Predicted_Risk'] = model.predict(X)
    return df[['Close', 'Predicted_Risk']].tail(prediction_period)
