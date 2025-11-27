# scripts/task3_sentiment_correlation.py

"""
Task 3: Correlation between news sentiment and stock movements
Python 3.13.3
"""

# -----------------------------
# Standard imports
# -----------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Step 3.2: Compute daily sentiment
# -----------------------------
# Load cleaned news
news_path = 'output/news_cleaned.csv'
if not os.path.exists(news_path):
    raise FileNotFoundError(f"{news_path} not found. Make sure you run Task 1 and Task 2 first.")

df = pd.read_csv(news_path)
df['date'] = pd.to_datetime(df['date'], utc=True)
df['headline'] = df['headline'].astype(str)

# Load sentiment function
from src.features.sentiment import vader_score

# Compute sentiment for each headline
print("Computing sentiment for headlines...")
df['sentiment'] = df['headline'].apply(vader_score)

# Aggregate daily sentiment per stock
df['date_day'] = df['date'].dt.date
daily_sent = df.groupby(['stock', 'date_day'])['sentiment'].mean().reset_index()
daily_sent.rename(columns={'sentiment': 'sent_mean'}, inplace=True)

# Save daily sentiment
os.makedirs('output', exist_ok=True)
daily_sent.to_csv('output/daily_sent.csv', index=False)
print("Saved daily sentiment -> output/daily_sent.csv")
print(daily_sent.head())

# -----------------------------
# Step 3.3: Merge with stock prices and compute correlation
# -----------------------------
import yfinance as yf

# Example: pick a ticker (or loop over multiple tickers later)
ticker = 'AAPL'
print(f"\nAnalyzing ticker: {ticker}")

# Download historical stock prices
p = yf.download(ticker, start='2020-01-01', end='2025-11-25', progress=False)
if 'Adj Close' not in p.columns and 'Close' in p.columns:
    p['Adj Close'] = p['Close']

# Compute daily returns
p['Return'] = p['Adj Close'].pct_change()
p['date_day'] = p.index.date

# Merge daily sentiment with stock prices
merged = pd.merge(
    daily_sent[daily_sent['stock'] == ticker],
    p,
    how='left',
    on='date_day'
)

# Compute next-day return
merged['Return_next'] = merged['Return'].shift(-1)

# Drop NaNs for correlation
m = merged.dropna(subset=['sent_mean', 'Return_next'])

# Compute correlation
corr = m['sent_mean'].corr(m['Return_next'])
print(f"Correlation between sentiment and next-day return for {ticker}: {corr:.4f}")

# Scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(m['sent_mean'], m['Return_next'], alpha=0.5)
plt.axhline(0, color='k', linewidth=0.5)
plt.xlabel('Daily Sentiment (Mean)')
plt.ylabel('Next-Day Return')
plt.title(f'Sentiment vs Next-Day Return ({ticker})')
plt.show()

# -----------------------------
# Optional: Compute correlation for all tickers
# -----------------------------
tickers = daily_sent['stock'].unique()
correlations = {}

for t in tickers:
    p = yf.download(t, start='2020-01-01', end='2025-11-25', progress=False)
    if 'Adj Close' not in p.columns and 'Close' in p.columns:
        p['Adj Close'] = p['Close']
    p['Return'] = p['Adj Close'].pct_change()
    p['date_day'] = p.index.date
    merged = pd.merge(daily_sent[daily_sent['stock'] == t], p, how='left', on='date_day')
    merged['Return_next'] = merged['Return'].shift(-1)
    m = merged.dropna(subset=['sent_mean', 'Return_next'])
    if not m.empty:
        correlations[t] = m['sent_mean'].corr(m['Return_next'])

print("\nCorrelation for all tickers:")
for k, v in correlations.items():
    print(f"{k}: {v:.4f}")
