import pandas as pd
from src.utils import compute_sentiment, daily_average_sentiment

def test_compute_sentiment():
    df = pd.DataFrame({"headline": ["Good news today", "Bad earnings report"]})
    df = compute_sentiment(df)
    assert "sentiment" in df.columns
    assert df["sentiment"].iloc[0] > df["sentiment"].iloc[1]

def test_daily_average_sentiment():
    df = pd.DataFrame({
        "headline": ["Good news", "Bad news"],
        "date": ["2025-11-25", "2025-11-25"]
    })
    df = compute_sentiment(df)
    daily = daily_average_sentiment(df)
    assert len(daily) == 1
    assert "sentiment" in daily.columns

