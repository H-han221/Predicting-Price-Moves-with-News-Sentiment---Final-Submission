import pandas as pd
from textblob import TextBlob

def compute_sentiment(df, column="headline"):
    """
    Adds a sentiment column to a DataFrame based on a text column.
    Sentiment polarity: -1 (negative) to +1 (positive)
    """
    df["sentiment"] = df[column].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return df

def daily_average_sentiment(df, date_col="date"):
    """
    Aggregates sentiment per day.
    """
    df["date_only"] = pd.to_datetime(df[date_col]).dt.date
    daily_sentiment = df.groupby("date_only")["sentiment"].mean().reset_index()
    return daily_sentiment
