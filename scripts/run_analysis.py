from src.utils import compute_sentiment, daily_average_sentiment
import pandas as pd

# Load news data
df_news = pd.read_csv("data/news.csv")

# Compute sentiment
df_news = compute_sentiment(df_news)

# Compute daily average sentiment
daily_sentiment = daily_average_sentiment(df_news)

# Save outputs
df_news.to_csv("outputs/news_with_sentiment.csv", index=False)
daily_sentiment.to_csv("outputs/daily_sentiment.csv", index=False)

print("Sentiment analysis completed. Outputs saved in 'outputs/' folder.")
