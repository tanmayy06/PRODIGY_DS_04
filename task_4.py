# task_4.py

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Download VADER lexicon if not already present
nltk.download('vader_lexicon')

# 1. Load dataset
df = pd.read_csv(r'T:\task 4\twitter_training.csv', header=None)
df.columns = ["ID", "Entity", "OriginalSentiment", "TweetText"]

# 2. Preprocess text
def clean_text(text):
    if not isinstance(text, str):  # Handle NaN or non-string values
        text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#[A-Za-z0-9_]+|[^a-z\s]", "", text)
    return text

df['TweetText'] = df['TweetText'].fillna("")  # Replace NaN with empty string
df['clean_text'] = df['TweetText'].apply(clean_text)
df['length'] = df['clean_text'].str.split().apply(len)

# 3. Sentiment scoring
sid = SentimentIntensityAnalyzer()
df['compound'] = df['clean_text'].apply(lambda x: sid.polarity_scores(x)['compound'])
df['sentiment'] = df['compound'].apply(
    lambda c: 'positive' if c > 0.05 else ('negative' if c < -0.05 else 'neutral')
)

# 4. Overall Sentiment Distribution (static)
sns.countplot(x='sentiment', data=df, order=['positive', 'neutral', 'negative'])
plt.title('Overall Sentiment Distribution')
plt.show()

# 5. Entity-Level Sentiment Distribution (interactive)
entity_sentiment_counts = df.groupby(['Entity', 'sentiment']).size().reset_index(name='count')
fig = px.bar(
    entity_sentiment_counts,
    x='Entity',
    y='count',
    color='sentiment',
    title='Sentiment by Entity',
    barmode='group'
)
fig.show()

# 6. Compound Score Distribution by Entity (interactive)
fig2 = px.box(
    df,
    x='Entity',
    y='compound',
    color='sentiment',
    title='Compound Score Distribution per Entity'
)
fig2.show()

# 7. Word Clouds for Positive and Negative Tweets
for s in ['positive', 'negative']:
    text = " ".join(df[df['sentiment'] == s]['clean_text'])
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud - {s.capitalize()} Tweets")
    plt.show()

# 8. TF-IDF Weighted Word Cloud (All Tweets)
tfidf = TfidfVectorizer(stop_words='english', max_features=200)
tfidf_matrix = tfidf.fit_transform(df['clean_text'])
word_scores = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
wc_tfidf = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_scores)
plt.imshow(wc_tfidf, interpolation='bilinear')
plt.axis('off')
plt.title("TF-IDF Weighted Word Cloud")
plt.show()
