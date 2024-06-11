import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
import matplotlib.pyplot as plt

# Download resources (run this only once)
nltk.download('punkt')  # Download tokenizer for sentence splitting
nltk.download('vader_lexicon')  # Download sentiment lexicon

# Sentiment Analyzer (global variable)
analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text):
  """Analyzes sentiment of the text and returns sentiment scores and label."""
  sentiment = analyzer.polarity_scores(text)
  compound = sentiment['compound']

  if compound > 0.05:
    label = "Positive"
  elif compound < -0.05:
    label = "Negative"
  else:
    label = "Neutral"

  return sentiment, label


def main():
  """Streamlit app for sentiment analysis"""

  # Title (text variable)
  st.title("Sentiment Analyzer")

  # Text area for user input (text variable)
  text = st.text_area("Enter your review here:")

  # Analyze Button (button variable)
  if st.button("Analyze Sentiment"):
    if text:
      sentiment, label = analyze_sentiment(text)

      # Sentiment Scores (dictionary variable)
      st.subheader("Sentiment Scores:")

      # Explain sentiment scores (comments)
      st.write("These scores indicate the proportion of the text that falls into each sentiment category:")
      st.write("- **Positive (pos):** The proportion of positive words and phrases.")
      st.write("- **Neutral (neu):** The proportion of words that express no sentiment.")
      st.write("- **Negative (neg):** The proportion of negative words and phrases.")
      st.write(sentiment)  # Display sentiment scores

      # Explain compound score (comment)
      st.write("**Compound Score:** This score combines the positive, neutral, and negative scores to provide a single value between -1 (most negative) and +1 (most positive). It reflects the overall sentiment of the text.")

      # Sentiment Label (text variable)
      st.subheader(f"Overall Sentiment: {label} ")

      # Positive, Neutral, Negative scores (numerical variables)
      positive = sentiment['pos']
      neutral = sentiment['neu']
      negative = sentiment['neg']

      # Figure and axis (plot objects)
      fig, ax = plt.subplots()
      ax.bar(['Positive', 'Neutral', 'Negative'], [positive, neutral, negative])
      ax.set_xlabel('Sentiment')
      ax.set_ylabel('Score')
      ax.set_title('Sentiment Distribution')
      st.pyplot(fig)
    else:
      st.warning("Please enter some text to analyze!")


if __name__ == "__main__":
  main()
