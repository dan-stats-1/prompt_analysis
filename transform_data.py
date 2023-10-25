import nltk
import polars as pl

from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

chatgpt = pl.read_csv("data/chatgpt_prompts.csv", columns=["prompt"])
jailbreak = pl.read_csv("data/jailbreak_prompts.csv", columns=["prompt"])


