import polars as pl

from nltk import download
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def word_sentiment_counter(prompt: str) -> int:
    num_negative_words = 0
    for word in prompt.split(" "):
        scores = senitment_analyser.polarity_scores(word)
        if scores["neg"] >= 0.34:
            num_negative_words += 1
    
    return num_negative_words


def transform_data(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.lit(0).alias("label"),
        pl.col("prompt").map_elements(word_sentiment_counter).alias("num_negative_words")
    )

    return df


download('vader_lexicon')
senitment_analyser = SentimentIntensityAnalyzer()

chatgpt = pl.read_csv("data/chatgpt_prompts.csv", columns=["prompt"])
jailbreak = pl.read_csv("data/jailbreak_prompts.csv", columns=["prompt"])

chatgpt = transform_data(chatgpt)
jailbreak = transform_data(jailbreak)

all_prompts = pl.concat([chatgpt, jailbreak])

all_prompts.write_csv("data/prompts_with_sentiment.csv", separator=",")
