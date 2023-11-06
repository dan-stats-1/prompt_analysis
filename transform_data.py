import polars as pl

from model import JailbreakSentimentModel


def transform_data(df: pl.DataFrame, label: int) -> pl.DataFrame:
    df = df.with_columns(
        pl.lit(label).cast(pl.Int16).alias("label"),
        pl.col("prompt").map_elements(senitment_analyser.word_sentiment_counter).cast(pl.Int16).alias("num_negative_words")
    )

    return df

if __name__ == "__main__":
    senitment_analyser = JailbreakSentimentModel()

    chatgpt = pl.read_csv("data/chatgpt_prompts.csv", columns=["prompt"])
    jailbreak = pl.read_csv("data/jailbreak_prompts.csv", columns=["prompt"])

    chatgpt = transform_data(chatgpt, 0)
    jailbreak = transform_data(jailbreak, 1)

    all_prompts = pl.concat([chatgpt, jailbreak])

    all_prompts.write_csv("data/prompts_with_sentiment.csv", separator=",")
