import polars as pl
from model import JailbreakDetector


TRAIN = "data/prompts_with_sentiment.csv"
BLOCKING_FPS = 0
ANALYST_FPS = 100

train = pl.read_csv(TRAIN)

detector = JailbreakDetector()

detector.train(train.select("prompt").to_series().to_list())

train = train.with_columns(
    pl.col("prompt").map_elements(detector.score).alias("p_value")
)

blocking_threshold = (
    train.filter(pl.col("label") == 0)
    .select("p_value")
    .sort("p_value", descending=True)
    .row(BLOCKING_FPS)
)

analyst_threshold = (
    train.filter(pl.col("label") == 0)
    .select("p_value")
    .sort("p_value", descending=True)
    .row(ANALYST_FPS)
)

detector.update_thresholds(block_thresh=blocking_threshold, analyst_thresh=analyst_threshold)

detector.save_model("trained_model.pickle")
