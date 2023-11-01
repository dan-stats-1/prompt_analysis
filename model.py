from typing import List

import polars as pl
import scipy.stats as stats
import statsmodels.api as sm


TRAIN_CSV = "data/prompts_with_sentiment.csv"

class JailbreakPromptDetector:

    def __init__(self, num_negative_words: List[int]) -> None:
        self.benign_model = self.train(num_negative_words)
        self.threshold = self.find_threshold()

    def train(self, num_negative_words: List[int]) -> stats._discrete_distns.nbinom_gen:
        mod = sm.NegativeBinomial(
            num_negative_words,
            [1 for _ in range(len(num_negative_words))]
        ).fit()

        return mod.get_distribution()
    
    def find_threshold(self) -> float:
        pass

df = pl.read_csv(TRAIN_CSV)

mod = train(df.select("num_negative_words").to_series().to_list())


