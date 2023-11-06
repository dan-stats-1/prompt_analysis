import pickle
from pathlib import Path
from typing import List, Optional, Union

import scipy.stats as stats
import statsmodels.api as sm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import download

class JailbreakCountModel:

    def __init__(self) -> None:
        self.model: stats._discrete_distns.nbinom_gen

    def train(self, num_negative_words: List[int]) -> None:
        mod = sm.NegativeBinomial(
            num_negative_words,
            [1 for _ in range(len(num_negative_words))]
        ).fit()

        self.model = mod.get_distribution()

    def score(self, num_negative_words: int) -> float:
        return self.model.cdf(num_negative_words)[0]


class JailbreakSentimentModel:

    def __init__(self) -> None:
        download("vader_lexicon")
        self.model = SentimentIntensityAnalyzer()
    
    def word_sentiment_counter(self, prompt: str) -> int:
        num_negative_words = 0
        for word in prompt.split(" "):
            scores = self.model.polarity_scores(word)
            if scores["neg"] >= 0.34:
                num_negative_words += 1
        
        return num_negative_words

class JailbreakDetector:

    def __init__(self, block_thresh: float = 0.95, analyst_thresh: float = 0.7) -> None:
        self.count_model = JailbreakCountModel()
        self.sentiment_model = JailbreakSentimentModel()

        self.block_thresh = block_thresh
        self.analyst_thresh = analyst_thresh

    def train(self, prompts: List[str]) -> None:
        num_negative_words = [
            self.sentiment_model.word_sentiment_counter(prompt) for prompt in prompts
        ]
        self.count_model.train(num_negative_words)
    
    def score(self, prompt: str) -> float:
        num_negative_words = self.sentiment_model.word_sentiment_counter(prompt)
        return self.count_model.score(num_negative_words)
    
    def classify(self, prompt: str) -> str:
        score = self.score(prompt)
        if score >= self.block_thresh:
            return "Prompt blocked"
        elif score < self.analyst_thresh:
            return "Prompt allowed"
        else:
            return "Prompt suspicious, account passed to analysts"
 
    def update_thresholds(self, block_thresh: Optional[float] = None, analyst_thresh: Optional[float] = None) -> None:
        if block_thresh is not None:
            self.block_thresh = block_thresh
        
        if analyst_thresh is not None:
            self.analyst_thresh = analyst_thresh
    
    def save_model(self, save_path: Union[str, Path]) -> None:
        conf = {
            "count_model": self.count_model,
            "sentiment_model": self.sentiment_model,
            "block_thresh": self.block_thresh,
            "analyst_thresh": self.analyst_thresh
        }

        with open(save_path, "wb") as f:
            pickle.dump(conf, f)
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        with open(model_path, "rb") as f:
            conf = pickle.load(f)
        
        self.count_model = conf["count_model"]
        self.sentiment_model = conf["sentiment_model"]
        self.block_thresh = conf["block_thresh"]
        self.analyst_thresh = conf["analyst_thresh"]
