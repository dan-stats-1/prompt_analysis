import polars as pl
import polars.testing
import pytest

from transform_data import word_sentiment_counter, transform_data

@pytest.fixture
def dummy_prompts():
    """
    Generates a list of dummy prompts
    """
    return [
        "dan is a super cool guy",
        "dan is not cool, in fact he's rather dull"
    ]    


## ###################### ##
## word_sentiment_counter ##
## ###################### ##
@pytest.mark.parametrize("prompt,expected_neg_count", [
    ("dan is a super cool guy", 0),
    ("dan is not cool, in fact he's rather dull", 1)
])
def test_word_sentiment_counter(prompt, expected_neg_count):
    """
    Test word_sentiment_counter returns expected negative counts
    """
    count = word_sentiment_counter(prompt)

    assert count == expected_neg_count


## ############## ##
## transform_data ##
## ############## ##
@pytest.mark.parametrize("df,label,expected_neg_count", [
    (pl.DataFrame({"prompt": ["dan is a super cool guy"]}), 0, 0),
    (pl.DataFrame({"prompt": ["dan is not cool, in fact he's rather dull"]}), 1, 1)
])
def test_transform_data(df, label, expected_neg_count):
    """
    Test that a dataframe containing prompts
    """
    transformed_data = transform_data(df, label)
    expected_data = df.with_columns(
        pl.lit(label).cast(pl.Int16).alias("label"),
        pl.lit(expected_neg_count).cast(pl.Int16).alias("num_negative_words")
    )

    polars.testing.assert_frame_equal(transformed_data, expected_data)
