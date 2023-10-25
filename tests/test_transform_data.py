import pytest

from transform_data import word_sentiment_counter

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
    ("dan is not cool, in fact he's rather dull",1)
])
def test_word_sentiment_counter(prompt, expected_neg_count):
    """
    Test word_sentiment_counter returns expected negative counts
    """
    count = word_sentiment_counter(prompt)

    assert count == expected_neg_count
    