import pytest
from .joke_bot import Bot


@pytest.fixture
def bot():
    return Bot()


def test_tell_joke(bot):
    joke = bot.tell_joke()

    # Check if the returned joke is a string
    assert isinstance(joke, str)
    # Check if the length of the joke is within the correct range
    assert len(joke) > 50


def test_rate_joke(bot):
    joke = "Why was the computer cold at the office? Because it left its Windows open."
    rating = bot.rate_joke(joke)

    # Check if the returned rating is a number or float
    assert isinstance(rating, (int, float))
    # Check if the rating is within the correct range (1 to 10)
    assert 1 <= rating <= 10
    # Check if the rating matches the expected value (6.0 in this case)
    assert rating == 6.0
