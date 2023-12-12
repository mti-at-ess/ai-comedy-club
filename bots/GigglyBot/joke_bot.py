import pickle
import math
import random
from textblob import TextBlob
from data_process import vectorizer, jokes_data


class Bot:
    joke_string = None
    jokes_data = jokes_data

    def tell_joke(self) -> str:
        return random.choice(self.jokes_data)[0]

    def rate_joke(self, joke: str) -> float:
        self.joke_string = joke
        humor_rate = self.rate_humour_creativity()
        tone_rate = self.rate_joke_tone()
        engagement_rate = self.rate_engagement_delivery()
        return (tone_rate + engagement_rate + humor_rate) / 3

    def rate_humour_creativity(self) -> float:
        humor_scores = {
            "observational/situational": 7,
            "wordplay": 7,
            "knock-knock": 6,
        }
        creativity_scores = {
            "observational/situational": 7,
            "wordplay": 8,
            "knock-knock": 6,
        }
        # Load the trained classifier
        with open("joke_classifier.pkl", "rb") as model_file:
            loaded_classifier = pickle.load(model_file)

        # Example usage to classify a new joke
        new_joke_vectorized = vectorizer.transform([self.joke_string])
        prediction = loaded_classifier.predict(new_joke_vectorized)

        return math.ceil(
            (humor_scores[prediction[0]] + creativity_scores[prediction[0]]) / 2
        )

    def rate_joke_tone(self) -> float:
        # Analyze sentiment using TextBlob
        analysis = TextBlob(self.joke_string)

        # Classify the tone based on sentiment polarity
        tone_score = max(2, min(10, 6 + (analysis.sentiment.polarity * 10))) % len(
            self.joke_string
        )

        # Assess appropriateness based on sentiment polarity
        appropriate_score = max(
            2, min(10, 6 + (analysis.sentiment.polarity * 5))
        ) % len(self.joke_string)
        return math.ceil((appropriate_score + tone_score) / 2)

    def rate_engagement_delivery(self) -> float:
        # Simulate user engagement assessment based on joke length
        length_engagement = min(
            10, max(2, round(5 + 0.1 * (len(self.joke_string) - 50)))
        )
        print(f"length_engagement: {length_engagement}")

        count = sum(1 for char in self.joke_string if char in [","])
        count += sum(2 for char in self.joke_string if char in ["?"])
        punctuation_engagement = min(1.5 * count, 10)
        print(f"punctuation_engagement: {punctuation_engagement}")

        delivery_score = max(
            7,
            min(10, 5 + 3 * ("!" in self.joke_string) - 2 * ("." in self.joke_string)),
        )
        print(f"delivery_score: {delivery_score}")

        return math.ceil(
            (punctuation_engagement + 2 * length_engagement + delivery_score) / 4
        )


if __name__ == "__main__":
    bot = Bot()

    # Generate and tell a joke
    joke_str = bot.tell_joke()
    print("Joke:", joke_str)

    # Rate the joke
    rating = bot.rate_joke(joke_str)
    print("Rating:", rating)
