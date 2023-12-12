from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Sample data for training the classifier
jokes_data = [
    (
        "Knock, knock. Who's there? Alpaca. Alpaca who? Alpaca the suitcase, you load up the car!",
        "knock-knock",
    ),
    (
        "Knock, knock. Who's there? Lettuce. Lettuce who? Lettuce in, it's cold out here!",
        "knock-knock",
    ),
    (
        "Knock, knock. Who's there? Cow says. Cow says who? No silly, cow says 'moo'!",
        "knock-knock",
    ),
    ("Knock, knock. Who's there? Atch. Atch who? Bless you!", "knock-knock"),
    (
        "Knock, knock. Who's there? Orange. Orange who? Orange you glad I didn't say banana?",
        "knock-knock",
    ),
    (
        "Knock, knock. Who's there? Boo. Boo who? Don't cry, it's just a joke!",
        "knock-knock",
    ),
    (
        "Knock, knock. Who's there? Interrupting cow. Interrupting cow wh—MOO!",
        "knock-knock",
    ),
    (
        "Knock, knock. Who's there? Honeydew. Honeydew who? Honeydew you know how funny these jokes are?",
        "knock-knock",
    ),
    (
        "Knock, knock. Who's there? Olive. Olive who? Olive your jokes are bad!",
        "knock-knock",
    ),
    (
        "Knock, knock. Who's there? Justin. Justin who? Just in time for dinner!",
        "knock-knock",
    ),
    ("Knock, knock. Who's there? Tank. Tank who? You're welcome!", "knock-knock"),
    (
        "Knock, knock. Who's there? Lettuce. Lettuce who? Lettuce in, it's freezing out here!",
        "knock-knock",
    ),
    (
        "Knock, knock. Who's there? Howard. Howard who? Howard you like to be wrapped in a big warm blanket right now?",
        "knock-knock",
    ),
    (
        "Knock, knock. Who's there? Noah. Noah who? Noah good place we can grab a bite to eat?",
        "knock-knock",
    ),
    (
        "Knock, knock. Who's there? Justin. Justin who? Just in time for a great joke!",
        "knock-knock",
    ),
    (
        "Knock, knock. Who's there? Ice cream. Ice cream who? Ice cream every time I see a scary movie!",
        "knock-knock",
    ),
    (
        "I used to play piano by ear, but now I use my hands and fingers.",
        "observational/situational",
    ),
    ("Why did the bicycle fall over? Because it was two-tired!", "wordplay"),
    ("I'm on a whiskey diet. I've lost three days already.", "wordplay"),
    (
        "Why don't scientists trust atoms? Because they make up everything!",
        "observational/situational",
    ),
    (
        "I told my wife she was drawing her eyebrows too high. She looked surprised.",
        "observational/situational",
    ),
    (
        "Why did the scarecrow win an award? Because he was outstanding in his field.",
        "wordplay",
    ),
    (
        "I only know 25 letters of the alphabet. I don't know y.",
        "observational/situational",
    ),
    ("I used to be a baker because I kneaded dough.", "wordplay"),
    (
        "I told my computer I needed a break, and now it won't stop sending me vacation ads.",
        "observational/situational",
    ),
    ("Why don't skeletons fight each other? They don't have the guts.", "wordplay"),
    ("The math teacher told me I'm average. How mean!", "wordplay"),
    ("I'm reading a book on anti-gravity. It's impossible to put down!", "wordplay"),
    (
        "Why do we press harder on the remote control when we know the batteries are weak?",
        "observational/situational",
    ),
    (
        "Why do they call it 'fast food' when you have to wait in line for it?",
        "observational/situational",
    ),
    (
        "I asked the librarian if the library had any books on paranoia. She whispered, 'They're right behind you.'",
        "observational/situational",
    ),
    ("Why do we park on driveways and drive on parkways?", "observational/situational"),
    (
        "I asked the waiter if the restaurant has free Wi-Fi. He said, 'Yes, but it's not on the menu.'",
        "observational/situational",
    ),
    (
        "Why do they call it 'chill mode' on the air conditioner remote? Shouldn't it be 'cool mode'?",
        "observational/situational",
    ),
]

# Split data into training and testing sets
jokes, labels = zip(*jokes_data)
X_train, X_test, y_train, y_test = train_test_split(
    jokes, labels, test_size=0.20, random_state=42
)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
