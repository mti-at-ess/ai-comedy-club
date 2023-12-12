from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
from data_process import X_train_vectorized, X_test_vectorized, y_train, y_test

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Predict on the test set
predictions = classifier.predict(X_test_vectorized)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)

with open("joke_classifier.pkl", "wb") as model_file:
    pickle.dump(classifier, model_file)
