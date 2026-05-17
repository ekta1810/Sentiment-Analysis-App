import pandas as pd
import re
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset.csv")

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Apply cleaning
df['clean_review'] = df['review'].apply(clean_text)

# Features and labels
X = df['clean_review']
y = df['sentiment']

# Train-test split
# Train-test split (IMPORTANT FIX)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y   # 🔥 IMPORTANT FIX
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Prediction
y_pred = model.predict(X_test_vec)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

y_pred = model.predict(X_test_vec)

cm = confusion_matrix(y_test, y_pred)

# Save confusion matrix
np.save("confusion_matrix.npy", cm)

print("Confusion matrix saved 🚀")

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved successfully 🚀")