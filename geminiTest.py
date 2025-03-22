from google import genai
import os

value = os.environ.get('api_key')
print(value)
client = genai.Client(api_key=value)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain how AI works",
)

print(response.text)

import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
import string

# Download necessary NLTK resources
nltk.download('stopwords')

# Step 1: Prepare the Data
# Sample data (Replace this with your actual dataset)
data = {
    'review': [
        'This professor is amazing! Really helpful and always available.',
        'Worst professor I have ever had! Unorganized and rude.',
        'The class was challenging but rewarding.',
        'This professor doesn\'t know how to teach. Terrible at explaining concepts.',
        'I loved this professor! Makes learning fun and easy.',
        'The worst experience of my life. The professor was mean and gave bad grades.'
    ],
    'label': [0, 1, 0, 1, 0, 1]  # 0 = not biased, 1 = biased
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Text Preprocessing
# Remove punctuation, convert text to lowercase, and remove stopwords
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df['cleaned_review'] = df['review'].apply(preprocess_text)

# Step 3: Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_review'])

# Step 4: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.3, random_state=42)

# Step 5: Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = classifier.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

# Predict on new reviews (example)
new_reviews = [
    'This professor is awful, rude, and unhelpful.',
    'I learned so much from this class. The professor made everything clear and easy.'
]
new_reviews_cleaned = [preprocess_text(review) for review in new_reviews]
X_new = vectorizer.transform(new_reviews_cleaned)
predictions = classifier.predict(X_new)

for review, prediction in zip(new_reviews, predictions):
    print(f'Review: {review}\nPrediction: {"Biased" if prediction == 1 else "Not Biased"}\n')