from flask import Flask, render_template, request
import joblib
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from nltk.corpus import stopwords
from nltk import download
import matplotlib.pyplot as plt

# Ensure NLTK stopwords are downloaded
download('stopwords')

app = Flask(__name__)

# Define the function to clean review text
def clean_review(review):
    stop_words = set(stopwords.words('english'))
    return ' '.join(word for word in review.split() if word.lower() not in stop_words)

# Define the path for saving/loading the model
pipeline_path = os.path.join('pipeline.pkl')
confusion_matrix_path = os.path.join('static', 'confusion_matrix.png')

# Check if the model file exists
if not os.path.exists(pipeline_path):
    # Load the dataset
    df = pd.read_csv('IMDB Dataset.csv')
    # Clean the review text
    df['Review'] = df['Review'].apply(clean_review)
    df['Sentiment'].replace({'negative': 0, 'positive': 1}, inplace=True)

    # Prepare the data
    X = df['Review']
    y = df['Sentiment']

    # Create a pipeline that includes the TfidfVectorizer and LogisticRegression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression())
    ])

    # Train the model
    pipeline.fit(X, y)

    # Save the entire pipeline
    joblib.dump(pipeline, pipeline_path)
    print(f"Model and vectorizer saved to '{pipeline_path}'")

    # Generate model information
    X_test = pipeline.named_steps['tfidf'].transform(X)
    y_pred = pipeline.named_steps['clf'].predict(X_test)
    accuracy = accuracy_score(y, y_pred)

    # Compute confusion matrix
    cm = confusion_matrix(y, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    display.plot()
    plt.savefig(confusion_matrix_path)
    plt.close()
    print(f"Confusion Matrix saved to '{confusion_matrix_path}'")

else:
    # Load the saved model
    pipeline = joblib.load(pipeline_path)
    print(f"Model and vectorizer loaded from '{pipeline_path}'")

    # Generate model information
    df = pd.read_csv('IMDB Dataset.csv')
    df['Review'] = df['Review'].apply(clean_review)
    df['Sentiment'].replace({'negative': 0, 'positive': 1}, inplace=True)

    X = df['Review']
    y = df['Sentiment']

    X_test = pipeline.named_steps['tfidf'].transform(X)
    y_pred = pipeline.named_steps['clf'].predict(X_test)
    accuracy = accuracy_score(y, y_pred)

    # Compute confusion matrix
    cm = confusion_matrix(y, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    display.plot()
    plt.savefig(confusion_matrix_path)
    plt.close()
    print(f"Confusion Matrix saved to '{confusion_matrix_path}'")

# Home route
@app.route('/')
def home():
    return render_template('index.html', accuracy=accuracy, confusion_matrix_path=confusion_matrix_path)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form.get('review', '')
        if not review:
            return render_template('index.html', error='Please provide a review.', accuracy=accuracy, confusion_matrix_path=confusion_matrix_path)

        # Clean the review text
        review_cleaned = clean_review(review)

        # Transform the review text and make prediction
        prediction = pipeline.predict([review_cleaned])

        # Determine sentiment
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'

        return render_template('index.html', review=review, sentiment=sentiment, accuracy=accuracy, confusion_matrix_path=confusion_matrix_path)

if __name__ == '__main__':
    app.run(debug=True)
