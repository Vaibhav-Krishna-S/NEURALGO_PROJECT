import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Tuple, Any
import joblib
import numpy as np
import csv
from collections import Counter

# Download required NLTK data
print("Downloading required NLTK data...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')  # Ensure punkt_tab is available
print("NLTK data download completed.")

def preprocess_text(text: str) -> str:
    """
    Advanced text preprocessing with focus on sentiment capture.
    """
    try:
        # Initialize lemmatizer
        lemmatizer = WordNetLemmatizer()
        # Convert to lowercase
        text = text.lower()
        # Replace specific patterns with stronger sentiment markers
        replacements = {
            r'\b(lost|missing)\s+(?:my |the |our )?(?:bag|baggage|luggage)\b': 'LOST_BAGGAGE_NEGATIVE',
            r'\b(rude|unhelpful|unfriendly)\b': 'VERY_NEGATIVE_SERVICE',
            r'\b(terrible|horrible|awful|worst)\b': 'EXTREMELY_NEGATIVE',
            r'\b(great|excellent|amazing|perfect)\b': 'VERY_POSITIVE',
            r'\b(delayed|late|delay)\b': 'DELAY_NEGATIVE',
            r'\b(never|not)\s+(?:fly|flying|use|using|recommend)\b': 'STRONG_REJECTION',
            r'!+': ' EXCLAMATION ',
            r'\?+': ' QUESTION '
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        # Remove special characters but keep our markers
        text = re.sub(r'[^a-zA-Z_\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords but keep important negative words
        important_words = {'no', 'not', 'never', 'but', 'very', 'too', 'only', 'down', 'again'}
        stop_words = set(stopwords.words('english')) - important_words
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        # Handle negations
        negation_words = {'no', 'not', 'never', 'nothing'}
        modified_tokens = []
        negate = False
        for token in tokens:
            if token in negation_words:
                negate = True
                modified_tokens.append('NEG')
            elif token in {'.', '!', '?', 'but'}:
                negate = False
                modified_tokens.append(token)
            else:
                modified_tokens.append(f'NEG_{token}' if negate else token)
        return ' '.join(modified_tokens)
    except Exception as e:
        print(f"Error processing text: {text}. Details: {e}")
        return ""

def extract_features(text: str) -> str:
    """
    Extract features with emphasis on sentiment detection.
    """
    features = []
    # Add preprocessed text
    processed_text = preprocess_text(text)
    features.append(processed_text)
    # Count important patterns
    lower_text = text.lower()
    # Strong negative indicators
    if any(word in lower_text for word in ['worst', 'terrible', 'horrible', 'awful']):
        features.extend(['STRONG_NEGATIVE'] * 3)  # Give more weight
    # Lost baggage is a strong negative
    if re.search(r'lost.*bag|bag.*lost|missing.*bag', lower_text):
        features.extend(['BAGGAGE_ISSUE'] * 3)
    # Service issues
    if re.search(r'rude|unhelpful|unfriendly', lower_text):
        features.extend(['BAD_SERVICE'] * 2)
    # Delay issues
    if re.search(r'delay|late|wait', lower_text):
        features.append('DELAY_ISSUE')
    # Strong rejection
    if re.search(r'never|not\s+(?:fly|use|recommend)', lower_text):
        features.extend(['CUSTOMER_REJECTION'] * 3)
    # Multiple exclamation marks often indicate strong sentiment
    if '!!' in text:
        features.append('STRONG_EMOTION')
    return ' '.join(features)

def train_sentiment_model(training_data: List[Tuple[str, str]]) -> Tuple[Any, float, str, str, List[float]]:
    """
    Train sentiment model with focus on sentiment accuracy.
    """
    # Extract texts and labels
    texts = [text for text, _ in training_data]
    labels = [label for _, label in training_data]

    # Check label distribution
    label_counts = Counter(labels)
    print("Label distribution:", label_counts)

    # Ensure at least 2 samples per class
    if any(count < 2 for count in label_counts.values()):
        raise ValueError("Each class must have at least 2 samples. Please balance the dataset.")

    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    # Create pipeline with enhanced TF-IDF and MultinomialNB
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            preprocessor=extract_features,
            max_features=5000,
            ngram_range=(1, 3),
            min_df=1,
            use_idf=True,
            sublinear_tf=True
        )),
        ('classifier', MultinomialNB(
            alpha=0.5  # Slightly stronger smoothing
        ))
    ])
    # Perform cross-validation
    cv_scores = cross_val_score(model, texts, labels, cv=3)  # Reduced folds to 3
    # Train the model
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    # Format confusion matrix
    conf_matrix_str = "\nConfusion Matrix:\n"
    conf_matrix_str += "                  Predicted Negative  Predicted Positive\n"
    conf_matrix_str += f"Actual Negative       {conf_matrix[0][0]}                {conf_matrix[0][1]}\n"
    conf_matrix_str += f"Actual Positive       {conf_matrix[1][0]}                {conf_matrix[1][1]}"
    return model, accuracy, conf_matrix_str, class_report, cv_scores

def predict_sentiment(model: Any, new_text: str) -> Tuple[str, float]:
    """
    Predict sentiment with confidence score.
    """
    prediction = model.predict([new_text])[0]
    proba = model.predict_proba([new_text])[0]
    confidence = max(proba)
    return prediction, confidence

if __name__ == "__main__":
    # Load the simplified dataset
    print("\nLoading and preparing training data...")
    training_data = []
    with open("submission.zip\Problem_2\AirlineReviews.csv", "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            try:
                if len(row) < 2:
                    raise ValueError(f"Malformed row: {row}")
                text, label = row[0], row[1]
                training_data.append((text, label))
            except Exception as e:
                print(f"Error processing row: {row}. Details: {e}")

    # Train model and get metrics
    print("\nTraining and evaluating the model...")
    model, accuracy, conf_matrix, class_report, cv_scores = train_sentiment_model(training_data)
    
    # Print metrics
    print(f"\nModel Accuracy: {accuracy:.2%}")
    print(f"Cross-validation scores: {[f'{score:.2%}' for score in cv_scores]}")
    print(f"Average CV Accuracy: {np.mean(cv_scores):.2%}")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    # Save the trained model
    joblib.dump(model, 'sentiment_model.pkl')
    print("\nTrained model saved as 'sentiment_model.pkl'.")

    # Test cases focusing on clear sentiment
    print("\nTesting the model with sample texts:")
    test_texts = [
        "The seats were comfortable and service was great!",
        "They lost my baggage and were very unhelpful!",
        "Nothing special, just an average flight.",
        "The crew was rude and the flight was delayed.",
        "Amazing service and on-time departure!",
        "The plane was clean and the journey was smooth.",
        "Worst experience ever, never flying with them again!",
        "Perfect flight with exceptional service.",
        "Delayed flight but staff handled it well.",
        "No legroom and terrible food service."
    ]
    # Make predictions
    print("\nPredictions on new texts:")
    for text in test_texts:
        sentiment, confidence = predict_sentiment(model, text)
        print(f"\nText: {text}")
        print(f"Predicted sentiment: {sentiment} (Confidence: {confidence:.2%})")