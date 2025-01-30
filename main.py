
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('training.csv')

# Splitting the dataset into features and labels
X = df['text']
y = df['sentiment']

# Convert labels to numerical values (positive: 1, negative: 0, neutral: 2 if applicable)
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing the text data using CountVectorizer
vectorizer = CountVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Create and train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

# Print evaluation metrics
print(f"Test Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Allow user to input a statement for sentiment analysis
print("\nEnter a statement to analyze its sentiment:")
user_input = input("> ")

# Vectorize the user's input
user_input_vec = vectorizer.transform([user_input])

# Predict sentiment for the user's input
user_pred = model.predict(user_input_vec)

# Decode the prediction back to sentiment
sentiment = le.inverse_transform(user_pred)[0]
print(f"The sentiment of the entered statement is: {sentiment}")
import pickle

# Save the model
with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the vectorizer
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

import streamlit as st
import pickle

# Load the model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit App
st.title("Sentiment Analysis App")
st.write("Enter a statement to analyze its sentiment:")

# Text input
user_input = st.text_area("Your text here", "")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.error("Please enter some text to analyze.")
    else:
        # Vectorize the input text
        user_input_vec = vectorizer.transform([user_input])

        # Predict sentiment
        user_pred = model.predict(user_input_vec)
        sentiment = "Positive" if user_pred[0] == 1 else "Negative" if user_pred[0] == 0 else "Neutral"

        # Display the sentiment
        st.success(f"The sentiment of the entered statement is: **{sentiment}**")

import streamlit as st
import pickle

st.write("Starting the app...")

# Load the model and vectorizer
try:
    with open('sentiment_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

try:
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    st.write("Vectorizer loaded successfully.")
except Exception as e:
    st.error(f"Error loading vectorizer: {e}")
with open(r"C:\Users\HP\Desktop\model\sentiment_model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

# with open('sentiment_model.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)



         
