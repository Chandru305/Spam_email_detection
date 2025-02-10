import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Title of the Streamlit app
st.title("Spam Message Classifier")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

# Train model function
@st.cache_data
def train_model(df):
    cv = CountVectorizer()
    X = cv.fit_transform(df['message'])
    X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    accuracy = mnb.score(X_test, y_test)
    y_pred = mnb.predict(X_test)
    return mnb, cv, accuracy, y_test, y_pred

# Load data and train model
df = load_data()
mnb, cv, accuracy, y_test, y_pred = train_model(df)

# Convert predictions to int type
y_test = y_test.astype(int)
y_pred = y_pred.astype(int)

# Live predictor section (moved to top)
st.header("📱 Live Message Classifier")
st.write("Enter a message below to check if it's spam or not:")
user_input = st.text_area("Message", height=100)
if user_input:
    user_message_transformed = cv.transform([user_input])
    prediction = mnb.predict(user_message_transformed)
    prediction_proba = mnb.predict_proba(user_message_transformed)[0]
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Prediction:** {'SPAM' if prediction[0] == 1 else 'HAM (Not Spam)'}")
    with col2:
        st.info(f"**Confidence:** {max(prediction_proba):.2%}")

# Model performance metrics (moved below)
st.header("🎯 Model Performance")
st.metric("Model Accuracy", f"{accuracy:.2%}")

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# Classification Report
st.subheader("Detailed Classification Report")
st.text(classification_report(y_test, y_pred))

# Dataset Preview (moved to bottom)
st.header("📊 Dataset Preview")
st.dataframe(df.head())

# Add some statistics about the dataset
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Messages", len(df))
with col2:
    st.metric("Spam Messages", len(df[df['label'] == 1]))