# Spam Email Detection Project

## Overview

This project detects spam emails using a **Naive Bayes classifier** and provides an interactive **Streamlit** web app. Users can:

- View and analyze the dataset.
- Check the model's accuracy and performance.
- Input messages to classify as spam or ham in real-time.

## How It Works

### Data Preprocessing
- The dataset is loaded, unnecessary columns are removed, and text is converted into numerical format using `CountVectorizer`.

### Feature Extraction
- The model uses the **Bag-of-Words** approach to represent text data numerically.

### Model Training
- A **Multinomial Naive Bayes classifier** is trained on the preprocessed dataset to distinguish between spam and ham messages.

### Evaluation
- The model's performance is assessed using accuracy metrics, a confusion matrix, and a classification report.

### Live Prediction
- Users can enter a message, and the app will instantly classify it as spam or ham.

## Key Features

- **User-Friendly Interface**: Built using Streamlit for an intuitive experience.
- **Real-Time Spam Detection**: Instant message classification.
- **Performance Analysis**: Visualizations and reports to understand model accuracy.
- **Efficient and Lightweight**: Uses Naive Bayes for fast and effective text classification.

This project is simple, efficient, and effective for spam detection tasks. 🚀
