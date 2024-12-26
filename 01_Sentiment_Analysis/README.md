**Sentiment Analysis App**

This is a **Sentiment Analysis Application** built using **Streamlit**. The app allows users to analyze the sentiment of text using two different methods:

**HuggingFace Transformers:** A pre-trained model for sentiment analysis.

**spaCy with TextBlob:** A lightweight, rule-based sentiment analysis.

The app also provides functionality to analyze sentiment for entire datasets via file uploads.

**Features**

**1. Single Text Sentiment Analysis**

Users can input text manually to analyze its sentiment using:

**HuggingFace Transformer Models.**
**spaCy with TextBlob.**

The sentiment is categorized as:
**Positive**

**Negative**

**Neutral**

Results are displayed with background colors for quick visual feedback.

**2. Dataset Sentiment Analysis**

Upload a CSV file with a text column.

Choose between HuggingFace or spaCy as the analysis method.

Analyze sentiment for each row in the dataset.

Download the results as a CSV file containing the text, sentiment, and confidence/polarity score.
