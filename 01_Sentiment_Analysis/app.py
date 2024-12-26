import streamlit as st
import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from transformers import pipeline
import subprocess

def load_hf_model():
    model_name = 'sentiment-analysis'
    return pipeline(model_name)

def analyze_sentiment_hf(text):
    sentiment_pipeline = load_hf_model()
    result = sentiment_pipeline(text)[0]
    sentiment = result['label']
    confidence = result['score']
    return sentiment, confidence

def download_spacy_model():
    model_name = "en_core_web_sm"
    subprocess.run(["python", "-m", "spacy", "download", model_name])

def analyze_sentiment_spacy(text):
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("spacytextblob")
    doc = nlp(text)
    polarity = doc._.blob.polarity
    if polarity > 0.1:
        sentiment = "POSITIVE"
    elif polarity < -0.1:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    return sentiment, polarity

def display_sentiment_with_color(sentiment, source):
    sentiment = sentiment.upper()
    if sentiment == "POSITIVE":
        sentiment_color = "green"
    elif sentiment == "NEGATIVE":
        sentiment_color = "red"
    else:
        sentiment_color = "white"
    st.markdown(
        f"<p style='background-color: {sentiment_color}; padding: 10px; border-radius: 5px;'>"
        f"{source} sentiment: {sentiment}"
        "</p>",
        unsafe_allow_html=True
    )

def process_uploaded_file(file, analysis_method):
    try:
        df = pd.read_csv(file)
        if 'text' not in df.columns:
            st.warning("Uploaded dataset must contain a 'text' column.")
            return None

        results = []
        if analysis_method == "HuggingFace":
            sentiment_pipeline = load_hf_model()
            for text in df['text']:
                result = sentiment_pipeline(text)[0]
                results.append((text, result['label'], result['score']))
        elif analysis_method == "spaCy":
            nlp = spacy.load("en_core_web_sm")
            nlp.add_pipe("spacytextblob")
            for text in df['text']:
                doc = nlp(text)
                polarity = doc._.blob.polarity
                if polarity > 0.1:
                    sentiment = "POSITIVE"
                elif polarity < -0.1:
                    sentiment = "NEGATIVE"
                else:
                    sentiment = "NEUTRAL"
                results.append((text, sentiment, polarity))

        results_df = pd.DataFrame(results, columns=['Text', 'Sentiment', 'Score/Polarity'])
        return results_df

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def main():
    st.set_page_config(page_title="Sentiment Analysis")

    st.title("Welcome to the Sentiment Analysis app!")
    st.write("This app performs sentiment analysis on text using two different methods: HuggingFace and spaCy.")
    st.write("The sentiment analysis classifies text into three categories: positive, negative, or neutral.")

    # Section 1: Single Text Sentiment Analysis
    st.header("Single Text Sentiment Analysis")

    # HuggingFace
    st.subheader("HuggingFace Sentiment Analysis")
    huggingface_text = st.text_area("Enter text for HuggingFace analysis:", height=150)
    if st.button("Analyze with HuggingFace"):
        if huggingface_text:
            huggingface_sentiment, huggingface_confidence = analyze_sentiment_hf(huggingface_text)
            display_sentiment_with_color(huggingface_sentiment, "HuggingFace")
            st.write(f"HuggingFace confidence: {huggingface_confidence}")
        else:
            st.warning("Please enter some text for HuggingFace analysis.")

    # spaCy
    st.subheader("spaCy Sentiment Analysis")
    spacy_text = st.text_area("Enter text for spaCy analysis:", height=150)
    if st.button("Analyze with spaCy"):
        if spacy_text:
            spacy_sentiment, spacy_polarity = analyze_sentiment_spacy(spacy_text)
            display_sentiment_with_color(spacy_sentiment, "spaCy")
            st.write(f"spaCy polarity: {spacy_polarity}")
        else:
            st.warning("Please enter some text for spaCy analysis.")

    # Section 2: Dataset Sentiment Analysis
    st.header("Dataset Sentiment Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column:", type="csv")
    analysis_method = st.selectbox("Select Analysis Method", ["HuggingFace", "spaCy"])

    if st.button("Analyze Dataset"):
        if uploaded_file:
            result_df = process_uploaded_file(uploaded_file, analysis_method)
            if result_df is not None:
                st.write("Analysis Results:")
                st.dataframe(result_df)
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results", data=csv, file_name="sentiment_analysis_results.csv", mime="text/csv")
        else:
            st.warning("Please upload a valid CSV file.")

if __name__ == "__main__":
    main()
