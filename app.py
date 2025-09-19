import streamlit as st
import pandas as pd
import os
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import laplace
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# --- CONFIGURATION ---
DATA_FILE = 'my_reviews.csv'
IMAGES_FOLDER = 'images'

# --- FUNCTIONS ---
@st.cache_resource
def setup_nltk():
    nltk.download('vader_lexicon')

@st.cache_data
def analyze_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return 'Neutral'
    sia = SentimentIntensityAnalyzer()
    compound_score = sia.polarity_scores(text)['compound']
    if compound_score <= -0.05:
        return 'Negative'
    return 'Positive'

@st.cache_data
def is_image_blurry(image_path, threshold=100.0):
    try:
        image = io.imread(image_path)
        grayscale_image = rgb2gray(image)
        laplacian_var = laplace(grayscale_image).var()
        return laplacian_var < threshold, laplacian_var
    except Exception:
        return False, 0

# --- STREAMLIT APP ---
st.set_page_config(layout="wide")
st.title("🛍️ Gwalior Handicrafts Review Analyzer")
st.write("This app uses NLP and Computer Vision to flag potentially problematic customer reviews.")

# Load data
try:
    df = pd.read_csv(DATA_FILE)
    setup_nltk()
except FileNotFoundError:
    st.error(f"Error: The data file '{DATA_FILE}' was not found. Please make sure it's in the repository.")
    st.stop()

# Run Analysis Button
if st.button("Run Analysis on Custom Dataset"):
    st.info(f"Analyzing {len(df)} reviews from your dataset...")
    flagged_reviews = []
    
    for index, row in df.iterrows():
        review_text = row['review_text']
        image_filename = row['image_filename']
        is_manually_defective = row['is_defective'] == 1
        image_path = os.path.join(IMAGES_FOLDER, image_filename)

        sentiment = analyze_sentiment(review_text)
        is_blurry = False
        blur_value = 0
        
        # --- KEY CHANGE: Only check the image if the file exists ---
        if os.path.exists(image_path):
            is_blurry, blur_value = is_image_blurry(image_path)

        if sentiment == 'Negative' or is_blurry or is_manually_defective:
            reason = []
            if sentiment == 'Negative':
                reason.append("Negative Sentiment")
            if is_blurry:
                reason.append(f"Blurry Image (Value: {blur_value:.2f})")
            if is_manually_defective and "Negative Sentiment" not in reason:
                reason.append("Manually Flagged Defect")
            
            flagged_reviews.append({
                "product_id": row['product_id'],
                "reason": ", ".join(reason),
                "review_text": review_text,
                "image_path": image_path,
                "image_exists": os.path.exists(image_path) # Track if the image exists
            })
    
    st.success(f"Analysis Complete! Found {len(flagged_reviews)} items needing attention.")
    
    # Display Report
    if flagged_reviews:
        st.subheader("🚩 Flagged Items Report")
        for review in flagged_reviews:
            col1, col2 = st.columns([1, 2])
            with col1:
                # --- KEY CHANGE: Display a warning if the image doesn't exist ---
                if review['image_exists']:
                    st.image(review['image_path'], caption=f"Product ID: {review['product_id']}")
                else:
                    st.warning(f"Image not found: {os.path.basename(review['image_path'])}")
                    st.write(f"Product ID: {review['product_id']}")
            with col2:
                st.error(f"**Reason:** {review['reason']}")
                st.write(f"**Review:** *'{review['review_text']}'*")