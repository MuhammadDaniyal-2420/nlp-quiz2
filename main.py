import streamlit as st
from transformers import pipeline

# Load pre-trained text classification model from Hugging Face
classifier = pipeline("SamLowe/roberta-base-go_emotions")

# Streamlit app
def main():
    st.title("Text Classification App")
    st.write("Enter text below to get sentiment analysis predictions.")

    # User input for text
    user_input = st.text_area("Enter text:")

    # Make prediction when the user clicks the button
    if st.button("Get Sentiment Analysis"):
        if user_input:
            # Perform text classification
            result = classifier(user_input)
            st.write("Prediction:", result[0]['label'])
            st.write("Score:", result[0]['score'])
        else:
            st.warning("Please enter text for analysis.")

# Run the app
if __name__ == "__main__":
    main()
