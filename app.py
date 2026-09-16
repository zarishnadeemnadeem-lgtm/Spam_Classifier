import streamlit as st
import joblib
import re
import string

model = joblib.load('spam_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.title("📩 SMS Spam Classifier")
st.write("Enter a message below to check if it's Spam or Ham.")

user_input = st.text_area("Message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned = clean_text(user_input)
        vec = tfidf.transform([cleaned])
        pred = model.predict(vec)[0]
        if pred == 1:
            st.error("🚨 This is SPAM")
        else:
            st.success("✅ This is HAM (Not Spam)")