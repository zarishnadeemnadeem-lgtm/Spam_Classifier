import streamlit as st
import joblib

# Saved model aur vectorizer load karein
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('spam_model .pkl')

# App ka Title
st.title("📩 SMS & Email Spam Classifier")
st.write("Enter text below to check if it's Spam or Safe (Ham).")

# Input Box
user_input = st.text_area("Message Text:", "")

if st.button("Predict"):
    if user_input.strip() != "":
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        if prediction == 1:
            st.error(f"🚨 **SPAM Message!** (Confidence: {probabilities[1]*100:.1f}%)")
        else:
            st.success(f"✅ **HAM (Safe) Message!** (Confidence: {probabilities[0]*100:.1f}%)")
    else:
        st.warning("Please enter some text first.")