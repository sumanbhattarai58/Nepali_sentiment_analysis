import streamlit as st
import requests

# -----------------------------
# Streamlit UI Configuration
# -----------------------------
st.set_page_config(page_title="Nepali Sentiment Analysis", layout="centered")
st.title("🇳🇵 Nepali Sentiment Analysis")
st.write("Enter a Nepali sentence below to analyze its sentiment (Positive, Negative, or Neutral).")

# -----------------------------
# Input Section
# -----------------------------
text_input = st.text_area("📝 Enter text:", "यो गीत एकदमै राम्रो छ ❤️")

if st.button("🔍 Analyze Sentiment"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Make request to FastAPI (make sure backend is running on this URL)
                response = requests.post(
                    "http://127.0.0.1:8000/predict",  # Local FastAPI endpoint
                    json={"text": text_input}
                )
                if response.status_code == 200:
                    result = response.json()
                    sentiment = result.get("sentiment", "Unknown").capitalize()
                    confidence = result.get("confidence", 0)

                    st.success(f"**Predicted Sentiment:** {sentiment}")
                    st.progress(float(confidence))
                else:
                    st.error(f"Server error: {response.status_code}")
            except Exception as e:
                st.error(f"⚠️ Could not connect to FastAPI backend.\n\nDetails: {e}")

st.caption("💡 Make sure your FastAPI server is running on port 8000 before analyzing.")
