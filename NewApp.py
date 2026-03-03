import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Emotion AI Analyzer",
    page_icon="🧠",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = joblib.load("emotion_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ---------------- EMOTION STYLES ----------------
emotion_styles = {
    "joy": {"emoji": "😄", "color": "#28a745"},
    "sadness": {"emoji": "😢", "color": "#007bff"},
    "anger": {"emoji": "😠", "color": "#dc3545"},
    "fear": {"emoji": "😨", "color": "#6f42c1"},
    "love": {"emoji": "❤️", "color": "#e83e8c"},
    "surprise": {"emoji": "😲", "color": "#fd7e14"}
}

emotion_suggestions = {
    "joy": "Keep spreading positivity! 🌟",
    "sadness": "Take a deep breath. Go for a short walk.",
    "anger": "Pause and breathe slowly. Try some movement.",
    "fear": "You are stronger than you think.",
    "love": "Share your gratitude with someone.",
    "surprise": "Enjoy the unexpected moment!"
}

# ---------------- MINI GAME ----------------
def mini_game():
    st.markdown("### 🎮 Mini Mood Game")
    st.write("Click the emoji that matches your predicted emotion!")

    col1, col2, col3 = st.columns(3)

    clicked = None
    with col1:
        if st.button("😄"):
            clicked = "joy"
    with col2:
        if st.button("😢"):
            clicked = "sadness"
    with col3:
        if st.button("😠"):
            clicked = "anger"

    if clicked:
        if clicked == st.session_state.get("last_prediction"):
            st.success("Correct! 🎉 Great job!")
        else:
            st.warning("Not quite! Try again!")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 Emotion AI")
page = st.sidebar.radio("Navigation", ["Home", "Input", "Output"])

# ---------------- HOME PAGE ----------------
if page == "Home":
    st.title("🧠 Emotion AI Analyzer")
    st.markdown("""
    This AI-powered app:
    - Detects emotion from your text  
    - Shows confidence percentage  
    - Displays probability graph  
    - Suggests mood improvement tips  
    - Includes a fun mini game  

    Built with Machine Learning + Streamlit 🚀
    """)

# ---------------- INPUT PAGE ----------------
elif page == "Input":
    st.header("💬 Enter Your Text")
    user_input = st.text_area("How are you feeling today?")

    if st.button("Submit"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            st.session_state["user_text"] = user_input
            st.success("Text submitted! Go to Output page.")

# ---------------- OUTPUT PAGE ----------------
elif page == "Output":

    if "user_text" not in st.session_state:
        st.info("Please enter text in the Input page first.")
    else:
        text = st.session_state["user_text"]
        st.subheader("Your Text")
        st.write(f"> {text}")

        text_vec = vectorizer.transform([text])
        prediction = model.predict(text_vec)[0]

        # Confidence
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(text_vec)[0]
            confidence = np.max(probabilities) * 100
            labels = model.classes_
        else:
            confidence = 100
            probabilities = [1]
            labels = [prediction]

        st.session_state["last_prediction"] = prediction

        style = emotion_styles.get(prediction, {"emoji": "", "color": "black"})

        st.markdown("## 🎯 Prediction Result")

        st.markdown(
            f"<h2 style='color:{style['color']}'>{style['emoji']} {prediction.upper()}</h2>",
            unsafe_allow_html=True
        )

        # Confidence bar
        st.subheader("Confidence Level")
        st.progress(int(confidence))
        st.write(f"**{confidence:.2f}% confident**")

        # Probability Graph
        if hasattr(model, "predict_proba"):
            st.subheader("📊 Emotion Probability Distribution")

            prob_df = pd.DataFrame({
                "Emotion": labels,
                "Probability": probabilities
            })

            fig, ax = plt.subplots()
            ax.bar(prob_df["Emotion"], prob_df["Probability"])
            ax.set_ylabel("Probability")
            ax.set_ylim([0, 1])
            st.pyplot(fig)

        # Suggestion
        st.markdown("### 💡 Suggestion")
        st.info(emotion_suggestions.get(prediction, "Stay positive!"))

        # Mini Game
        mini_game()
