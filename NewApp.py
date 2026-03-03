import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import random

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Emotion Analyzer AI",
    page_icon="😊",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Analyze Emotion", "🎮 Mood Games"]
)

# =====================================================
# 🏠 HOME PAGE
# =====================================================
if page == "🏠 Home":

    st.title("😊 Emotion Analyzer AI")
    st.markdown("---")

    st.markdown("## 📖 About The Project")
    st.write("""
    Emotion Analyzer AI is a Machine Learning based web application 
    built using **Streamlit** and **Natural Language Processing (NLP)**.

    It analyzes user text input and predicts the emotion behind it.
    """)

    st.markdown("## 🧠 Technologies Used")
    st.write("""
    - Python
    - Scikit-learn
    - TF-IDF Vectorizer
    - Machine Learning Model
    - Streamlit (Frontend + Deployment)
    - Matplotlib (Visualization)
    """)

    st.markdown("## ⚙️ How It Works")
    st.write("""
    1. User enters text.
    2. Text is transformed using TF-IDF Vectorizer.
    3. Machine Learning model predicts the emotion.
    4. Confidence score is calculated.
    5. Emotion probability distribution is visualized.
    """)

    st.markdown("## 🎯 Features")
    st.write("""
    ✅ Emotion Detection  
    ✅ Confidence Percentage  
    ✅ Probability Graph  
    ✅ Mood Suggestions  
    ✅ Interactive Games  
    ✅ Professional UI  
    """)

    st.markdown("## 💡 Why This Project?")
    st.write("""
    This project demonstrates NLP, Machine Learning deployment,
    interactive UI design, and real-time prediction using Streamlit.
    """)

    st.markdown("---")
    st.info("Use the sidebar to analyze emotions or play games 🎮")

# =====================================================
# 📊 ANALYZE EMOTION PAGE
# =====================================================
elif page == "📊 Analyze Emotion":

    st.title("📊 Emotion Analyzer")

    user_input = st.text_area("Enter your text:", height=150)

    if st.button("Analyze Emotion"):

        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            vect_input = vectorizer.transform([user_input])
            prediction = model.predict(vect_input)[0]
            probabilities = model.predict_proba(vect_input)[0]
            confidence = np.max(probabilities) * 100

            st.session_state.prediction = prediction
            st.session_state.probabilities = probabilities
            st.session_state.confidence = confidence

    if "prediction" in st.session_state:

        prediction = st.session_state.prediction
        probabilities = st.session_state.probabilities
        confidence = st.session_state.confidence

        emotion_emojis = {
            "joy": "😄",
            "sadness": "😢",
            "anger": "😡",
            "fear": "😨",
            "love": "❤️",
            "surprise": "😲"
        }

        emoji = emotion_emojis.get(prediction, "")

        st.success(f"### Emotion: {prediction.upper()} {emoji}")
        st.info(f"Confidence: {confidence:.2f}%")
        st.progress(float(confidence / 100))

        st.markdown("### 📊 Probability Distribution")
        fig = plt.figure()
        plt.bar(model.classes_, probabilities)
        plt.xlabel("Emotions")
        plt.ylabel("Probability")
        plt.title("Prediction Confidence for Each Emotion")
        st.pyplot(fig)

        st.markdown("### 💡 Suggestion")

        suggestions = {
            "joy": "Keep spreading positivity! 🌟",
            "sadness": "Take a break and talk to someone 💙",
            "anger": "Try deep breathing exercises 🌿",
            "fear": "Stay calm. Everything will be okay 💪",
            "love": "Beautiful emotion ❤️",
            "surprise": "Hope it's a good surprise 😄"
        }

        st.write(suggestions.get(prediction, ""))

# =====================================================
# 🎮 GAMES PAGE
# =====================================================
elif page == "🎮 Mood Games":

    st.title("🎮 Mood Booster Games")

    game_choice = st.radio(
        "Choose a game:",
        ["🎯 Number Guessing", "🎈 Inflate the Balloon"]
    )

    # ---------------- NUMBER GUESSING ----------------
    if game_choice == "🎯 Number Guessing":

        if "secret_number" not in st.session_state:
            st.session_state.secret_number = random.randint(1, 10)

        st.write("Guess a number between 1 and 10")

        guess = st.number_input("Enter your guess:", 1, 10, step=1)

        if st.button("Check Guess"):
            if guess == st.session_state.secret_number:
                st.success("🎉 Correct! You guessed it!")
                st.session_state.secret_number = random.randint(1, 10)
            elif guess < st.session_state.secret_number:
                st.warning("Too low! Try again.")
            else:
                st.warning("Too high! Try again.")

    # ---------------- BALLOON GAME ----------------
    elif game_choice == "🎈 Inflate the Balloon":

        if "balloon_size" not in st.session_state:
            st.session_state.balloon_size = 50

        if "balloon_status" not in st.session_state:
            st.session_state.balloon_status = "playing"

        st.write("Inflate the balloon to EXACTLY 150 to win! 🎯")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Inflate 🎈") and st.session_state.balloon_status == "playing":
                st.session_state.balloon_size += 10

        with col2:
            if st.button("Reset Balloon"):
                st.session_state.balloon_size = 50
                st.session_state.balloon_status = "playing"

        st.markdown(
            f"<div style='text-align:center; font-size:{st.session_state.balloon_size}px;'>🎈</div>",
            unsafe_allow_html=True
        )

        if st.session_state.balloon_size == 150:
            st.success("🎉 You Win! Perfect Inflation!")
            st.session_state.balloon_status = "won"

        elif st.session_state.balloon_size > 150:
            st.error("💥 Boom! The balloon popped!")
            st.session_state.balloon_status = "lost"
