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

# ---------------- TITLE ----------------
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
        😊 Emotion Analyzer AI
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- USER INPUT ----------------
user_input = st.text_area("Enter your text:", height=150)

# ---------------- EMOTION EMOJIS ----------------
emotion_emojis = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

# ---------------- PREDICTION ----------------
if st.button("Analyze Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize
        vect_input = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(vect_input)[0]
        probabilities = model.predict_proba(vect_input)[0]

        confidence = np.max(probabilities) * 100

        st.markdown("## 🔍 Prediction Result")

        emoji = emotion_emojis.get(prediction, "")

        st.success(f"**Emotion:** {prediction.upper()} {emoji}")
        st.info(f"**Confidence:** {confidence:.2f}%")

        st.progress(float(confidence / 100))

        # ---------------- PROBABILITY GRAPH ----------------
        st.markdown("### 📊 Emotion Probability Distribution")

        fig = plt.figure()
        plt.bar(model.classes_, probabilities)
        plt.xlabel("Emotions")
        plt.ylabel("Probability")
        plt.title("Prediction Confidence for Each Emotion")
        st.pyplot(fig)

        st.markdown("---")

        # ---------------- SUGGESTIONS ----------------
        st.markdown("### 💡 Suggestion")

        suggestions = {
            "joy": "Keep spreading positivity and happiness! 🌟",
            "sadness": "Take a break and talk to someone you trust 💙",
            "anger": "Try deep breathing or a short walk 🚶‍♂️",
            "fear": "Everything will be okay. Stay calm and strong 💪",
            "love": "Beautiful emotion! Share it with the world ❤️",
            "surprise": "Hope it’s a good one! Stay curious 😄"
        }

        st.write(suggestions.get(prediction, ""))

        st.markdown("---")

        # ---------------- MINI GAMES ----------------
        st.markdown("## 🎮 Mood Booster Games")

        game_choice = st.radio(
            "Choose a game:",
            ["🎯 Number Guessing", "🎈 Inflate the Balloon"]
        )

        # ---------------- NUMBER GUESSING GAME ----------------
        if game_choice == "🎯 Number Guessing":

            if "secret_number" not in st.session_state:
                st.session_state.secret_number = random.randint(1, 10)

            st.write("Guess a number between 1 and 10")

            guess = st.number_input(
                "Enter your guess:",
                min_value=1,
                max_value=10,
                step=1
            )

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

            st.write("Click inflate to grow the balloon — but don't pop it! 💥")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Inflate 🎈"):
                    st.session_state.balloon_size += 10

            with col2:
                if st.button("Reset Balloon"):
                    st.session_state.balloon_size = 50

            # Display balloon
            st.markdown(
                f"<div style='text-align:center; font-size:{st.session_state.balloon_size}px;'>🎈</div>",
                unsafe_allow_html=True
            )

            if st.session_state.balloon_size > 150:
                st.error("💥 Boom! The balloon popped!")
                st.session_state.balloon_size = 50
