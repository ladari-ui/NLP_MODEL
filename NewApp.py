import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import random

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Emotion Analyzer AI", page_icon="😊")

# ---------------- LOAD MODEL ----------------
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Analyze Emotion", "🎮 Games"])

# ---------------- HOME PAGE ----------------
if page == "🏠 Home":
    st.title("😊 Emotion Analyzer AI")
    st.write("Analyze your emotions using AI and boost your mood with fun games!")
    st.write("Use the sidebar to navigate.")

# ---------------- ANALYZE PAGE ----------------
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

    # Show results if available
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

        st.success(f"Emotion: {prediction.upper()} {emoji}")
        st.info(f"Confidence: {confidence:.2f}%")
        st.progress(float(confidence / 100))

        # Graph
        st.subheader("📊 Probability Distribution")
        fig = plt.figure()
        plt.bar(model.classes_, probabilities)
        plt.xlabel("Emotions")
        plt.ylabel("Probability")
        st.pyplot(fig)

# ---------------- GAMES PAGE ----------------
elif page == "🎮 Games":

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

        st.write("Click inflate but don't pop it! 💥")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Inflate 🎈"):
                st.session_state.balloon_size += 10

        with col2:
            if st.button("Reset"):
                st.session_state.balloon_size = 50

        st.markdown(
            f"<div style='text-align:center; font-size:{st.session_state.balloon_size}px;'>🎈</div>",
            unsafe_allow_html=True
        )

        if st.session_state.balloon_size > 150:
            st.error("💥 Boom! The balloon popped!")
            st.session_state.balloon_size = 50
