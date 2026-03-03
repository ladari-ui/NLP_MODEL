import streamlit as st
import joblib
import random

# Load model & vectorizer
@st.cache_data
def load_model():
    model = joblib.load("emotion_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'user_text' not in st.session_state:
    st.session_state.user_text = ""
if 'number_game' not in st.session_state:
    st.session_state.number_game = random.randint(1, 20)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Input", "Output"])
st.session_state.page = page

# ------------------- HOME PAGE -------------------
if st.session_state.page == "Home":
    st.title("😊 Emotion Detection & Mood Booster App")
    st.write("""
    Welcome! This app helps you understand your emotions from text.
    
    Features:
    - Predict your emotion using a machine learning model
    - Suggestions to improve your mood
    - A small interactive game to ease your mind
    
    Navigate to the **Input** page to type your thoughts.
    """)

# ------------------- INPUT PAGE -------------------
elif st.session_state.page == "Input":
    st.title("📝 Share Your Thoughts")
    user_text = st.text_area("Type here what's on your mind:", value=st.session_state.user_text)
    st.session_state.user_text = user_text

    if st.button("Submit for Analysis"):
        if user_text.strip() != "":
            st.success("Text submitted! Go to Output page for results.")
        else:
            st.warning("Please enter some text.")

# ------------------- OUTPUT PAGE -------------------
elif st.session_state.page == "Output":
    if st.session_state.user_text.strip() == "":
        st.warning("Go to the Input page and enter your text first.")
    else:
        user_text = st.session_state.user_text
        st.title("🔮 Your Emotion & Mood Boost")

        # Predict emotion
        text_vec = vectorizer.transform([user_text])
        prediction = model.predict(text_vec)[0]
        st.subheader(f"Predicted Emotion: {prediction.upper()}")

        # Suggestions based on emotion
        suggestions = {
            "joy": ["Keep smiling! Share your joy with someone.", "Listen to your favorite upbeat song!"],
            "sadness": ["Talk to a friend you trust.", "Listen to uplifting music or take a short walk."],
            "anger": ["Take deep breaths or meditate.", "Do a quick workout to release tension."],
            "love": ["Express your feelings to someone you care about.", "Write down why you feel grateful."],
            "fear": ["Identify the cause and face it calmly.", "Talk it out with a trusted person."],
            "surprise": ["Embrace the moment!", "Share the exciting news with someone."],
            "neutral": ["Take a short break and relax.", "Do something you enjoy."]
        }

        st.markdown("### Suggestions to Improve Mood:")
        for s in suggestions.get(prediction.lower(), ["Take a deep breath and relax."]):
            st.write("- " + s)

        # ----------------- Mini Game -----------------
        st.markdown("---")
        st.markdown("### 🎮 Quick Mood Game: Guess the Number!")
        number = st.session_state.number_game

        guess = st.number_input("Guess a number between 1 and 20:", min_value=1, max_value=20, step=1)
        if st.button("Check Guess"):
            if guess == number:
                st.balloons()
                st.success("🎉 Correct! You got it!")
                st.session_state.number_game = random.randint(1, 20)
            elif guess < number:
                st.info("Too low! Try again.")
            else:
                st.info("Too high! Try again.")