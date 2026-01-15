# =====================================================
# 🎭 EMOTION DETECTION & MENTAL STATE ANALYSIS SYSTEM
# ML Powered using NLP + TF-IDF + Logistic Regression
# =====================================================

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
import string
from datetime import datetime

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="Emotion Detection System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------
# LOAD MODEL AND VECTORIZER
# -----------------------------------------------------
@st.cache_resource
def load_model():
    with open("emotion_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("emotion_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

emotion_model, emotion_vectorizer = load_model()

# -----------------------------------------------------
# TEXT CLEANING FUNCTION
# -----------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text

# -----------------------------------------------------
# SESSION STATE (HISTORY STORAGE)
# -----------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------------------------------
# EMOJI MAP
# -----------------------------------------------------
emotion_emojis = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😡",
    "fear": "😨",
    "neutral": "😐"
}

# -----------------------------------------------------
# CUSTOM CSS (UPGRADED AESTHETICS)
# -----------------------------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 3rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #7C3AED, #2563EB, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom:0.2rem;
}
.subtitle {
    text-align:center;
    font-size:1.1rem;
    color:#6B7280;
    margin-bottom:25px;
}
.result-card {
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    font-size: 1.6rem;
    font-weight: 800;
    margin-top: 15px;
}
.happy {background: #ECFDF5; color: #065F46; border: 3px solid #10B981;}
.sad {background: #EFF6FF; color: #1E3A8A; border: 3px solid #3B82F6;}
.angry {background: #FEE2E2; color: #7F1D1D; border: 3px solid #EF4444;}
.fear {background: #FEF3C7; color: #92400E; border: 3px solid #F59E0B;}
.neutral {background: #F3F4F6; color: #374151; border: 3px solid #9CA3AF;}
.metric-box {
    padding: 1rem;
    background: #F9FAFB;
    border-radius: 10px;
    border: 1px solid #E5E7EB;
    text-align:center;
    font-weight:600;
}
.footer {
    text-align:center;
    color:#6B7280;
    font-size:0.9rem;
    margin-top:20px;
}
.stButton>button {
    background: linear-gradient(135deg, #7C3AED, #2563EB);
    color:white;
    font-size:1rem;
    font-weight:600;
    border-radius:8px;
    padding:0.6rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------
with st.sidebar:
    st.markdown("## 🎭 Emotion AI System")
    st.markdown("""
    **Course:** BSD3513 Artificial Intelligence  
    **Model:** TF-IDF + Logistic Regression  
    **Classes:**  
    - happy 😊  
    - sad 😢  
    - angry 😡  
    - fear 😨  
    - neutral 😐  
    """)
    st.markdown("---")
    st.markdown("### 📌 How to Use")
    st.markdown("""
    1. Enter a sentence  
    2. Click **Analyze Emotion**  
    3. View predicted emotion & confidence  
    4. Check history and dashboard  
    """)

# -----------------------------------------------------
# MAIN TITLE
# -----------------------------------------------------
st.markdown('<div class="main-title">🎭 Emotion Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Emotion Recognition using NLP & Machine Learning</div>', unsafe_allow_html=True)

# -----------------------------------------------------
# INPUT SECTION WITH BUTTONS
# -----------------------------------------------------
input_container = st.container()
with input_container:
    st.markdown("## 📝 Enter a Sentence for Emotion Analysis")
    text_input = st.text_area(
        "Type your sentence here:",
        height=120,
        placeholder="Example: I feel very happy and excited today!"
    )

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        analyze_btn = st.button("🔍 Analyze Emotion", use_container_width=True)
    with col2:
        clear_btn = st.button("🧹 Clear History", use_container_width=True)
    with col3:
        sample_btn = st.button("🎲 Load Sample", use_container_width=True)

    if sample_btn:
        text_input = "I feel really sad and lonely today."

    if clear_btn:
        st.session_state.history = []
        st.success("History cleared successfully!")

# -----------------------------------------------------
# PREDICTION LOGIC
# -----------------------------------------------------
if analyze_btn:
    if text_input.strip() == "":
        st.error("❌ Please enter a sentence before analyzing.")
    else:
        cleaned_text = clean_text(text_input)
        vector = emotion_vectorizer.transform([cleaned_text])
        prediction = emotion_model.predict(vector)[0].lower()
        probabilities = emotion_model.predict_proba(vector)[0]
        confidence = np.max(probabilities)

        # Save to history
        st.session_state.history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Sentence": text_input,
            "Predicted Emotion": prediction,
            "Confidence (%)": round(confidence * 100, 2)
        })

        # ------------------- DISPLAY RESULTS IN TABS -------------------
        tabs = st.tabs(["🎯 Result", "📊 Stats & Dashboard", "📜 History"])
        
        # ------------------- RESULT TAB -------------------
        with tabs[0]:
            emoji = emotion_emojis.get(prediction, "❓")
            st.markdown(
                f"<div class='result-card {prediction}'>{emoji} Predicted Emotion: <b>{prediction.upper()}</b><br>Confidence: {confidence*100:.2f}%</div>",
                unsafe_allow_html=True
            )

            # Probability Distribution
            st.markdown("### 📊 Emotion Probability Distribution")
            prob_df = pd.DataFrame({
                "Emotion": emotion_model.classes_,
                "Probability (%)": probabilities * 100
            })
            st.bar_chart(prob_df.set_index("Emotion"))

        # ------------------- STATS TAB -------------------
        with tabs[1]:
            st.markdown("### 🧾 Text Statistics")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"<div class='metric-box'>Words<br><b>{len(text_input.split())}</b></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='metric-box'>Characters<br><b>{len(text_input)}</b></div>", unsafe_allow_html=True)
            with c3:
                caps = sum(1 for c in text_input if c.isupper())
                st.markdown(f"<div class='metric-box'>Capital Letters<br><b>{caps}</b></div>", unsafe_allow_html=True)

            if st.session_state.history:
                st.markdown("---")
                st.markdown("### 📊 Dashboard Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", len(st.session_state.history))
                with col2:
                    most_common = pd.DataFrame(st.session_state.history)["Predicted Emotion"].value_counts().idxmax()
                    st.metric("Most Detected Emotion", most_common.upper())
                with col3:
                    avg_conf = pd.DataFrame(st.session_state.history)["Confidence (%)"].mean()
                    st.metric("Average Confidence (%)", f"{avg_conf:.2f}")

        # ------------------- HISTORY TAB -------------------
        with tabs[2]:
            if st.session_state.history:
                history_df = pd.DataFrame(st.session_state.history)
                st.dataframe(history_df, use_container_width=True)
                csv = history_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download History as CSV",
                    data=csv,
                    file_name="emotion_prediction_history.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No predictions yet. Start by analyzing a sentence above!")

# -----------------------------------------------------
# CASE STUDY EXAMPLES (COLLAPSIBLE)
# -----------------------------------------------------
with st.expander("📚 Case Study Examples (Click to Expand)"):
    case_col1, case_col2 = st.columns(2)
    with case_col1:
        st.success("😊 HAPPY Example")
        st.write("“I finally achieved my goals and I feel amazing today!”")
        st.error("😡 ANGRY Example")
        st.write("“This situation is unfair and I am extremely upset.”")
    with case_col2:
        st.warning("😨 FEAR Example")
        st.write("“I am scared about what will happen tomorrow.”")
        st.info("😐 NEUTRAL Example")
        st.write("“The meeting will start at 10 AM tomorrow.”")

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------
st.markdown("---")
st.markdown("""
<div class="footer">
🎭 <b>Emotion Detection & Mental State Analysis System</b><br>
BSD3513 Artificial Intelligence Group Project<br>
Built using NLP, TF-IDF, Logistic Regression & Streamlit<br>
For academic and educational purposes only.
</div>
""", unsafe_allow_html=True)
