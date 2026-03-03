
import streamlit as st
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from inference import MotivationEngine
from config import MODEL_PATH

st.set_page_config(page_title="Motivation AI", page_icon="💪")
st.title("💪 Motivation AI")

if not MODEL_PATH.exists():
    st.error("Model not trained yet. Run: python backend/train.py first.")
else:
    engine = MotivationEngine()
    user_input = st.text_area("How are you feeling?")

    if st.button("Generate Motivation"):
        if user_input.strip():
            response = engine.generate(user_input)
            st.success(response)
        else:
            st.warning("Please enter a message.")
