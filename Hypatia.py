

import streamlit as st
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from absl.testing.parameterized import parameters
from art.utils import to_categorical

from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from PIL import Image

st.set_page_config(
    page_title="Hypatia - Adversarial ML Testing Platform",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .css-1d391kg {display: none !important;}
    .css-1rs6os {display: none !important;}
    .css-17eq0hr {display: none !important;}
    section[data-testid="stSidebar"] {display: none !important;}
    .css-164nlkn {display: none !important;}
    button[kind="header"] {display: none !important;}
    .css-1cypcdb {display: none !important;}
    .css-k1vhr4 {display: none !important;}
    .hypatia-header {
        text-align: center;
        background: linear-gradient(135deg, #87CEEB 0%, #4682B4 50%, #1E90FF 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }

 

    .hypatia-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .hypatia-subtitle {
        font-size: 1.2rem;
        margin-bottom: 1rem;
        opacity: 0.9;
    }

    .hypatia-quote {
        font-style: italic;
        font-size: 1.1rem;
        margin-top: 1rem;
        padding: 1rem;
        background: rgba(255,255,255,0.1);
        border-radius: 5px;
        border-left: 4px solid #ffd700;
    }
    

    .attack-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }

    .attack-card:hover {
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Header section with logos
img_col1, img_col_empty, img_col2 = st.columns([1, 6, 1])

with img_col1:
    try:
        image1 = Image.open("pages/ulfg logo.png")
        st.image(image1, width=100)
    except:
        st.write("🏛️")  # Fallback icon

with img_col2:
    try:
        image2 = Image.open("pages/versifai_logo.png")
        st.image(image2, width=100)
    except:
        st.write("🔬")  # Fallback icon

# Main Hypatia header
st.markdown("""
<div class="hypatia-header">
    <div class="hypatia-title">🧠 HYPATIA</div>
    <div class="hypatia-subtitle">Advanced Adversarial Machine Learning Testing Platform</div>
    <div style="font-size: 1rem; opacity: 0.8;">
        Empowering researchers and practitioners to evaluate ML model robustness through comprehensive adversarial testing
    </div>
    <div class="hypatia-quote">
        "Reserve your right to think, for even to think wrongly is better than not to think at all."<br>
        <strong>— Hypatia of Alexandria</strong>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("### 🔍 About Hypatia")
st.markdown("""
Named after **Hypatia of Alexandria**, the renowned ancient mathematician, astronomer, and philosopher, 
this platform embodies her spirit of rigorous inquiry and fearless pursuit of knowledge. Just as Hypatia 
challenged conventional thinking in her time, **Hypatia** challenges your machine learning models to reveal 
their vulnerabilities and strengthen their defenses against adversarial threats.

Our comprehensive suite of adversarial testing tools helps you:
- **Discover hidden vulnerabilities** in your ML models
- **Evaluate robustness** against various attack vectors  
- **Strengthen defenses** through systematic testing
- **Build more secure** and reliable AI systems
""")

# Initialize session state for showing attack methods
if 'show_attacks' not in st.session_state:
    st.session_state.show_attacks = False

st.markdown("---")

# Show button to go to attack methods page
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🛡️ **Explore Adversarial Testing Methods**", key="show_methods_btn", help="Go to Adversarial Testing Methods page", use_container_width=True):
        st.switch_page("pages/AdversarialAttack.py")

st.markdown("---")

# Footer with additional information
st.markdown("""
<div style="text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
    <h4>🧭 Begin Your Adversarial Testing Journey</h4>
    <p>Select an attack type above to start evaluating your model's adversarial robustness. 
    Each testing method provides comprehensive insights into different aspects of ML security.</p>
    <p><em>Remember: In the spirit of Hypatia, question everything and test thoroughly.</em></p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="position: fixed; bottom: 0; left: 0; right: 0; background: linear-gradient(90deg, #2c3e50 0%, #34495e 100%); 
            padding: 10px; text-align: center; color: white; font-size: 0.9em; z-index: 999;">
    🧠 Hypathia Platform • Inspired by the Mathematical Genius of Ancient Alexandria • 
    <span style="color: #87CEEB;">Advanced ML Security Research</span>
</div>
""", unsafe_allow_html=True)
