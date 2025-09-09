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
    page_title="Adversarial Attack Demo",
    page_icon="🎯",
    layout="wide"
)

# Hide Streamlit default UI + Add custom styles
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

    .attack-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .attack-card h4 {
        margin-top: 0;
        color: white !important;
    }

    .doc-container {
        background-color: #f8f9fa;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        border: 1px solid #e9ecef;
        height: 450px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .doc-container h4 {
        margin-top: 0;
        margin-bottom: 15px;
    }

    .doc-container p {
        margin-bottom: 10px;
        line-height: 1.5;
    }

    .doc-container ul {
        margin-top: 10px;
        padding-left: 20px;
    }

    .doc-container li {
        margin-bottom: 5px;
        line-height: 1.4;
    }

    .attack-intro {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 20px 0;
    }

    .feature-box {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #007acc;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)



# Title
st.title("🛡️ Adversarial Machine Learning Attacks ")

st.markdown("""
<div class="attack-intro">
    <h3>🔬 Test Your Model's Security & Robustness</h3>
    <p>Explore different types of adversarial attacks and evaluate your machine learning model's vulnerability to various security threats.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Documentation Section
st.markdown("## 📚 Understanding Adversarial Attacks")

# Create tabs for better organization
doc_tab1, doc_tab2 = st.tabs(["📖 Attack Types Overview", "🚀 Launch Attacks"])

with doc_tab1:
    st.markdown("### What are Adversarial Attacks?")
    st.markdown("""
    **Adversarial attacks** are sophisticated techniques designed to exploit vulnerabilities in machine learning models by manipulating input data or the training process. These attacks reveal critical weaknesses in AI systems that appear to work perfectly under normal conditions but can be fooled, compromised, or exploited through carefully crafted malicious inputs or strategies.
The primary purpose of studying adversarial attacks is not malicious—it's to understand and improve the security, robustness, and reliability of AI systems before they're deployed in critical applications like healthcare, autonomous vehicles, or security systems.
     """)

    doc_col1, doc_col2 = st.columns(2)

    with doc_col1:
        st.markdown("""
        <div class="doc-container">
            <div>
                <h4>🎯 Evasion Attacks</h4>
                <p><strong>What it is:</strong> Evasion attacks involve making tiny, often invisible changes to input data to trick a trained model into making wrong predictions.</p>
                <p><strong>Real-world example:</strong> Adding imperceptible noise to a stop sign image that makes a self-driving car's AI see it as a speed limit sign.</p>
                <p><strong>Why it matters:</strong> Tests how robust your model is against malicious inputs in production environments.</p>
            </div>
            <div>
                <p><strong>Key characteristics:</strong></p>
                <ul>
                    <li>Happens during model deployment/testing phase</li>
                    <li>Attacker has no control over training data</li>
                    <li>Goal: Make the model misclassify specific inputs</li>
                    <li>Most common type of adversarial attack</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
                <div class="doc-container">
                    <div>
                        <h4>🕵️ Inference Attacks</h4>
                        <p><strong>What it is:</strong> Inference attacks attempt to extract sensitive information about the training data or individual data points by analyzing the model's outputs and behavior.</p>
                        <p><strong>Real-world example:</strong> An attacker queries a medical AI model repeatedly to determine if a specific patient's data was used in training.</p>
                        <p><strong>Why it matters:</strong> Essential for protecting privacy, especially with sensitive data like medical records or personal information.</p>
                    </div>
                    <div>
                        <p><strong>Key characteristics:</strong></p>
                        <ul>
                            <li>Happens after model deployment phase</li>
                            <li>Attacker only needs access to model outputs</li>
                            <li>Goal: Infer information about training data</li>
                            <li>Privacy-focused security vulnerability</li>
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with doc_col2:
        st.markdown("""
            <div class="doc-container">
                <div>
                    <h4>💀 Poisoning Attacks</h4>
                    <p><strong>What it is:</strong> Poisoning attacks involve injecting malicious or corrupted data into the training dataset to compromise the model's behavior.</p>
                    <p><strong>Real-world example:</strong> An attacker submits fake reviews with specific patterns during training, causing a sentiment analysis model to always classify certain products negatively.</p>
                    <p><strong>Why it matters:</strong> Critical for understanding risks when training on data from untrusted sources or crowdsourced datasets.</p>
                </div>
                <div>
                    <p><strong>Key characteristics:</strong></p>
                    <ul>
                        <li>Happens during the training phase only</li>
                        <li>Attacker influences the training data directly</li>
                        <li>Goal: Degrade overall model performance</li>
                        <li>Can create persistent backdoors in models</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="doc-container">
            <div>
                <h4>🔓 Extraction Attacks</h4>
                <p><strong>What it is:</strong> Extraction attacks aim to steal or replicate the functionality of a machine learning model by analyzing its inputs and outputs.</p>
                <p><strong>Real-world example:</strong> A competitor queries your proprietary image classification API thousands of times with different images to train their own model.</p>
                <p><strong>Why it matters:</strong> Protects intellectual property and prevents unauthorized copying of expensive-to-train models.</p>
            </div>
            <div>
                <p><strong>Key characteristics:</strong></p>
                <ul>
                    <li>Happens after model deployment phase</li>
                    <li>Attacker systematically queries the model</li>
                    <li>Goal: Create a substitute model copy</li>
                    <li>Intellectual property theft concern</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

with doc_tab2:
    # Attack categories
    st.markdown("### Choose Your Attack Type")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Evasion Attacks")
        st.markdown("Test model robustness against adversarial examples")
        if st.button("🚀 Launch Evasion Attacks", key="evasion_btn", help="Go to Evasion Attacks page"):
            st.session_state.attack_type = "evasion"
            st.switch_page("pages/Evasion.py")

        st.markdown("### 🕵️ Inference Attacks")
        st.markdown("Test model vulnerability to Inference")
        if st.button("🕵️ Launch Inference Attacks", key="inference_btn", help="Go to Inference Attacks page"):
            st.session_state.attack_type = "inference"
            st.switch_page("pages/Inference.py")

    with col2:
        st.markdown("### ☠️ Poisoning Attacks")
        st.markdown("Test model vulnerability to data poisoning")
        if st.button("💀 Launch Poisoning Attacks", key="poison_btn", help="Go to Poisoning Attacks page"):
            st.session_state.attack_type = "poisoning"
            st.switch_page("pages/Poisoning.py")

        st.markdown("### 🔓 Extraction Attack")
        st.markdown("Test model vulnerability to Extraction attacks")
        if st.button("🔓 Launch Extraction Attacks", key="extr_btn", help="Go to Extraction Attacks page"):
            st.session_state.attack_type = "extraction"
            st.switch_page("pages/Extraction.py")

    # Getting Started Guide
    st.markdown("---")
    st.markdown("## 🚀 Getting Started")

    guide_col1, guide_col2, guide_col3 = st.columns(3)

    with guide_col1:
        st.markdown("""
        <div class="feature-box">
            <h4>1️⃣ Learn</h4>
            <p>Read about different attack types in the documentation tab above to understand which attack suits your testing needs.</p>
        </div>
        """, unsafe_allow_html=True)

    with guide_col2:
        st.markdown("""
        <div class="feature-box">
            <h4>2️⃣ Choose</h4>
            <p>Select an attack type based on your security concerns and the vulnerability you want to test.</p>
        </div>
        """, unsafe_allow_html=True)

    with guide_col3:
        st.markdown("""
        <div class="feature-box">
            <h4>3️⃣ Test</h4>
            <p>Configure attack parameters and analyze the results to understand your model's robustness.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Final CTA
    st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin: 20px 0;">
        <h3>🛡️ Ready to Test Your Model's Security?</h3>
        <p>Select an attack type above to begin comprehensive adversarial robustness testing.</p>
    </div>
    """, unsafe_allow_html=True)
