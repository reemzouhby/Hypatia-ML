import  streamlit as st
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
</style>
""", unsafe_allow_html=True)

img_col1, img_col_empty, img_col2 = st.columns([1, 6, 1])

with img_col1:
    image1 = Image.open("Practice/task1/Poisnning/ulfg logo.png")
    st.image(image1, width=100)

with img_col2:
    image2 = Image.open("Practice/task1/Poisnning/versifai_logo.png")
    st.image(image2, width=100)

st.title("🛡️ Adversarial Machine Learning Attacks ")
st.markdown("---")


col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 Evasion Attacks")
    st.markdown("Test model robustness against adversarial examples")
    if st.button("🚀 Launch Evasion Attacks", key="evasion_btn", help="Go to Evasion Attacks page"):
        st.session_state.attack_type = "evasion"
        st.switch_page("pages/Evasion.py")
    st.markdown("###  🕵️ Inference  Attacks")
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
       st.session_state.aimport  streamlit as st
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
</style>
""", unsafe_allow_html=True)

img_col1, img_col_empty, img_col2 = st.columns([1, 6, 1])

with img_col1:
    image1 = Image.open("pages/ulfg logo.png")
    st.image(image1, width=100)

with img_col2:
    image2 = Image.open("pages/versifai_logo.png")
    st.image(image2, width=100)

st.title("🛡️ Adversarial Machine Learning Attacks ")
st.markdown("---")


col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 Evasion Attacks")
    st.markdown("Test model robustness against adversarial examples")
    if st.button("🚀 Launch Evasion Attacks", key="evasion_btn", help="Go to Evasion Attacks page"):
        st.session_state.attack_type = "evasion"
        st.switch_page("pages/Evasion.py")
    st.markdown("###  🕵️ Inference  Attacks")
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

st.markdown("---")
st.markdown("Select an attack type to begin testing your model's adversarial robustness.")ttack_type = "extraction"
       st.switch_page("pages/Extraction.py")

st.markdown("---")
st.markdown("Select an attack type to begin testing your model's adversarial robustness.")
