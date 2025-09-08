import streamlit as st
import os
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense
from art.estimators.classification import KerasClassifier
from art.attacks.extraction import CopycatCNN
from art.utils import to_categorical
import tensorflow as tf

# ==== Config ====
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
st.set_page_config(page_title="Light CopyCatCNN Demo", layout="wide")
st.title("🔓 Light CopyCatCNN Attack Demo")
st.markdown("---")

# ==== Load MNIST ====
@st.cache_data
def load_data(max_train=5000, max_test=1000):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images[:max_train] / 255.0
    test_images = test_images[:max_test] / 255.0
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)
    train_labels_cat = to_categorical(train_labels[:max_train], num_classes=10)
    test_labels_cat = to_categorical(test_labels[:max_test], num_classes=10)
    return (train_images, train_labels_cat), (test_images, test_labels_cat)

(train_images, train_labels), (test_images, test_labels) = load_data()

# ==== Define Tiny Models ====
def get_tiny_model():
    tf.keras.backend.clear_session()
    model = Sequential([
        Flatten(input_shape=(28,28,1)),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==== Streamlit Sidebar ====
st.sidebar.header("⚔️ Attack Configuration")
nb_stolen = st.sidebar.slider("Number of Samples to Steal", 100, 500, 300, step=50)
batch_size_fit = st.sidebar.slider("Batch Size (Training)", 8, 64, 16, step=8)
batch_size_query = st.sidebar.slider("Batch Size (Query)", 8, 64, 16, step=8)
nb_epochs = st.sidebar.slider("Training Epochs", 1, 5, 3)

run_button = st.button("🚀 Run CopyCatCNN Attack")

# ==== Run Attack ====
if run_button:
    st.info("⚠️ Running light CopyCatCNN attack. Please wait...")

    # Original classifier
    target_model = get_tiny_model()
    classifier = KerasClassifier(model=target_model, clip_values=(0,1))

    # Train target model quickly
    target_model.fit(train_images, train_labels,
                     epochs=2, batch_size=32, verbose=0)
    
    # Prepare stolen model
    stolen_model = get_tiny_model()
    classifier_stolen = KerasClassifier(stolen_model, clip_values=(0,1))

    # Select samples to steal
    x_steal = test_images[:nb_stolen]

    # CopyCatCNN Attack
    attack = CopycatCNN(
        classifier,
        batch_size_fit=batch_size_fit,
        batch_size_query=batch_size_query,
        nb_epochs=nb_epochs,
        use_probability=True,
        nb_stolen=nb_stolen
    )

    classifier_stolen = attack.extract(thieved_classifier=classifier_stolen, x=x_steal)

    # Evaluate
    test_remaining = test_images[nb_stolen:]
    y_test_remaining = test_labels[nb_stolen:]
    loss_orig, acc_orig = classifier.model.evaluate(test_remaining, y_test_remaining, verbose=0)
    loss_stolen, acc_stolen = classifier_stolen.model.evaluate(test_remaining, y_test_remaining, verbose=0)

    # Fidelity
    org_pred = np.argmax(classifier.predict(test_remaining), axis=1)
    stol_pred = np.argmax(classifier_stolen.predict(test_remaining), axis=1)
    fidelity = np.mean(org_pred == stol_pred)

    # Show metrics
    st.success("✅ CopyCatCNN attack completed!")
    col1, col2, col3 = st.columns(3)
    col1.metric("Original Accuracy", f"{acc_orig:.3f}", f"{acc_orig*100:.1f}%")
    col2.metric("Stolen Accuracy", f"{acc_stolen:.3f}", f"{acc_stolen*100:.1f}%")
    col3.metric("Fidelity", f"{fidelity:.3f}", f"{fidelity*100:.1f}%")
