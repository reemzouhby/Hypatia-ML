import streamlit as st
import os
import gc
import psutil

from art.utils import to_categorical
from sklearn.model_selection import train_test_split

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import backend as K
import warnings
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import cv2
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import pandas as pd
from art.estimators.classification import KerasClassifier
from art.attacks.extraction import CopycatCNN, FunctionallyEquivalentExtraction, KnockoffNets

# Memory optimization settings
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0],
                                         True) if tf.config.list_physical_devices('GPU') else None
tf.keras.backend.clear_session()

st.set_page_config(
    page_title="Extraction Attacks on MNIST",
    page_icon="🔓",
    layout="wide"
)
st.title("🔓 Extraction Attacks on MNIST")
st.markdown("---")


# --- Memory & helper functions ---
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB


def clear_memory():
    gc.collect()
    K.clear_session()


def clamp_nb_stolen(nb_stolen, total_len):
    if nb_stolen >= total_len:
        st.warning(
            f"Requested nb_stolen ({nb_stolen}) >= available test samples ({total_len}). Clamping to leave at least 1 sample.")
        nb_stolen = max(0, total_len - 1)
    return nb_stolen





# --- Data & models ---
@st.cache_resource
def load_mnist_model():
    return load_model("pages/mnist_model.h5")


@st.cache_data
def load_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    max_train_size, max_test_size = 10000, 5000
    train_images, train_labels = train_images[:max_train_size], train_labels[:max_train_size]
    test_images, test_labels = test_images[:max_test_size], test_labels[:max_test_size]
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)
    return (train_images, train_labels), (test_images, test_labels)


@st.cache_data
def load_external_dataset(dataset_name, max_samples=5000):
    if dataset_name == "CIFAR-10":
        (x_train, _), (x_test, _) = cifar10.load_data()
        x_combined = np.concatenate([x_train, x_test])[:max_samples]
        x_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_combined])
        x_processed = np.array([cv2.resize(img, (28, 28)) for img in x_gray])
        return x_processed.reshape(-1, 28, 28, 1) / 255.0
    elif dataset_name == "Fashion-MNIST":
        (x_train, _), (x_test, _) = fashion_mnist.load_data()
        x_combined = np.concatenate([x_train, x_test])[:max_samples]
        return x_combined.reshape(-1, 28, 28, 1) / 255.0


def get_model(NUM_CLASSES):
    tf.keras.backend.clear_session()
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model_lightweight(NUM_CLASSES):
    tf.keras.backend.clear_session()
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model_FEE():
    tf.keras.backend.clear_session()
    model = Sequential([
        Dense(128, activation="relu", input_shape=(784,)),
        Dense(10, activation="linear")
    ])
    return model


def process_knockoff_in_batches(attack, classifier_stolen, x_steal, y_steal, batch_size=500):
    total_samples = len(x_steal)
    num_batches = (total_samples + batch_size - 1) // batch_size
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)
        batch_x = x_steal[start_idx:end_idx]
        batch_y = y_steal[start_idx:end_idx]
        status_text.text(f"Processing batch {i + 1}/{num_batches} ({len(batch_x)} samples)")
        if i == 0:
            classifier_stolen = attack.extract(thieved_classifier=classifier_stolen, x=batch_x, y=batch_y)
        else:
            classifier_stolen._model.fit(batch_x, batch_y, batch_size=min(32, len(batch_x)), epochs=1, verbose=0)
        progress_bar.progress((i + 1) / num_batches)
        clear_memory()
    progress_bar.empty()
    status_text.empty()
    return classifier_stolen


# --- Load model & data ---
if 'data_loaded' not in st.session_state:
    with st.spinner("Loading model and data..."):
        model = load_mnist_model()
        if model:
            (train_images, train_labels), (test_images, test_labels) = load_data()
            classifier = KerasClassifier(model=model, clip_values=(0, 1))
            st.session_state.model = model
            st.session_state.classifier = classifier
            st.session_state.train_data = (train_images, train_labels)
            st.session_state.test_data = (test_images, test_labels)
            st.session_state.data_loaded = True
        else:
            st.error("Failed to load model")
            st.stop()

classifier = st.session_state.classifier
train_images, train_labels = st.session_state.train_data
test_images, test_labels = st.session_state.test_data
model = st.session_state.model

st.sidebar.header("⚔️ Attack Configuration")
options = ["CopyCatCNN", "Functionally Equivalent Extraction", "Knockoff Nets"]
attack_type = st.sidebar.selectbox("Select Attack", options)

# --- Parameters ---
param = {}
if attack_type == "CopyCatCNN":
    steal_dataset = st.sidebar.selectbox("Dataset for Stealing", ["MNIST Test Set", "CIFAR-10", "Fashion-MNIST"])
    param["batch_size_fit"] = st.sidebar.slider("Batch Size (Training)", 16, 128, 64, 16)
    param["batch_size_query"] = st.sidebar.slider("Batch Size (Query)", 16, 128, 64, 16)
    param["nb_epochs"] = st.sidebar.slider("Training Epochs", 1, 20, 10)
    param["nb_stolen"] = st.sidebar.slider("Number of Samples to Steal", 1000, 10000, 5000, 500)
    param["use_probability"] = st.sidebar.checkbox("Use Probability Output", value=True)
elif attack_type == "Functionally Equivalent Extraction":
    st.warning(
        "⚠️ **Research Note**: This attack can take approximately **4 days** to complete with optimal parameters. For researchers seeking perfect results, please plan accordingly.")

    param["num_neurons"] = st.sidebar.number_input("Number of Neurons", min_value=64, max_value=512, value=128, step=64)
    with st.sidebar.expander("🔧 Advanced Parameters"):
        param["delta_0"] = st.number_input("Delta 0 (Initial step size)", min_value=0.001, max_value=0.1, value=0.05,
                                           step=0.001, format="%.3f")
        param["fraction_true"] = st.number_input("Fraction True", min_value=0.1, max_value=0.9, value=0.3, step=0.1)
        param["rel_diff_slope"] = st.number_input("Relative Diff Slope", min_value=1e-7, max_value=1e-3, value=1e-5,
                                                  format="%.2e")
        param["rel_diff_value"] = st.number_input("Relative Diff Value", min_value=1e-8, max_value=1e-4, value=1e-6,
                                                  format="%.2e")
        param["delta_init_value"] = st.number_input("Delta Init Value", min_value=0.01, max_value=1.0, value=0.1,
                                                    step=0.01)
        param["delta_value_max"] = st.number_input("Delta Value Max", min_value=10, max_value=100, value=50, step=10)

elif attack_type == "Knockoff Nets":
    steal_dataset = st.sidebar.selectbox("Dataset for Stealing", ["MNIST Test Set", "CIFAR-10", "Fashion-MNIST"])
    param["batch_size_fit"] = st.sidebar.slider("Batch Size (Training)", 8, 64, 16, 8)
    param["batch_size_query"] = st.sidebar.slider("Batch Size (Query)", 8, 64, 16, 8)
    param["nb_epochs"] = st.sidebar.slider("Training Epochs", 1, 10, 3)
    param["nb_stolen"] = st.sidebar.slider("Number of Samples to Steal", 500, 3000, 1500, 250)
    param["use_probability"] = st.sidebar.checkbox("Use Probability Output", value=True)
    param["sampling_strategy"] = st.sidebar.selectbox("Sampling Strategy", ["random", "adaptive"])
    param["reward"] = st.sidebar.selectbox("Reward Strategy", ["cert", "div", "loss","all"])

run_button = st.button("🚀 Run Attack", type="primary")

if run_button:
    nb_stolen = clamp_nb_stolen(param.get("nb_stolen", 500), len(test_images))

    if attack_type == "CopyCatCNN":
        with st.spinner("⏳ Running " + attack_type + " attack... Please wait"):
            if steal_dataset == "MNIST Test Set":
                x_steal = test_images[:nb_stolen]
            else:
                x_steal = load_external_dataset(steal_dataset, nb_stolen)
            attack = CopycatCNN(classifier, batch_size_fit=param["batch_size_fit"],
                                batch_size_query=param["batch_size_query"],
                                nb_epochs=param["nb_epochs"],
                                use_probability=param["use_probability"],
                                nb_stolen=nb_stolen)
            stolen_model = get_model(10)
            classifier_stolen = KerasClassifier(stolen_model, clip_values=(0, 1))
            classifier_stolen = attack.extract(thieved_classifier=classifier_stolen, x=x_steal)
            y_test_cat = to_categorical(test_labels[nb_stolen:], 10)
            loss_org, acc_org = classifier.model.evaluate(classifier.model, test_images[nb_stolen:], test_labels[nb_stolen:])
            loss, acc = classifier_stolen._model.evaluate(classifier_stolen._model, test_images[nb_stolen:], y_test_cat)
            org_pred = classifier.predict(test_images[nb_stolen:])
            stol_pred = classifier_stolen.predict(test_images[nb_stolen:])
            if len(org_pred.shape) > 1:
                original_classes = np.argmax(org_pred, axis=1)
            else:
                original_classes = org_pred
            if len(stol_pred.shape) > 1:
                stolen_classes = np.argmax(stol_pred, axis=1)
            else:
                stolen_classes = stol_pred
            fidelity = np.mean(original_classes == stolen_classes)
            col1, col2, col3 = st.columns(3)
            col1.metric("Original Accuracy", f"{acc_org:.3f}", f"{acc_org * 100:.1f}%")
            col2.metric("Stolen Accuracy", f"{acc:.3f}", f"{acc * 100:.1f}%")
            col3.metric("Fidelity", f"{fidelity:.3f}", f"{fidelity * 100:.1f}%")


    elif attack_type == "Functionally Equivalent Extraction":
        with st.spinner("⏳ Running " + attack_type + " attack... Please wait"):
            train_images_flat = train_images.reshape(train_images.shape[0], -1)
            test_images_flat = test_images.reshape(test_images.shape[0], -1)
            target_model = get_model_FEE()
            classifier_target = KerasClassifier(target_model, clip_values=(0, 1))
            attack = FunctionallyEquivalentExtraction(classifier_target, num_neurons=param["num_neurons"])
            stolen_model = get_model_FEE()
            classifier_stolen = KerasClassifier(stolen_model, clip_values=(0, 1))
            classifier_stolen = attack.extract(thieved_classifier=classifier_stolen, x=train_images_flat[:nb_stolen],
                                               y=None, delta_0=param["delta_0"],
                                               fraction_true=param["fraction_true"],
                                               rel_diff_slope=param["rel_diff_slope"],
                                               rel_diff_value=param["rel_diff_value"],
                                               delta_init_value=param["delta_init_value"],
                                               delta_value_max=param["delta_value_max"]
                                               )
            y_test_cat = to_categorical(test_labels[:nb_stolen], 10)
            loss_org, acc_org = classifier._model.evaluate(classifier_target.model, test_images_flat[:nb_stolen],
                                              test_labels[:nb_stolen])
            loss, acc = classifier_stolen._model.evaluate(classifier_stolen._model, test_images_flat[:nb_stolen], y_test_cat)
            st.write(f"Original Accuracy: {acc_org:.3f}, Stolen Accuracy: {acc:.3f}")



    elif attack_type == "Knockoff Nets":
        with st.spinner("⏳ Running " + attack_type + " attack... Please wait"):
            if steal_dataset == "MNIST Test Set":
                x_steal = test_images[:nb_stolen]
                y_steal = to_categorical(test_labels[:nb_stolen], 10)
            else:
                x_steal = load_external_dataset(steal_dataset, nb_stolen)
                y_steal = np.zeros((len(x_steal), 10))
            model_to_use = get_model_lightweight(10) if param.get("use_lightweight_model", True) else get_model(10)
            classifier_stolen = KerasClassifier(model_to_use, clip_values=(0, 1))
            attack = KnockoffNets(classifier, batch_size_fit=param["batch_size_fit"],
                                  batch_size_query=param["batch_size_query"],
                                  nb_epochs=param["nb_epochs"],
                                  sampling_strategy=param.get("sampling_strategy", "random"),
                                  reward=param.get("reward", "all"),
                                  use_probability=param.get("use_probability", True))

            classifier_stolen = process_knockoff_in_batches(attack, classifier_stolen, x_steal, y_steal)

            y_test_cat = to_categorical(test_labels[nb_stolen:], 10)
            loss_org, acc_org = classifier._model.evaluate(classifier.model, test_images[nb_stolen:], test_labels[nb_stolen:])
            loss, acc = classifier_stolen._model.evaluate(classifier_stolen._model, test_images[nb_stolen:], y_test_cat)
            org_pred = classifier.predict(test_images[nb_stolen:])
            stol_pred = classifier_stolen.predict(test_images[nb_stolen:])
            fidelity = np.mean(np.argmax(org_pred, axis=1) == np.argmax(stol_pred, axis=1))
            col1, col2, col3 = st.columns(3)
            col1.metric("Original Accuracy", f"{acc_org:.3f}", f"{acc_org * 100:.1f}%")
            col2.metric("Stolen Accuracy", f"{acc:.3f}", f"{acc * 100:.1f}%")
            col3.metric("Fidelity", f"{fidelity:.3f}", f"{fidelity * 100:.1f}%")

