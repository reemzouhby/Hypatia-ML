import streamlit as st
import os
import gc

from art.utils import to_categorical
from sklearn.model_selection import train_test_split

# Disable GPU + TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras import backend as K
from keras.datasets import mnist, fashion_mnist, cifar10
import cv2
import tensorflow as tf
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from art.estimators.classification import KerasClassifier
from art.attacks.extraction import CopycatCNN, FunctionallyEquivalentExtraction, KnockoffNets


# ================== Streamlit UI ==================
st.set_page_config(
    page_title="Extraction Attacks on MNIST",
    page_icon="🔓",
    layout="wide"
)
st.title("🔓 Extraction Attacks on MNIST")
st.markdown("---")


# ================== Helpers ==================
@st.cache_resource
def load_mnist_model():
    """Load pretrained MNIST model"""
    model = load_model("pages/mnist_model.h5")
    return model


@st.cache_data
def load_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    max_train_size = 14000
    max_test_size = 7000

    train_images = train_images[:max_train_size] / 255.0
    test_images = test_images[:max_test_size] / 255.0

    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    train_labels = train_labels[:max_train_size]
    test_labels = test_labels[:max_test_size]

    return (train_images, train_labels), (test_images, test_labels)


@st.cache_data
def load_external_dataset(dataset_name, max_samples=10000):
    """Load and preprocess external datasets for stealing"""
    if dataset_name == "CIFAR-10":
        (x_train, _), (x_test, _) = cifar10.load_data()
        x_combined = np.concatenate([x_train, x_test])[:max_samples]
        x_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_combined])
        x_processed = np.array([cv2.resize(img, (28, 28)) for img in x_gray])
        x_processed = x_processed.reshape(-1, 28, 28, 1) / 255.0

    elif dataset_name == "Fashion-MNIST":
        (x_train, _), (x_test, _) = fashion_mnist.load_data()
        x_combined = np.concatenate([x_train, x_test])[:max_samples]
        x_processed = x_combined.reshape(-1, 28, 28, 1) / 255.0

    else:
        x_processed = None

    return x_processed


def get_model(num_classes):
    """Basic CNN for MNIST-like datasets"""
    tf.keras.backend.clear_session()
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model


def get_model_FEE():
    """Dense model for Functionally Equivalent Extraction"""
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=(784,)))
    model.add(Dense(10, activation="linear"))
    return model


def clear_memory():
    """Free up memory"""
    K.clear_session()
    gc.collect()


# ================== Load model & data ==================
if 'data_loaded' not in st.session_state:
    with st.spinner("Loading model and data..."):
        model = load_mnist_model()   # ✅ fixed here
        if model is not None:
            (train_images, train_labels), (test_images, test_labels) = load_data()
            classifier = KerasClassifier(model=model, clip_values=(0, 1))

            st.session_state.model = model
            st.session_state.classifier = classifier
            st.session_state.train_data = (train_images, train_labels)
            st.session_state.test_data = (test_images, test_labels)
            st.session_state.data_loaded = True
        else:
            st.error("❌ Failed to load model")
            st.stop()

# Get from session
classifier = st.session_state.classifier
train_images, train_labels = st.session_state.train_data
test_images, test_labels = st.session_state.test_data
model = st.session_state.model


# ================== Sidebar ==================
st.sidebar.header("⚔️ Attack Configuration")
options = ["CopyCatCNN", "Functionally Equivalent Extraction", "Knockoff Nets"]
attack_type = st.sidebar.selectbox("Select Attack", options)

param = {}
if attack_type == "CopyCatCNN":
    st.sidebar.subheader("🎯 CopyCatCNN Parameters")
    steal_dataset = st.sidebar.selectbox("Dataset", ["MNIST Test Set", "CIFAR-10", "Fashion-MNIST"])
    param["batch_size_fit"] = st.sidebar.slider("Batch Size (Training)", 16, 128, 64, step=16)
    param["batch_size_query"] = st.sidebar.slider("Batch Size (Query)", 16, 128, 64, step=16)
    param["nb_epochs"] = st.sidebar.slider("Training Epochs", 1, 20, 10)
    param["nb_stolen"] = st.sidebar.slider("Samples to Steal", 1000, 10000, 5000, step=500)
    param["use_probability"] = st.sidebar.checkbox("Use Probability Output", value=True)

elif attack_type == "Functionally Equivalent Extraction":
    st.sidebar.subheader("⚡ Functionally Equivalent Extraction Parameters")
    param["num_neurons"] = st.sidebar.number_input("Number of Neurons", 64, 512, 128, step=64)
    with st.sidebar.expander("🔧 Advanced Parameters"):
        param["delta_0"] = st.number_input("Delta 0", 0.001, 0.1, 0.05, step=0.001, format="%.3f")
        param["fraction_true"] = st.number_input("Fraction True", 0.1, 0.9, 0.3, step=0.1)
        param["rel_diff_slope"] = st.number_input("Relative Diff Slope", 1e-7, 1e-3, 1e-5, format="%.2e")
        param["rel_diff_value"] = st.number_input("Relative Diff Value", 1e-8, 1e-4, 1e-6, format="%.2e")
        param["delta_init_value"] = st.number_input("Delta Init Value", 0.01, 1.0, 0.1, step=0.01)
        param["delta_value_max"] = st.number_input("Delta Value Max", 10, 100, 50, step=10)

elif attack_type == "Knockoff Nets":
    st.sidebar.subheader("🎯 Knockoff Nets Parameters")
    steal_dataset = st.sidebar.selectbox("Dataset", ["MNIST Test Set", "CIFAR-10", "Fashion-MNIST"])
    param["batch_size_fit"] = st.sidebar.slider("Batch Size (Training)", 16, 128, 64, step=16)
    param["batch_size_query"] = st.sidebar.slider("Batch Size (Query)", 16, 128, 64, step=16)
    param["nb_epochs"] = st.sidebar.slider("Training Epochs", 1, 20, 10)
    param["nb_stolen"] = st.sidebar.slider("Samples to Steal", 1000, 10000, 5000, step=500)
    param["use_probability"] = st.sidebar.checkbox("Use Probability Output", value=True)
    param["sampling_strategy"] = st.selectbox("Sampling Strategy", ["random", "adaptive"], index=1)
    param["reward"] = st.selectbox("Reward Strategy", ["cert", "div", "loss", "all"], index=3)


# ================== Run Attack ==================
run_button = st.button("🚀 Run Attack", type="primary")
if run_button:
    clear_memory()

    if attack_type == "CopyCatCNN":
        with st.spinner("⏳ Running CopyCatCNN..."):
            nb_stolen = param["nb_stolen"]
            if steal_dataset == "MNIST Test Set":
                x_steal = test_images[:nb_stolen]
            else:
                x_steal = load_external_dataset(steal_dataset, nb_stolen)

            attack = CopycatCNN(
                classifier,
                batch_size_fit=param["batch_size_fit"],
                batch_size_query=param["batch_size_query"],
                nb_epochs=param["nb_epochs"],
                use_probability=param["use_probability"],
                nb_stolen=nb_stolen
            )
            stolen_model = get_model(10)
            classifier_stolen = KerasClassifier(stolen_model, clip_values=(0, 1))
            classifier_stolen = attack.extract(thieved_classifier=classifier_stolen, x=x_steal)

            y_test_cat = to_categorical(test_labels[nb_stolen:], nb_classes=10)
            _, acc_org = classifier.model.evaluate(test_images[nb_stolen:], test_labels[nb_stolen:])
            _, acc = classifier_stolen._model.evaluate(test_images[nb_stolen:], y_test_cat)

            org_pred = np.argmax(classifier.predict(test_images[nb_stolen:]), axis=1)
            stol_pred = np.argmax(classifier_stolen.predict(test_images[nb_stolen:]), axis=1)
            fidelity = np.mean(org_pred == stol_pred)

            st.success("✅ CopyCatCNN completed!")
            col1, col2, col3 = st.columns(3)
            col1.metric("Original Acc", f"{acc_org:.3f}")
            col2.metric("Stolen Acc", f"{acc:.3f}")
            col3.metric("Fidelity", f"{fidelity:.3f}")

    elif attack_type == "Functionally Equivalent Extraction":
        st.info("⚠️ Dense model will be trained for FEE attack.")
        train_flat = train_images.reshape(train_images.shape[0], -1)
        test_flat = test_images.reshape(test_images.shape[0], -1)

        target_model = get_model_FEE()
        target_model.compile(optimizer="adam",
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                             metrics=["accuracy"])
        target_model.fit(train_flat, train_labels, epochs=5)
        _, acc_org = target_model.evaluate(test_flat[:5000], test_labels[:5000])

        classifier_dense = KerasClassifier(target_model, clip_values=(0, 1), use_logits=True)
        attack = FunctionallyEquivalentExtraction(classifier_dense, num_neurons=param["num_neurons"])
        stolen_classifier = attack.extract(
            test_flat[5000:], test_labels[5000:],
            delta_0=param["delta_0"],
            fraction_true=param["fraction_true"],
            rel_diff_slope=param["rel_diff_slope"],
            rel_diff_value=param["rel_diff_value"],
            delta_init_value=param["delta_init_value"],
            delta_value_max=param["delta_value_max"]
        )

        _, acc_stolen = stolen_classifier.model.evaluate(test_flat[:5000], test_labels[:5000])
        org_pred = np.argmax(classifier_dense.predict(test_flat[:5000]), axis=1)
        stol_pred = np.argmax(stolen_classifier.predict(test_flat[:5000]), axis=1)
        fidelity = np.mean(org_pred == stol_pred)

        st.success("✅ Functionally Equivalent Extraction completed!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Original Acc", f"{acc_org:.3f}")
        col2.metric("Stolen Acc", f"{acc_stolen:.3f}")
        col3.metric("Fidelity", f"{fidelity:.3f}")

    elif attack_type == "Knockoff Nets":
        with st.spinner("⏳ Running Knockoff Nets..."):
            nb_stolen = param["nb_stolen"]
            if steal_dataset == "MNIST Test Set":
                x_steal = test_images[:nb_stolen]
            else:
                x_steal = load_external_dataset(steal_dataset, nb_stolen)

            attack = KnockoffNets(
                classifier,
                batch_size_fit=param["batch_size_fit"],
                batch_size_query=param["batch_size_query"],
                nb_epochs=param["nb_epochs"],
                use_probability=param["use_probability"],
                nb_stolen=nb_stolen,
                sampling_strategy=param["sampling_strategy"],
                reward=param["reward"]
            )
            stolen_model = get_model(10)
            classifier_stolen = KerasClassifier(stolen_model, clip_values=(0, 1))
            y_steal = classifier.predict(x_steal)
            classifier_stolen = attack.extract(thieved_classifier=classifier_stolen, x=x_steal, y=y_steal)

            y_test_cat = to_categorical(test_labels[nb_stolen:], nb_classes=10)
            _, acc = classifier_stolen._model.evaluate(test_images[nb_stolen:], y_test_cat)
            _, acc_org = classifier.model.evaluate(test_images[nb_stolen:], test_labels[nb_stolen:])

            org_pred = np.argmax(classifier.predict(test_images[nb_stolen:]), axis=1)
            stol_pred = np.argmax(classifier_stolen.predict(test_images[nb_stolen:]), axis=1)
            fidelity = np.mean(org_pred == stol_pred)

            st.success("✅ Knockoff Nets completed!")
            col1, col2, col3 = st.columns(3)
            col1.metric("Original Acc", f"{acc_org:.3f}")
            col2.metric("Stolen Acc", f"{acc:.3f}")
            col3.metric("Fidelity", f"{fidelity:.3f}")

    clear_memory()
