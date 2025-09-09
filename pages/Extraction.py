import streamlit as st
import os
import gc
import psutil
import uuid
from threading import Lock
import time

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

# Global lock for thread safety
global_lock = Lock()

# Memory optimization settings
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    st.warning(f"GPU configuration warning: {e}")

st.set_page_config(
    page_title="Extraction Attacks on MNIST",
    page_icon="🔓",
    layout="wide"
)

# Generate unique session ID for each user
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

st.title(f"🔓 Extraction Attacks ")
st.markdown("---")


# --- Memory & helper functions ---
def get_memory_usage():
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    except:
        return 0


def clear_memory():
    try:
        gc.collect()
        # Don't clear session globally - only clear current session's models
        if hasattr(st.session_state, 'temp_models'):
            for model in st.session_state.temp_models:
                try:
                    del model
                except:
                    pass
            st.session_state.temp_models = []
    except Exception as e:
        st.warning(f"Memory clear warning: {e}")


def clamp_nb_stolen(nb_stolen, total_len):
    if nb_stolen >= total_len:
        st.warning(
            f"Requested nb_stolen ({nb_stolen}) >= available test samples ({total_len}). Clamping to leave at least 1 sample.")
        nb_stolen = max(0, total_len - 1)
    return nb_stolen


@st.cache_resource
def load_mnist_model():
    try:
        return load_model("pages/mnist_model.h5")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_data():
    try:
        with global_lock:  # Thread-safe data loading
            (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
            max_train_size, max_test_size = 10000, 5000
            train_images, train_labels = train_images[:max_train_size], train_labels[:max_train_size]
            test_images, test_labels = test_images[:max_test_size], test_labels[:max_test_size]
            train_images, test_images = train_images / 255.0, test_images / 255.0
            train_images = train_images.reshape(-1, 28, 28, 1)
            test_images = test_images.reshape(-1, 28, 28, 1)
            return (train_images, train_labels), (test_images, test_labels)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


@st.cache_data
def load_external_dataset(dataset_name, max_samples=5000):
    try:
        with global_lock:  # Thread-safe dataset loading
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
    except Exception as e:
        st.error(f"Error loading external dataset: {e}")
        return None


def get_model(NUM_CLASSES, session_id=None):
    try:
        # Create session-specific model to avoid conflicts
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(NUM_CLASSES, activation='softmax')
        ], name=f"model_{session_id}_{int(time.time())}")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Track models for cleanup
        if 'temp_models' not in st.session_state:
            st.session_state.temp_models = []
        st.session_state.temp_models.append(model)

        return model
    except Exception as e:
        st.error(f"Error creating model: {e}")
        return None


def get_model_lightweight(NUM_CLASSES, session_id=None):
    try:
        model = Sequential([
            Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(NUM_CLASSES, activation='softmax')
        ], name=f"lightweight_model_{session_id}_{int(time.time())}")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Track models for cleanup
        if 'temp_models' not in st.session_state:
            st.session_state.temp_models = []
        st.session_state.temp_models.append(model)

        return model
    except Exception as e:
        st.error(f"Error creating lightweight model: {e}")
        return None


def get_model_FEE(session_id=None):
    try:
        model = Sequential([
            Dense(128, activation="relu", input_shape=(784,)),
            Dense(10, activation="linear")
        ], name=f"fee_model_{session_id}_{int(time.time())}")

        # Track models for cleanup
        if 'temp_models' not in st.session_state:
            st.session_state.temp_models = []
        st.session_state.temp_models.append(model)

        return model
    except Exception as e:
        st.error(f"Error creating FEE model: {e}")
        return None


def process_knockoff_in_batches(attack, classifier_stolen, x_steal, y_steal, batch_size=500):
    try:
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

            # Clear memory periodically
            if i % 5 == 0:
                clear_memory()

        progress_bar.empty()
        status_text.empty()
        return classifier_stolen
    except Exception as e:
        st.error(f"Error in batch processing: {e}")
        return classifier_stolen


# --- Load model & data with better error handling ---
def initialize_session():
    if 'data_loaded' not in st.session_state:
        with st.spinner("Loading model and data..."):
            try:
                model = load_mnist_model()
                if model is None:
                    st.error("Failed to load model. Please ensure the model file exists.")
                    st.stop()

                data_result = load_data()
                if data_result[0] is None:
                    st.error("Failed to load data.")
                    st.stop()

                (train_images, train_labels), (test_images, test_labels) = data_result
                classifier = KerasClassifier(model=model, clip_values=(0, 1))

                st.session_state.model = model
                st.session_state.classifier = classifier
                st.session_state.train_data = (train_images, train_labels)
                st.session_state.test_data = (test_images, test_labels)
                st.session_state.data_loaded = True
                st.session_state.temp_models = []

                

            except Exception as e:
                st.error(f"Initialization failed: {e}")
                st.stop()


# Initialize session
initialize_session()

# Get session data
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
    param["nb_stolen"] = st.sidebar.slider("Number of Samples to Steal", 1000, 5000, 2500, 500)
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
    param["nb_stolen"] = st.sidebar.slider("Number of Samples to Steal", 500, 2000, 1000, 250)
    param["use_probability"] = st.sidebar.checkbox("Use Probability Output", value=True)
    param["sampling_strategy"] = st.sidebar.selectbox("Sampling Strategy", ["random", "adaptive"])
    param["reward"] = st.sidebar.selectbox("Reward Strategy", ["cert", "div", "loss", "all"])

run_button = st.button("🚀 Run Attack", type="primary")

if run_button:
    if 'attack_running' in st.session_state and st.session_state.attack_running:
        st.warning("An attack is already running in this session. Please wait for it to complete.")
    else:
        st.session_state.attack_running = True

        try:
            clear_memory()
            nb_stolen = clamp_nb_stolen(param.get("nb_stolen", 500), len(test_images))

            if attack_type == "CopyCatCNN":
                with st.spinner("⏳ Running " + attack_type + " attack... Please wait"):
                    if steal_dataset == "MNIST Test Set":
                        x_steal = test_images[:nb_stolen]
                    else:
                        x_steal = load_external_dataset(steal_dataset, nb_stolen)
                        if x_steal is None:
                            st.error("Failed to load external dataset")
                            st.session_state.attack_running = False
                            st.stop()

                    attack = CopycatCNN(classifier,
                                        batch_size_fit=param["batch_size_fit"],
                                        batch_size_query=param["batch_size_query"],
                                        nb_epochs=param["nb_epochs"],
                                        use_probability=param["use_probability"],
                                        nb_stolen=nb_stolen)

                    stolen_model = get_model(10, st.session_state.session_id)
                    if stolen_model is None:
                        st.error("Failed to create stolen model")
                        st.session_state.attack_running = False
                        st.stop()

                    classifier_stolen = KerasClassifier(stolen_model, clip_values=(0, 1))
                    classifier_stolen = attack.extract(thieved_classifier=classifier_stolen, x=x_steal)

                    # Evaluation
                    test_subset = test_images[nb_stolen:]
                    test_labels_subset = test_labels[nb_stolen:]
                    y_test_cat = to_categorical(test_labels_subset, 10)

                    loss_org, acc_org = classifier.model.evaluate(test_subset, test_labels_subset, verbose=0)
                    loss, acc = classifier_stolen._model.evaluate(test_subset, y_test_cat, verbose=0)

                    org_pred = classifier.predict(test_subset)
                    stol_pred = classifier_stolen.predict(test_subset)

                    original_classes = np.argmax(org_pred, axis=1) if len(org_pred.shape) > 1 else org_pred
                    stolen_classes = np.argmax(stol_pred, axis=1) if len(stol_pred.shape) > 1 else stol_pred
                    fidelity = np.mean(original_classes == stolen_classes)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Original Accuracy", f"{acc_org:.3f}", f"{acc_org * 100:.1f}%")
                    col2.metric("Stolen Accuracy", f"{acc:.3f}", f"{acc * 100:.1f}%")
                    col3.metric("Fidelity", f"{fidelity:.3f}", f"{fidelity * 100:.1f}%")

            elif attack_type == "Functionally Equivalent Extraction":
                with st.spinner("⏳ Running " + attack_type + " attack... Please wait"):
                    train_images_flat = train_images.reshape(train_images.shape[0], -1)
                    test_images_flat = test_images.reshape(test_images.shape[0], -1)

                    target_model = get_model_FEE(st.session_state.session_id)
                    if target_model is None:
                        st.error("Failed to create target model")
                        st.session_state.attack_running = False
                        st.stop()

                    classifier_target = KerasClassifier(target_model, clip_values=(0, 1))
                    attack = FunctionallyEquivalentExtraction(classifier_target, num_neurons=param["num_neurons"])

                    stolen_model = get_model_FEE(st.session_state.session_id)
                    classifier_stolen = KerasClassifier(stolen_model, clip_values=(0, 1))

                    classifier_stolen = attack.extract(
                        thieved_classifier=classifier_stolen,
                        x=train_images_flat[:param.get("nb_stolen", 1000)],
                        y=None,
                        delta_0=param["delta_0"],
                        fraction_true=param["fraction_true"],
                        rel_diff_slope=param["rel_diff_slope"],
                        rel_diff_value=param["rel_diff_value"],
                        delta_init_value=param["delta_init_value"],
                        delta_value_max=param["delta_value_max"]
                    )


                    test_subset = test_images_flat[:1000]
                    test_labels_subset = test_labels[:1000]


                    loss_org, acc_org = classifier.model.evaluate(test_images[:1000], test_labels_subset, verbose=0)


                    org_pred = classifier.predict(test_images[:1000])
                    stol_pred = classifier_stolen.predict(test_subset)

                    if len(org_pred.shape) > 1:
                        original_classes = np.argmax(org_pred, axis=1)
                    else:
                        original_classes = org_pred

                    if len(stol_pred.shape) > 1:
                        stolen_classes = np.argmax(stol_pred, axis=1)
                    else:
                        stolen_classes = stol_pred

                    fidelity = np.mean(original_classes == stolen_classes)

                    st.success("✅ Functionally Equivalent Extraction completed!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Original Accuracy", f"{acc_org:.3f}", f"{acc_org * 100:.1f}%")
                    with col2:
                        st.metric("Fidelity", f"{fidelity:.3f}", f"{fidelity * 100:.1f}%")


            elif attack_type == "Knockoff Nets":
                with st.spinner("⏳ Running " + attack_type + " attack... Please wait"):
                    if steal_dataset == "MNIST Test Set":
                        x_steal = test_images[:nb_stolen]
                        y_steal = to_categorical(test_labels[:nb_stolen], 10)
                    else:
                        x_steal = load_external_dataset(steal_dataset, nb_stolen)
                        if x_steal is None:
                            st.error("Failed to load external dataset")
                            st.session_state.attack_running = False
                            st.stop()
                        y_steal = np.zeros((len(x_steal), 10))

                    model_to_use = get_model_lightweight(10, st.session_state.session_id)
                    if model_to_use is None:
                        st.error("Failed to create model")
                        st.session_state.attack_running = False
                        st.stop()

                    classifier_stolen = KerasClassifier(model_to_use, clip_values=(0, 1))
                    attack = KnockoffNets(classifier,
                                          batch_size_fit=param["batch_size_fit"],
                                          batch_size_query=param["batch_size_query"],
                                          nb_epochs=param["nb_epochs"],
                                          sampling_strategy=param.get("sampling_strategy", "random"),
                                          reward=param.get("reward", "all"),
                                          use_probability=param.get("use_probability", True))

                    classifier_stolen = process_knockoff_in_batches(attack, classifier_stolen, x_steal, y_steal)

                    # Evaluation
                    test_subset = test_images[nb_stolen:]
                    test_labels_subset = test_labels[nb_stolen:]
                    y_test_cat = to_categorical(test_labels_subset, 10)

                    loss_org, acc_org = classifier._model.evaluate(test_subset, test_labels_subset, verbose=0)
                    loss, acc = classifier_stolen._model.evaluate(test_subset, y_test_cat, verbose=0)

                    org_pred = classifier.predict(test_subset)
                    stol_pred = classifier_stolen.predict(test_subset)
                    fidelity = np.mean(np.argmax(org_pred, axis=1) == np.argmax(stol_pred, axis=1))

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Original Accuracy", f"{acc_org:.3f}", f"{acc_org * 100:.1f}%")
                    col2.metric("Stolen Accuracy", f"{acc:.3f}", f"{acc * 100:.1f}%")
                    col3.metric("Fidelity", f"{fidelity:.3f}", f"{fidelity * 100:.1f}%")

        except Exception as e:
            st.error(f"Attack failed: {str(e)}")
            st.error("Please try again with different parameters or contact support.")

        finally:
            # Always reset the attack running flag
            st.session_state.attack_running = False
            clear_memory()

