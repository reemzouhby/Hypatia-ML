import streamlit as st
import os
import gc
import psutil
import uuid
from threading import Lock, Semaphore
import time
import queue
import threading
from contextlib import contextmanager

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

# Global locks and semaphores for resource management
global_lock = Lock()
MAX_CONCURRENT_ATTACKS = 2  # Limit concurrent attacks
attack_semaphore = Semaphore(MAX_CONCURRENT_ATTACKS)
model_creation_lock = Lock()

# Memory optimization settings
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    st.warning(f"GPU configuration warning: {e}")

# Set TensorFlow to use less memory
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

st.set_page_config(
    page_title="Extraction Attacks ",
    page_icon="🔓",
    layout="wide"
)


def get_model_FEE():
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=(784,)))
    model.add(Dense(10, activation="linear"))
    return model


# Generate unique session ID for each user
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.session_state.session_created = time.time()

st.title(f"🔓 Extraction Attacks ")
st.markdown("---")


# --- Memory & helper functions ---
def get_memory_usage():
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    except:
        return 0


def get_system_resources():
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        return {
            'cpu': cpu_percent,
            'memory_total': memory.total / 1024 / 1024 / 1024,  # GB
            'memory_available': memory.available / 1024 / 1024 / 1024,  # GB
            'memory_percent': memory.percent
        }
    except:
        return {'cpu': 0, 'memory_total': 0, 'memory_available': 0, 'memory_percent': 0}


def aggressive_memory_cleanup():
    try:
        # Clear TensorFlow session
        K.clear_session()

        # Clear session-specific models
        if hasattr(st.session_state, 'temp_models'):
            for model in st.session_state.temp_models:
                try:
                    del model
                except:
                    pass
            st.session_state.temp_models = []

        # Force garbage collection
        gc.collect()

        # Clear any cached computations
        if hasattr(st.session_state, 'cached_predictions'):
            del st.session_state.cached_predictions

    except Exception as e:
        st.warning(f"Memory cleanup warning: {e}")


@contextmanager
def resource_manager():
    """Context manager for controlling resource usage"""
    acquired = False
    try:
        # Try to acquire semaphore with timeout
        acquired = attack_semaphore.acquire(timeout=300)  # 5 minute timeout
        if not acquired:
            raise TimeoutError("System is busy. Please try again later.")
        yield
    finally:
        if acquired:
            attack_semaphore.release()
            aggressive_memory_cleanup()


def clamp_nb_stolen(nb_stolen, total_len):
    if nb_stolen >= total_len:
        st.warning(
            f"Requested nb_stolen ({nb_stolen}) >= available test samples ({total_len}). Clamping to leave at least 1 sample.")
        nb_stolen = max(0, total_len - 1)
    return nb_stolen


def check_system_resources():
    """Check if system has enough resources"""
    resources = get_system_resources()

    if resources['memory_percent'] > 85:
        st.error("⚠️ System memory usage is too high (>85%). Please try again later.")
        return False

    if resources['cpu'] > 90:
        st.warning("⚠️ High CPU usage detected. Performance may be degraded.")

    return True


# --- Data & models with session isolation ---
@st.cache_resource
def load_mnist_model():
    try:
        model_path = "pages/mnist_model.h5"
        if not os.path.exists(model_path):
            model_path = "mnist_model.h5"
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            return None
        return load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_data():
    try:
        with global_lock:
            (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
            max_train_size, max_test_size = 5000, 2500
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
def load_external_dataset(dataset_name, max_samples=2500):  # Reduced default
    try:
        with global_lock:
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
        with model_creation_lock:
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(NUM_CLASSES, activation='softmax')
            ], name=f"simple_model_{session_id}_{int(time.time())}")

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            # Track models for cleanup
            if 'temp_models' not in st.session_state:
                st.session_state.temp_models = []
            st.session_state.temp_models.append(model)

            return model
    except Exception as e:
        st.error(f"Error creating model: {e}")
        return None


def get_model_FEE(session_id=None):
    try:
        with model_creation_lock:
            model = Sequential([
                Dense(64, activation="relu", input_shape=(784,)),  
                Dense(10, activation="linear")
            ], name=f"fee_model_{session_id}_{int(time.time())}")

            if 'temp_models' not in st.session_state:
                st.session_state.temp_models = []
            st.session_state.temp_models.append(model)

            return model
    except Exception as e:
        st.error(f"Error creating FEE model: {e}")
        return None


def process_attack_with_limits(attack_func, *args, **kwargs):
    """Process attack with resource monitoring"""
    try:
        with resource_manager():
            return attack_func(*args, **kwargs)
    except TimeoutError as e:
        st.error(f"⏱️ {str(e)}")
        return None
    except Exception as e:
        st.error(f"Attack failed: {str(e)}")
        return None


# --- Session initialization ---
def initialize_session():
    if 'data_loaded' not in st.session_state:
        if not check_system_resources():
            st.stop()

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
options = ["CopyCatCNN", "Knockoff Nets", "Functionally Equivalent Extraction"]
attack_type = st.sidebar.selectbox("Select Attack", options)

param = {}
if attack_type == "CopyCatCNN":
    steal_dataset = st.sidebar.selectbox("Dataset for Stealing", ["MNIST Test Set", "CIFAR-10", "Fashion-MNIST"])
    param["batch_size_fit"] = st.sidebar.slider("Batch Size (Training)", 16, 128, 32, 16)
    param["batch_size_query"] = st.sidebar.slider("Batch Size (Query)", 8, 64, 16, 8)
    param["nb_epochs"] = st.sidebar.slider("Training Epochs", 5, 20, 10)
    param["nb_stolen"] = st.sidebar.slider("Number of Samples to Steal", 500, 5000, 1000, 500)
    param["use_probability"] = st.sidebar.checkbox("Use Probability Output", value=True)
elif attack_type == "Functionally Equivalent Extraction":
    st.sidebar.subheader("⚡ Functionally Equivalent Extraction Parameters")
    st.sidebar.warning(
        "⚠️ **Research Note**: This attack can take approximately **4 days** to complete with optimal parameters. For researchers seeking perfect results, please plan accordingly.")

    # Core parameters
    param["num_neurons"] = st.sidebar.number_input("Number of Neurons", min_value=64, max_value=512, value=128, step=64)

    # Advanced parameters
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
    param["batch_size_fit"] = st.sidebar.slider("Batch Size (Training)", 16, 128, 32, 16)
    param["batch_size_query"] = st.sidebar.slider("Batch Size (Query)", 8, 64, 16, 8)
    param["nb_epochs"] = st.sidebar.slider("Training Epochs", 5, 20, 10)
    param["nb_stolen"] = st.sidebar.slider("Number of Samples to Steal", 250, 5000, 500, 500)
    param["use_probability"] = st.sidebar.checkbox("Use Probability Output", value=True)
    param["sampling_strategy"] = st.sidebar.selectbox("Sampling Strategy", ["random", "adaptive"])
    param["reward"] = st.sidebar.selectbox("Reward Strategy", ["cert", "div", "loss", "all"])

queue_position = MAX_CONCURRENT_ATTACKS - attack_semaphore._value + 1 if attack_semaphore._value == 0 else 0
if queue_position > 0:
    st.warning(f"⏳ You are #{queue_position} in queue. Please wait for your turn.")

run_button = st.button("🚀 Run Attack", type="primary", disabled=(queue_position > 3))

if run_button:
    if not check_system_resources():
        st.stop()

    if 'attack_running' in st.session_state and st.session_state.attack_running:
        st.warning("An attack is already running in this session.")
    else:
        st.session_state.attack_running = True

        try:
            aggressive_memory_cleanup()
            nb_stolen = clamp_nb_stolen(param.get("nb_stolen", 500), len(test_images))

            model_func = get_model

            if attack_type == "CopyCatCNN":
                with st.spinner("⏳ Running CopyCatCNN attack..."):
                    with resource_manager():
                        if steal_dataset == "MNIST Test Set":
                            x_steal = test_images[:nb_stolen]
                        else:
                            x_steal = load_external_dataset(steal_dataset, nb_stolen)
                            if x_steal is None:
                                st.error("Failed to load external dataset")
                                st.stop()

                        attack = CopycatCNN(classifier,
                                            batch_size_fit=param["batch_size_fit"],
                                            batch_size_query=param["batch_size_query"],
                                            nb_epochs=param["nb_epochs"],
                                            use_probability=param["use_probability"],
                                            nb_stolen=nb_stolen)

                        stolen_model = model_func(10, st.session_state.session_id)
                        if stolen_model is None:
                            st.error("Failed to create stolen model")
                            st.stop()

                        classifier_stolen = KerasClassifier(stolen_model, clip_values=(0, 1))
                        classifier_stolen = attack.extract(thieved_classifier=classifier_stolen, x=x_steal)

                        # Evaluation with smaller subset
                        eval_size = min(500, len(test_images) - nb_stolen)
                        test_subset = test_images[nb_stolen:nb_stolen + eval_size]
                        test_labels_subset = test_labels[nb_stolen:nb_stolen + eval_size]
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

            elif attack_type == "Knockoff Nets":
                with st.spinner("⏳ Running Knockoff Nets attack..."):
                    with resource_manager():
                        if steal_dataset == "MNIST Test Set":
                            x_steal = test_images[:nb_stolen]
                            y_steal = to_categorical(test_labels[:nb_stolen], 10)
                        else:
                            x_steal = load_external_dataset(steal_dataset, nb_stolen)
                            y_steal=classifier.predict(x_steal)
                            # Check if predictions are already probabilities or need conversion
                           if len(y_steal.shape) > 1 and y_steal.shape[1] == 10:
                                 # Already in probability format
                                  y_steal =y_steal
                           else:
                               # Convert to categorical if needed
                              y_steal = np.argmax(y_steal, axis=1) if len(y_steal.shape) > 1 else y_steal
                              y_steal = to_categorical(y_steal, 10)
                            if x_steal is None:
                                st.error("Failed to load external dataset")
                                st.stop()
                            

                        stolen_model = model_func(10, st.session_state.session_id)
                        if stolen_model is None:
                            st.error("Failed to create model")
                            st.stop()

                        classifier_stolen = KerasClassifier(stolen_model, clip_values=(0, 1))
                        attack = KnockoffNets(classifier,
                                              batch_size_fit=param["batch_size_fit"],
                                              batch_size_query=param["batch_size_query"],
                                              nb_epochs=param["nb_epochs"],
                                              sampling_strategy=param.get("sampling_strategy", "random"),
                                              reward=param.get("reward", "all"),
                                              use_probability=param.get("use_probability", True))

                        classifier_stolen = attack.extract(thieved_classifier=classifier_stolen, x=x_steal, y=y_steal)

                        # Evaluation
                        eval_size = min(500, len(test_images) - nb_stolen)
                        test_subset = test_images[nb_stolen:nb_stolen + eval_size]
                        test_labels_subset = test_labels[nb_stolen:nb_stolen + eval_size]
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
            elif attack_type == "Functionally Equivalent Extraction":
                with st.spinner("⏳ Running " + attack_type + " attack... Please wait"):
                    st.info(
                        "⚠️ Note: This attack requires a dense neural network model. Using pre-trained dense model.")
                    # flatt the images
                    train_images = train_images.reshape(train_images.shape[0], -1)
                    test_images = test_images.reshape(test_images.shape[0], -1)
                    target_model = get_model_FEE()
                    target_model.compile(optimizer="adam",
                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                         metrics=["accuracy"])
                    target_model.fit(train_images, train_labels, epochs=5)
                    loss, acc_org = target_model.evaluate(test_images[:5000], test_labels[:5000])
                    classifier = KerasClassifier(target_model, clip_values=(0, 1), use_logits=True)
                    attack = FunctionallyEquivalentExtraction(classifier, num_neurons=param["num_neurons"])
                    stolen_classifier = attack.extract(
                        test_images[5000:], test_labels[5000:],
                        delta_0=param["delta_0"],
                        fraction_true=param["fraction_true"],
                        rel_diff_slope=param["rel_diff_slope"],
                        rel_diff_value=param["rel_diff_value"],
                        delta_init_value=param["delta_init_value"],
                        delta_value_max=param["delta_value_max"]
                    )
                    loss, acc_stolen = stolen_classifier.model.evaluate(test_images[:5000], test_labels[:5000])
                    acc_drop = acc_org - acc_stolen
                    st.success("✅ Functionally Equivalent Extraction completed!")
                    org_pred = classifier.predict(test_images[:5000])
                    stol_pred = stolen_classifier.predict(test_images[:5000])
                    if len(org_pred.shape) > 1:  # If probability outputs
                        original_classes = np.argmax(org_pred, axis=1)
                    else:
                        original_classes = org_pred

                    if len(stol_pred.shape) > 1:  # If probability outputs
                        stolen_classes = np.argmax(stol_pred, axis=1)
                    else:
                        stolen_classes = stol_pred
                    fidelity = np.mean(original_classes == stolen_classes)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Accuracy", f"{acc_org:.3f}", f"{acc_org * 100:.1f}%")
                    with col2:
                        st.metric("Stolen Accuracy", f"{acc_stolen:.3f}", f"{acc_stolen * 100:.1f}%")
                    with col3:
                        st.metric("Fidelity", f"{fidelity}", f"{fidelity * 100:.1f}%")
                

        except Exception as e:
            st.error(f"Attack failed: {str(e)}")

        finally:
            st.session_state.attack_running = False
            aggressive_memory_cleanup()

