# pages/Extraction.py
import streamlit as st
import os
import gc
import sys
import traceback

# Environment before TF import
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.datasets import mnist, fashion_mnist, cifar10
from art.utils import to_categorical
from art.estimators.classification import KerasClassifier
from art.attacks.extraction import CopycatCNN, FunctionallyEquivalentExtraction, KnockoffNets

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Extraction Attacks on MNIST (Low-Mem)", layout="wide")
st.title("🔓 Extraction Attacks on MNIST — Memory-friendly")
st.markdown("Use this version for low-RAM Streamlit servers. Heavy experiments should run on Colab or a beefy VM.")

# ------------- Helpers -------------
def clear_memory():
    try:
        K.clear_session()
    except Exception:
        pass
    gc.collect()

def try_load_h5(path="pages/mnist_model.h5"):
    """Try loading H5 with safe fallbacks."""
    if not os.path.exists(path):
        return None, f"File not found: {path}"

    try:
        # try loading with compile disabled (safer)
        model = load_model(path, compile=False, safe_mode=False)
        return model, "loaded"
    except Exception as e:
        # return None and the error text for UI
        return None, f"Failed to load H5: {str(e)}"

@st.cache_resource
def load_or_build_mnist_model(h5path="pages/mnist_model.h5"):
    # Attempt load
    model, msg = try_load_h5(h5path)
    if model is not None:
        return model

    # Fallback: build a tiny model and (optionally) save it
    (x_train, y_train), _ = mnist.load_data()
    x_train = (x_train / 255.0).reshape(-1, 28, 28, 1)[:2000]
    y_train = y_train[:2000]

    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    try:
        model.fit(x_train, y_train, epochs=1, batch_size=64, verbose=0)
        # Try saving (best-effort)
        try:
            os.makedirs(os.path.dirname(h5path), exist_ok=True)
            model.save(h5path)
        except Exception:
            pass
    except Exception:
        # training may fail on small memory — ignore
        pass
    return model

@st.cache_data
def load_data_small(max_train=2000, max_test=1000):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images[:max_train] / 255.0
    test_images = test_images[:max_test] / 255.0
    train_images = train_images.reshape(-1,28,28,1)
    test_images = test_images.reshape(-1,28,28,1)
    train_labels = train_labels[:max_train]
    test_labels = test_labels[:max_test]
    return (train_images, train_labels), (test_images, test_labels)

@st.cache_data
def load_external_dataset_small(dataset_name, max_samples=1000):
    if dataset_name == "CIFAR-10":
        (x_train, _), (x_test, _) = cifar10.load_data()
        x_combined = np.concatenate([x_train, x_test])[:max_samples]
        x_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_combined])
        x_resized = np.array([cv2.resize(img, (28,28)) for img in x_gray])
        return x_resized.reshape(-1,28,28,1) / 255.0
    if dataset_name == "Fashion-MNIST":
        (x_train, _), (x_test, _) = fashion_mnist.load_data()
        x_combined = np.concatenate([x_train, x_test])[:max_samples]
        return x_combined.reshape(-1,28,28,1) / 255.0
    return None

def tiny_cnn(num_classes=10):
    clear_memory()
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def tiny_dense_FEE():
    clear_memory()
    model = Sequential([Dense(64, activation='relu', input_shape=(784,)), Dense(10, activation='linear')])
    return model

# ------------- Load model & data -------------
if 'data_loaded' not in st.session_state:
    with st.spinner("Loading model and small dataset..."):
        model = load_or_build_mnist_model()
        (train_images, train_labels), (test_images, test_labels) = load_data_small()
        classifier = KerasClassifier(model=model, clip_values=(0,1))
        st.session_state.model = model
        st.session_state.classifier = classifier
        st.session_state.train_data = (train_images, train_labels)
        st.session_state.test_data = (test_images, test_labels)
        st.session_state.data_loaded = True

# expose session vars
classifier = st.session_state.classifier
train_images, train_labels = st.session_state.train_data
test_images, test_labels = st.session_state.test_data
model = st.session_state.model

# ------------- Sidebar config (memory-friendly defaults) -------------
st.sidebar.header("⚔️ Attack Configuration (Low-Mem Defaults)")
options = ["CopyCatCNN", "Functionally Equivalent Extraction", "Knockoff Nets"]
attack_type = st.sidebar.selectbox("Select Attack", options)

param = {}
if attack_type == "CopyCatCNN":
    st.sidebar.subheader("CopyCatCNN parameters")
    steal_dataset = st.sidebar.selectbox("Steal dataset", ["MNIST Test Set","CIFAR-10","Fashion-MNIST"])
    param["nb_stolen"] = st.sidebar.slider("Samples to steal", 100, 2000, 500, step=100)  # default 500
    param["batch_size_query"] = st.sidebar.slider("Batch size (query)", 8, 64, 16)
    param["batch_size_fit"] = st.sidebar.slider("Batch size (train)", 8, 64, 32)
    param["nb_epochs"] = st.sidebar.slider("Training epochs (stolen model)", 1, 5, 1)
    param["use_probability"] = st.sidebar.checkbox("Use probability outputs", value=True)

elif attack_type == "Functionally Equivalent Extraction":
    st.sidebar.subheader("FEE (Warning: heavy)")
    st.sidebar.warning("FEE can be very heavy. Prefer to run on Colab/VM. Here we use tiny defaults.")
    param["num_neurons"] = st.sidebar.number_input("Neurons (tiny)", 32, 256, 64)
    param["quick_mode"] = st.sidebar.checkbox("Quick mode (1 epoch, small dataset)", True)

elif attack_type == "Knockoff Nets":
    st.sidebar.subheader("Knockoff Nets (memory-limited)")
    steal_dataset = st.sidebar.selectbox("Steal dataset", ["MNIST Test Set","CIFAR-10","Fashion-MNIST"])
    param["nb_stolen"] = st.sidebar.slider("Samples to steal", 100, 2000, 500, step=100)
    param["batch_size_query"] = st.sidebar.slider("Batch size (query)", 8, 64, 16)
    param["batch_size_fit"] = st.sidebar.slider("Batch size (train)", 8, 64, 32)
    param["nb_epochs"] = st.sidebar.slider("Training epochs (stolen model)", 1, 5, 1)
    param["use_probability"] = st.sidebar.checkbox("Use probability outputs", value=True)
    param["sampling_strategy"] = st.sidebar.selectbox("Sampling strategy", ["random","adaptive"])
    param["reward"] = st.sidebar.selectbox("Reward", ["cert","div","loss","all"])

# ------------- Run button -------------
run = st.button("🚀 Run Attack (memory-friendly)")

def shape_check(x, expected_shape=(28,28,1)):
    return x.ndim == 4 and x.shape[1:]==expected_shape

if run:
    clear_memory()
    try:
        if attack_type == "CopyCatCNN":
            nb = int(param["nb_stolen"])
            if steal_dataset == "MNIST Test Set":
                x_steal = test_images[:nb]
            else:
                x_steal = load_external_dataset_small(steal_dataset, max_samples=nb)
            st.write("x_steal shape:", None if x_steal is None else x_steal.shape)

            if x_steal is None or not shape_check(x_steal):
                st.error("Stealing dataset shape mismatch or not found. Use MNIST Test Set or small external dataset.")
            else:
                attack = CopycatCNN(
                    classifier,
                    batch_size_fit=param["batch_size_fit"],
                    batch_size_query=param["batch_size_query"],
                    nb_epochs=param["nb_epochs"],
                    use_probability=param["use_probability"],
                    nb_stolen=nb
                )
                st.info("Running CopyCatCNN on small dataset — this may still use CPU heavily.")
                with st.spinner("Extracting..."):
                    stolen_model = tiny_cnn(10)
                    classifier_stolen = KerasClassifier(stolen_model, clip_values=(0,1))
                    classifier_stolen = attack.extract(thieved_classifier=classifier_stolen, x=x_steal)
                # Evaluate (use small slice)
                test_slice = min(200, test_images.shape[0]-nb)
                y_test_cat = to_categorical(test_labels[nb:nb+test_slice], nb_classes=10)
                _, acc_stolen = classifier_stolen._model.evaluate(test_images[nb:nb+test_slice], y_test_cat, verbose=0)
                _, acc_orig = classifier.model.evaluate(test_images[nb:nb+test_slice], test_labels[nb:nb+test_slice], verbose=0)
                org_pred = np.argmax(classifier.predict(test_images[nb:nb+test_slice]), axis=1)
                stol_pred = np.argmax(classifier_stolen.predict(test_images[nb:nb+test_slice]), axis=1)
                fidelity = np.mean(org_pred == stol_pred)
                st.success("Done — small evaluation:")
                st.metric("Original acc", f"{acc_orig:.3f}")
                st.metric("Stolen acc", f"{acc_stolen:.3f}")
                st.metric("Fidelity", f"{fidelity:.3f}")

        elif attack_type == "Functionally Equivalent Extraction":
            st.info("FEE: running in QUIRK mode (tiny). For real experiments run on Colab/VM.")
            # flatten
            train_flat = train_images.reshape(train_images.shape[0], -1)
            test_flat = test_images.reshape(test_images.shape[0], -1)
            target_model = tiny_dense_FEE()
            try:
                target_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
                epochs = 1 if param.get("quick_mode", True) else 3
                target_model.fit(train_flat, train_labels, epochs=epochs, batch_size=64, verbose=0)
            except Exception as e:
                st.error("Training FEE target model failed on this server — try Colab.")
                st.exception(e)
                raise e
            classifier_fee = KerasClassifier(target_model, clip_values=(0,1), use_logits=True)
            attack = FunctionallyEquivalentExtraction(classifier_fee, num_neurons=param.get("num_neurons",64))
            st.info("Running FEE extraction (tiny)...")
            with st.spinner("Extracting..."):
                stolen_classifier = attack.extract(test_flat[:200], test_labels[:200])
            _, acc_stolen = stolen_classifier.model.evaluate(test_flat[:200], test_labels[:200], verbose=0)
            _, acc_orig = target_model.evaluate(test_flat[:200], test_labels[:200], verbose=0)
            st.success("FEE tiny done")
            st.metric("Original acc", f"{acc_orig:.3f}")
            st.metric("Stolen acc", f"{acc_stolen:.3f}")

        elif attack_type == "Knockoff Nets":
            nb = int(param["nb_stolen"])
            if steal_dataset == "MNIST Test Set":
                x_steal = test_images[:nb]
            else:
                x_steal = load_external_dataset_small(steal_dataset, max_samples=nb)
            st.write("x_steal shape:", None if x_steal is None else x_steal.shape)
            if x_steal is None or not shape_check(x_steal):
                st.error("Stealing dataset shape mismatch or not found. Use MNIST Test Set or small external dataset.")
            else:
                attack = KnockoffNets(
                    classifier,
                    batch_size_fit=param["batch_size_fit"],
                    batch_size_query=param["batch_size_query"],
                    nb_epochs=param["nb_epochs"],
                    use_probability=param["use_probability"],
                    nb_stolen=nb,
                    sampling_strategy=param["sampling_strategy"],
                    reward=param["reward"]
                )
                st.info("Running Knockoff (tiny). This still uses CPU.")
                with st.spinner("Extracting..."):
                    stolen_model = tiny_cnn(10)
                    classifier_stolen = KerasClassifier(stolen_model, clip_values=(0,1))
                    y_steal = classifier.predict(x_steal)
                    classifier_stolen = attack.extract(thieved_classifier=classifier_stolen, x=x_steal, y=y_steal)
                test_slice = min(200, test_images.shape[0]-nb)
                y_test_cat = to_categorical(test_labels[nb:nb+test_slice], nb_classes=10)
                _, acc_stolen = classifier_stolen._model.evaluate(test_images[nb:nb+test_slice], y_test_cat, verbose=0)
                _, acc_orig = classifier.model.evaluate(test_images[nb:nb+test_slice], test_labels[nb:nb+test_slice], verbose=0)
                org_pred = np.argmax(classifier.predict(test_images[nb:nb+test_slice]), axis=1)
                stol_pred = np.argmax(classifier_stolen.predict(test_images[nb:nb+test_slice]), axis=1)
                fidelity = np.mean(org_pred == stol_pred)
                st.success("Knockoff tiny done")
                st.metric("Original acc", f"{acc_orig:.3f}")
                st.metric("Stolen acc", f"{acc_stolen:.3f}")
                st.metric("Fidelity", f"{fidelity:.3f}")

    except Exception as e:
        st.error("Runtime error during attack. Streamlit servers often have low RAM — try running the heavy job on Colab or use smaller parameters.")
        st.exception(traceback.format_exc())
    finally:
        clear_memory()

# ------------- Footer advice -------------
st.markdown("---")
st.write("Notes:")
st.write("- This page is tuned for low-memory execution. For real research-scale experiments, run on Colab/VM and then import the saved models/results here.")
st.write("- Recommended: run Streamlit with Python 3.10 or 3.11 for best TensorFlow compatibility.")
