import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import types

from keras.datasets import mnist, fashion_mnist, cifar10
from keras.models import load_model as keras_load_model
from art.attacks.extraction import KnockoffNets
from art.estimators.classification import KerasClassifier

# ===============================
#  App Configuration
# ===============================
st.set_page_config(page_title="Model Extraction Attack", layout="wide")

# ===============================
#  Cached model loader
# ===============================
@st.cache_resource
def load_mnist_model():
    model = keras_load_model("pages/mnist_model.h5")

    # 🔥 Patch __call__ so ART works with Keras 3 (ignore unsupported verbose arg)
    def patched_call(self, inputs, training=False, **kwargs):
        return tf.keras.Model.__call__(self, inputs, training=training)

    model.__call__ = types.MethodType(patched_call, model)
    return model

# ===============================
#  Dataset loader
# ===============================
def load_data(dataset_name):
    if dataset_name == "MNIST Test Set":
        (_, _), (x_test, y_test) = mnist.load_data()
        x_test = x_test.astype("float32") / 255.0
        x_test = np.expand_dims(x_test, -1)
        return x_test, y_test
    elif dataset_name == "Fashion-MNIST":
        (_, _), (x_test, y_test) = fashion_mnist.load_data()
        x_test = x_test.astype("float32") / 255.0
        x_test = np.expand_dims(x_test, -1)
        return x_test, y_test
    elif dataset_name == "CIFAR-10":
        (_, _), (x_test, y_test) = cifar10.load_data()
        x_test = x_test.astype("float32") / 255.0
        x_test = tf.image.rgb_to_grayscale(x_test)
        x_test = tf.image.resize(x_test, [28, 28]).numpy()
        return x_test, y_test.flatten()
    else:
        st.error("Unknown dataset")
        return None, None

# ===============================
#  UI Layout
# ===============================
st.title("🛡️ Model Extraction Attack (KnockoffNets)")
st.markdown("Simulating a model extraction attack where an adversary trains a surrogate model to mimic a target model.")

# Initialize session state
if "data_loaded" not in st.session_state:
    with st.spinner("Loading model and data..."):
        model = load_mnist_model()
    st.session_state.data_loaded = True
    st.session_state.model = model

# Dataset selection
dataset_name = st.selectbox("Select dataset for adversary queries:", ["MNIST Test Set", "Fashion-MNIST", "CIFAR-10"])
x_steal, y_steal = load_data(dataset_name)

if x_steal is None:
    st.stop()

# Number of samples for the attack
num_samples = st.slider("Number of samples adversary can query:", 100, 5000, 1000, step=100)
x_steal_subset, y_steal_subset = x_steal[:num_samples], y_steal[:num_samples]

# ===============================
#  Run KnockoffNets Attack
# ===============================
if st.button("🚀 Run Extraction Attack"):
    with st.spinner("Running KnockoffNets... this may take some time"):
        target_model = st.session_state.model
        target_classifier = KerasClassifier(model=target_model, clip_values=(0, 1), use_logits=False)

        # Define and run the attack
        attack = KnockoffNets(
            target_classifier,
            batch_size=64,
            nb_epochs=3,
            use_probability=True,
        )

        surrogate_classifier = attack.extract(x_steal_subset, y_steal_subset)

        # Evaluate surrogate model accuracy
        _, (x_test, y_test) = mnist.load_data()
        x_test = x_test.astype("float32") / 255.0
        x_test = np.expand_dims(x_test, -1)

        surrogate_acc = np.mean(
            np.argmax(surrogate_classifier.predict(x_test), axis=1) == y_test
        )

        st.success(f"✅ Surrogate model trained with accuracy on MNIST test set: {surrogate_acc:.2f}")

        # ===============================
        #  Visualization
        # ===============================
        st.subheader("📊 Surrogate Model Predictions (first 10 samples)")
        fig, axes = plt.subplots(1, 10, figsize=(15, 3))
        preds = np.argmax(surrogate_classifier.predict(x_test[:10]), axis=1)

        for i, ax in enumerate(axes):
            ax.imshow(x_test[i].squeeze(), cmap="gray")
            ax.set_title(f"Pred: {preds[i]}\nTrue: {y_test[i]}")
            ax.axis("off")

        st.pyplot(fig)
