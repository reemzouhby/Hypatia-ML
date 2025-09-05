import  streamlit as st
import os

from art.utils import to_categorical
from sklearn.model_selection import train_test_split

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import backend as K
import warnings
from keras.datasets import mnist,fashion_mnist,cifar10,cifar100
import cv2
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import pandas as pd
from art.estimators.classification import KerasClassifier
from art.attacks.extraction import CopycatCNN, FunctionallyEquivalentExtraction, KnockoffNets

st.set_page_config(
    page_title="Extraction Attacks on MNIST",
    page_icon="🔓 ",
    layout="wide"
)
st.title(" 🔓  Extraction Attacks on MNIST")
st.markdown("---")
@st.cache_resource
def load_model():
    """Load model with caching"""
    try:
        model = tf.keras.models.load_model("Practice/Streamlitapp/pages/mnist_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_data():

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    max_train_size = 14000
    max_test_size = 7000

    train_images = train_images[:max_train_size]
    train_labels = train_labels[:max_train_size]
    test_images = test_images[:max_test_size]
    test_labels = test_labels[:max_test_size]

    # Normalize
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Reshape
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    return (train_images, train_labels), (test_images, test_labels)

if 'data_loaded' not in st.session_state:
    with st.spinner("Loading model and data..."):
        model = load_model()
        if model is not None:
            (train_images, train_labels), (test_images, test_labels) = load_data()
            classifier = KerasClassifier(model=model, clip_values=(0, 1))

            # Store in session state
            st.session_state.model = model
            st.session_state.classifier = classifier
            st.session_state.train_data = (train_images, train_labels)
            st.session_state.test_data = (test_images, test_labels)
            st.session_state.data_loaded = True

        else:
            st.error("❌ Failed to load model")
            st.stop()

def get_model_FEE():
    tf.keras.backend.clear_session()
    model=Sequential()
    model.add(Dense(128,activation="relu",input_shape=(784,)))
    model.add(Dense(10,activation="linear"))
    return model
@st.cache_data
def load_external_dataset(dataset_name, max_samples=10000):
    """Load and preprocess external datasets for stealing"""
    if dataset_name == "CIFAR-10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # Convert to grayscale and resize to 28x28
        x_combined = np.concatenate([x_train, x_test])[:max_samples]
        x_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_combined])
        x_processed = np.array([cv2.resize(img, (28, 28)) for img in x_gray])
        x_processed = x_processed.reshape(-1, 28, 28, 1) / 255.0

    elif dataset_name == "Fashion-MNIST":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_combined = np.concatenate([x_train, x_test])[:max_samples]
        x_processed = x_combined.reshape(-1, 28, 28, 1) / 255.0

    return x_processed


def get_model(NUM_CLASSES):
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model
# Get data from session state
classifier = st.session_state.classifier
train_images, train_labels = st.session_state.train_data
test_images, test_labels = st.session_state.test_data
model=st.session_state.model
st.sidebar.header("⚔️ Attack Configuration")
options = ["CopyCatCNN","Functionally Equivalent Extraction","Knockoff Nets"]
attack_type = st.sidebar.selectbox("Select Attack", options, help="""CopyCatCNN: 🔍 Creates a substitute model by querying the target model with synthetic data and training a neural network to replicate its predictions and decision boundaries,
Functionally Equivalent Extraction: ⚡ Extracts model functionality without replicating internal structure - focuses on achieving similar input-output behavior with different architecture,
    Knockoff Nets: 🎯 Advanced model stealing using adversarial perturbations and transfer learning to create functional copies with minimal queries to the target model.""")

param={}
if attack_type=="CopyCatCNN":
    #parameters here
#attack=CopycatCNN(classifier,batch_size_fit=64,batch_size_query=64,nb_epochs=10,use_probability=True,nb_stolen=len_steal)
    st.sidebar.subheader("🎯CopyCatCNN Parameters ")
    steal_dataset = st.sidebar.selectbox(
        "Select Dataset for Stealing",
        ["MNIST Test Set", "CIFAR-10", "Fashion-MNIST"],
        help="Choose which dataset to use for querying the target model"
    )
    param["batch_size_fit"] = st.sidebar.slider("Batch Size (Training)", 16, 128, 64, step=16)
    param["batch_size_query"] = st.sidebar.slider("Batch Size (Query)", 16, 128, 64, step=16)
    param["nb_epochs"] = st.sidebar.slider("Training Epochs", 1, 20, 10)
    param["nb_stolen"] = st.sidebar.slider("Number of Samples to Steal", 1000, 10000, 5000, step=500)
    param["use_probability"] = st.sidebar.checkbox("Use Probability Output", value=True)
elif attack_type=="Functionally Equivalent Extraction":
    st.sidebar.subheader("⚡ Functionally Equivalent Extraction Parameters")

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
    st.sidebar.subheader("🎯 Knockoff Nets Parameters")
    steal_dataset = st.sidebar.selectbox(
        "Select Dataset for Stealing",
        ["MNIST Test Set", "CIFAR-10", "Fashion-MNIST"],
        help="Choose which dataset to use for querying the target model"
    )
    param["batch_size_fit"] = st.sidebar.slider("Batch Size (Training)", 16, 128, 64, step=16)
    param["batch_size_query"] = st.sidebar.slider("Batch Size (Query)", 16, 128, 64, step=16)
    param["nb_epochs"] = st.sidebar.slider("Training Epochs", 1, 20, 10)
    param["nb_stolen"] = st.sidebar.slider("Number of Samples to Steal", 1000, 10000, 5000, step=500)
    param["use_probability"] = st.sidebar.checkbox("Use Probability Output", value=True)
    param["sampling_strategy"] = st.selectbox(
        "Sampling Strategy",
        ["random", "adaptive"],
        index=1,
        help="Sampling strategy for selecting queries: 'random' for random sampling, 'adaptive' for adaptive sampling based on model uncertainty"
    )
    param["reward"] = st.selectbox(
        "Reward Strategy",
        ["cert", "div", "loss", "all"],
        index=3,
        help="Reward strategy for adaptive sampling: 'cert' (certainty), 'div' (diversity), 'loss' (loss-based), 'all' (combination)"
    )

run_button = st.button("🚀 Run  Attack", type="primary")
if run_button:
    if attack_type=="CopyCatCNN":
        with st.spinner("⏳ Running " + attack_type + " attack... Please wait"):
            nb_stolen = param["nb_stolen"]
            if steal_dataset=="MNIST Test Set":
                x_steal = test_images[:param["nb_stolen"]]
            else:
                # Use external dataset
                dataset_name = steal_dataset
                st.write(f"The victim model is reconstruct based on{dataset_name} Dataset")
                x_steal = load_external_dataset(dataset_name, nb_stolen)
            attack = CopycatCNN(
                classifier,
                batch_size_fit=param["batch_size_fit"],
                batch_size_query=param["batch_size_query"],
                nb_epochs=param["nb_epochs"],
                use_probability=param["use_probability"],
                nb_stolen=param["nb_stolen"]
            )
            Stolen_model = get_model(10)
            classifier_stolen = KerasClassifier(Stolen_model, clip_values=(0, 1))
            classifier_stolen = attack.extract(thieved_classifier=classifier_stolen, x=x_steal)
            y_test_cat = to_categorical(test_labels[nb_stolen:], nb_classes=10)
            loss_or,acc_org=classifier.model.evaluate(test_images[nb_stolen:], test_labels[nb_stolen:])
            loss, acc = classifier_stolen._model.evaluate(test_images[nb_stolen:], y_test_cat)
            acc_drop=acc_org -acc

            org_pred=classifier.predict(test_images[nb_stolen:])
            stol_pred=classifier_stolen.predict(test_images[nb_stolen:])
            if len(org_pred.shape) > 1:  # If probability outputs
                original_classes = np.argmax(org_pred, axis=1)
            else:
                original_classes = org_pred

            if len(stol_pred.shape) > 1:  # If probability outputs
                stolen_classes = np.argmax(stol_pred, axis=1)
            else:
                stolen_classes = stol_pred
            fidelity=np.mean(original_classes == stolen_classes)
            st.success("✅ CopyCatCNN attack completed!")
            col1,col2,col3 =st.columns(3)
            with col1:
              st.metric("Original Accuracy", f"{acc_org:.3f}", f"{acc_org * 100:.1f}%")
            with col2:
                st.metric("Stolen Accuracy", f"{acc:.3f}", f"{acc * 100:.1f}%")
            with col3:
                st.metric("Fidelity",f"{fidelity}",f"{fidelity * 100:.1f}%")

    elif attack_type=="Functionally Equivalent Extraction":
        st.info("⚠️ Note: This attack requires a dense neural network model. Using pre-trained dense model.")
        #flatt the images
        train_images = train_images.reshape(train_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)
        target_model = get_model_FEE()
        target_model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
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
        acc_drop=acc_org-acc_stolen
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
    elif attack_type == "Knockoff Nets":
        with st.spinner("⏳ Running " + attack_type + " attack... Please wait"):
            nb_stolen = param["nb_stolen"]
            if steal_dataset=="MNIST Test Set":
                x_steal = test_images[:param["nb_stolen"]]
            else:
                # Use external dataset
                dataset_name = steal_dataset
                st.write(f"The victim model is reconstructed based on {dataset_name} Dataset")
                x_steal = load_external_dataset(dataset_name, nb_stolen)
            attack = KnockoffNets(
                classifier,
                batch_size_fit=param["batch_size_fit"],
                batch_size_query=param["batch_size_query"],
                nb_epochs=param["nb_epochs"],
                use_probability=param["use_probability"],
                nb_stolen=param["nb_stolen"],
                sampling_strategy=param["sampling_strategy"],
                reward=param["reward"]
            )
            stolen_model = get_model(10)
            classifier_stolen = KerasClassifier(stolen_model, clip_values=(0, 1))
            y_steal = classifier.predict(x_steal)
            classifier_stolen = attack.extract(
                thieved_classifier=classifier_stolen,
                x=x_steal,
                y=y_steal
            )
            y_test_cat = to_categorical(test_labels[nb_stolen:], nb_classes=10)
            loss, acc = classifier_stolen._model.evaluate(test_images[nb_stolen:], y_test_cat)
            loss,acc_org=classifier.model.evaluate(test_images[nb_stolen:], y_test_cat)
            st.success("✅ Knockoff Nets attack completed!")
            acc_drop = acc_org - acc

            org_pred = classifier.predict(test_images[nb_stolen:])
            stol_pred = classifier_stolen.predict(test_images[nb_stolen:])
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
                st.metric("Stolen Accuracy", f"{acc:.3f}", f"{acc * 100:.1f}%")
            with col3:
                st.metric("Fidelity", f"{fidelity}", f"{fidelity * 100:.1f}%")








