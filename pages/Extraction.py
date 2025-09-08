import streamlit as st
import os
import gc  # Add garbage collection
import psutil  # For memory monitoring

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

# Memory management functions
def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def clear_memory():
    """Clear memory and run garbage collection"""
    gc.collect()
    K.clear_session()

def monitor_memory(stage):
    """Monitor and display memory usage"""
    memory_mb = get_memory_usage()
    st.sidebar.text(f"{stage}: {memory_mb:.1f} MB")
    return memory_mb

st.set_page_config(
    page_title="Extraction Attacks on MNIST",
    page_icon="🔓 ",
    layout="wide"
)
st.title(" 🔓  Extraction Attacks on MNIST")
st.markdown("---")

from keras.models import load_model

@st.cache_resource
def load_mnist_model():
    model = load_model("pages/mnist_model.h5")
    return model

@st.cache_data
def load_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # Reduced dataset size for memory efficiency
    max_train_size = 10000  # Reduced from 14000
    max_test_size = 5000    # Reduced from 7000

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
        monitor_memory("Initial")
        model = load_mnist_model()
        if model is not None:
            (train_images, train_labels), (test_images, test_labels) = load_data()
            classifier = KerasClassifier(model=model, clip_values=(0, 1))

            # Store in session state
            st.session_state.model = model
            st.session_state.classifier = classifier
            st.session_state.train_data = (train_images, train_labels)
            st.session_state.test_data = (test_images, test_labels)
            st.session_state.data_loaded = True
            monitor_memory("After loading")
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
def load_external_dataset(dataset_name, max_samples=5000):  # Reduced default
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
    # Smaller model for memory efficiency
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))  # Reduced from 32
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))  # Reduced from 64
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))  # Reduced from 128
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model

# Get data from session state
classifier = st.session_state.classifier
train_images, train_labels = st.session_state.train_data
test_images, test_labels = st.session_state.test_data
model = st.session_state.model

st.sidebar.header("⚔️ Attack Configuration")

# Memory usage display
st.sidebar.subheader("💾 Memory Usage")
monitor_memory("Current")

options = ["CopyCatCNN","Functionally Equivalent Extraction","Knockoff Nets"]
attack_type = st.sidebar.selectbox("Select Attack", options, help="""CopyCatCNN: 🔍 Creates a substitute model by querying the target model with synthetic data and training a neural network to replicate its predictions and decision boundaries,
Functionally Equivalent Extraction: ⚡ Extracts model functionality without replicating internal structure - focuses on achieving similar input-output behavior with different architecture,
    Knockoff Nets: 🎯 Advanced model stealing using adversarial perturbations and transfer learning to create functional copies with minimal queries to the target model.""")

param={}
if attack_type=="CopyCatCNN":
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
    st.sidebar.warning(
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
    st.sidebar.subheader("🎯 Knockoff Nets Parameters")
    
    # Memory warning for Knockoff Nets
    st.sidebar.warning("⚠️ **Memory Intensive**: Knockoff Nets requires significant memory. Consider using smaller parameters for Streamlit deployment.")
    
    steal_dataset = st.sidebar.selectbox(
        "Select Dataset for Stealing",
        ["MNIST Test Set", "CIFAR-10", "Fashion-MNIST"],
        help="Choose which dataset to use for querying the target model"
    )
    
    # Reduced default values for memory efficiency
    param["batch_size_fit"] = st.sidebar.slider("Batch Size (Training)", 8, 64, 32, step=8)  # Reduced max and default
    param["batch_size_query"] = st.sidebar.slider("Batch Size (Query)", 8, 64, 32, step=8)   # Reduced max and default
    param["nb_epochs"] = st.sidebar.slider("Training Epochs", 1, 10, 5)  # Reduced max and default
    param["nb_stolen"] = st.sidebar.slider("Number of Samples to Steal", 500, 3000, 1500, step=250)  # Reduced significantly
    param["use_probability"] = st.sidebar.checkbox("Use Probability Output", value=True)
    param["sampling_strategy"] = st.sidebar.selectbox(
        "Sampling Strategy",
        ["random", "adaptive"],
        index=0,  # Default to random (less memory intensive)
        help="Sampling strategy for selecting queries: 'random' for random sampling, 'adaptive' for adaptive sampling based on model uncertainty"
    )
    param["reward"] = st.sidebar.selectbox(
        "Reward Strategy",
        ["cert", "div", "loss", "all"],
        index=0,  # Default to cert (less memory intensive)
        help="Reward strategy for adaptive sampling: 'cert' (certainty), 'div' (diversity), 'loss' (loss-based), 'all' (combination)"
    )

run_button = st.button("🚀 Run Attack", type="primary")

if run_button:
    # Clear memory before starting attack
    clear_memory()
    monitor_memory("Before Attack")
    
    try:
        if attack_type=="CopyCatCNN":
            with st.spinner("⏳ Running " + attack_type + " attack... Please wait"):
                nb_stolen = param["nb_stolen"]
                if steal_dataset=="MNIST Test Set":
                    x_steal = test_images[:param["nb_stolen"]]
                else:
                    dataset_name = steal_dataset
                    st.write(f"The victim model is reconstructed based on {dataset_name} Dataset")
                    x_steal = load_external_dataset(dataset_name, nb_stolen)
                
                attack = CopycatCNN(
                    classifier,
                    batch_size_fit=param["batch_size_fit"],
                    batch_size_query=param["batch_size_query"],
                    nb_epochs=param["nb_epochs"],
                    use_probability=param["use_probability"],
                    nb_stolen=param["nb_stolen"]
                )
                
                stolen_model = get_model(10)
                classifier_stolen = KerasClassifier(stolen_model, clip_values=(0, 1))
                classifier_stolen = attack.extract(thieved_classifier=classifier_stolen, x=x_steal)
                
                y_test_cat = to_categorical(test_labels[nb_stolen:], nb_classes=10)
                loss_or,acc_org=classifier.model.evaluate(test_images[nb_stolen:], test_labels[nb_stolen:])
                loss, acc = classifier_stolen._model.evaluate(test_images[nb_stolen:], y_test_cat)
                
                org_pred=classifier.predict(test_images[nb_stolen:])
                stol_pred=classifier_stolen.predict(test_images[nb_stolen:])
                
                if len(org_pred.shape) > 1:
                    original_classes = np.argmax(org_pred, axis=1)
                else:
                    original_classes = org_pred

                if len(stol_pred.shape) > 1:
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
                    st.metric("Fidelity",f"{fidelity:.3f}",f"{fidelity * 100:.1f}%")

        elif attack_type=="Functionally Equivalent Extraction":
            st.info("⚠️ Note: This attack requires a dense neural network model. Using pre-trained dense model.")
            
            train_images_flat = train_images.reshape(train_images.shape[0], -1)
            test_images_flat = test_images.reshape(test_images.shape[0], -1)
            
            target_model = get_model_FEE()
            target_model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                               metrics=["accuracy"])
            target_model.fit(train_images_flat, train_labels, epochs=5)
            
            loss, acc_org = target_model.evaluate(test_images_flat[:2500], test_labels[:2500])  # Reduced test size
            classifier = KerasClassifier(target_model, clip_values=(0, 1), use_logits=True)
            
            attack = FunctionallyEquivalentExtraction(classifier, num_neurons=param["num_neurons"])
            stolen_classifier = attack.extract(
                test_images_flat[2500:], test_labels[2500:],
                delta_0=param["delta_0"],
                fraction_true=param["fraction_true"],
                rel_diff_slope=param["rel_diff_slope"],
                rel_diff_value=param["rel_diff_value"],
                delta_init_value=param["delta_init_value"],
                delta_value_max=param["delta_value_max"]
            )
            
            loss, acc_stolen = stolen_classifier.model.evaluate(test_images_flat[:2500], test_labels[:2500])
            st.success("✅ Functionally Equivalent Extraction completed!")
            
            org_pred = classifier.predict(test_images_flat[:2500])
            stol_pred = stolen_classifier.predict(test_images_flat[:2500])
            
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
            with col1:
                st.metric("Original Accuracy", f"{acc_org:.3f}", f"{acc_org * 100:.1f}%")
            with col2:
                st.metric("Stolen Accuracy", f"{acc_stolen:.3f}", f"{acc_stolen * 100:.1f}%")
            with col3:
                st.metric("Fidelity", f"{fidelity:.3f}", f"{fidelity * 100:.1f}%")

        elif attack_type == "Knockoff Nets":
            with st.spinner("⏳ Running " + attack_type + " attack... Please wait"):
                nb_stolen = param["nb_stolen"]
                
                # Memory check before proceeding
                if get_memory_usage() > 800:  # MB threshold
                    st.error("⚠️ Memory usage too high. Please reduce parameters or restart the app.")
                    st.stop()
                
                if steal_dataset=="MNIST Test Set":
                    x_steal = test_images[:nb_stolen]
                else:
                    dataset_name = steal_dataset
                    st.write(f"The victim model is reconstructed based on {dataset_name} Dataset")
                    x_steal = load_external_dataset(dataset_name, nb_stolen)
                
                monitor_memory("After data prep")
                
                # Process in smaller batches to manage memory
                batch_size_memory = min(param["batch_size_fit"], 16)  # Limit batch size
                
                attack = KnockoffNets(
                    classifier,
                    batch_size_fit=batch_size_memory,
                    batch_size_query=batch_size_memory,
                    nb_epochs=param["nb_epochs"],
                    use_probability=param["use_probability"],
                    nb_stolen=nb_stolen,
                    sampling_strategy=param["sampling_strategy"],
                    reward=param["reward"]
                )
                
                monitor_memory("After attack init")
                
                stolen_model = get_model(10)
                classifier_stolen = KerasClassifier(stolen_model, clip_values=(0, 1))
                
                # Get labels in smaller batches to manage memory
                y_steal = []
                batch_size_predict = 100
                for i in range(0, len(x_steal), batch_size_predict):
                    batch_x = x_steal[i:i+batch_size_predict]
                    batch_y = classifier.predict(batch_x)
                    y_steal.append(batch_y)
                    clear_memory()  # Clear memory after each batch
                
                y_steal = np.concatenate(y_steal, axis=0)
                monitor_memory("After prediction")
                
                classifier_stolen = attack.extract(
                    thieved_classifier=classifier_stolen,
                    x=x_steal,
                    y=y_steal
                )
                
                monitor_memory("After extraction")
                
                # Evaluate on smaller test set
                test_size = min(1000, len(test_images) - nb_stolen)
                y_test_cat = to_categorical(test_labels[nb_stolen:nb_stolen+test_size], nb_classes=10)
                
                loss, acc_org = classifier.model.evaluate(test_images[nb_stolen:nb_stolen+test_size], 
                                                         test_labels[nb_stolen:nb_stolen+test_size])
                loss, acc = classifier_stolen._model.evaluate(test_images[nb_stolen:nb_stolen+test_size], y_test_cat)
                
                # Calculate fidelity on smaller set
                org_pred = classifier.predict(test_images[nb_stolen:nb_stolen+test_size])
                stol_pred = classifier_stolen.predict(test_images[nb_stolen:nb_stolen+test_size])
                
                if len(org_pred.shape) > 1:
                    original_classes = np.argmax(org_pred, axis=1)
                else:
                    original_classes = org_pred

                if len(stol_pred.shape) > 1:
                    stolen_classes = np.argmax(stol_pred, axis=1)
                else:
                    stolen_classes = stol_pred
                    
                fidelity = np.mean(original_classes == stolen_classes)
                
                st.success("✅ Knockoff Nets attack completed!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Accuracy", f"{acc_org:.3f}", f"{acc_org * 100:.1f}%")
                with col2:
                    st.metric("Stolen Accuracy", f"{acc:.3f}", f"{acc * 100:.1f}%")
                with col3:
                    st.metric("Fidelity", f"{fidelity:.3f}", f"{fidelity * 100:.1f}%")
                
                monitor_memory("Final")
                
    except Exception as e:
        st.error(f"❌ Attack failed: {str(e)}")
        st.info("💡 Try reducing the number of samples, batch size, or epochs.")
        clear_memory()
        
    finally:
        # Always clear memory after attack
        clear_memory()
        monitor_memory("After cleanup")
