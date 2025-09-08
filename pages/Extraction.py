import streamlit as st
import os
import gc  # Garbage collector for memory management

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
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import pandas as pd
from art.estimators.classification import KerasClassifier
from art.attacks.extraction import CopycatCNN, FunctionallyEquivalentExtraction, KnockoffNets

# Configure TensorFlow for memory efficiency
def configure_tensorflow():
    """Configure TensorFlow to use less memory"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass
    
    # Set memory limit for CPU
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)

configure_tensorflow()

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
    try:
        model = load_model("pages/mnist_model.h5")
        return model
    except:
        st.error("Model file not found. Creating a simple model for demonstration.")
        return create_simple_model()

def create_simple_model():
    """Create a simple model if the original model file is not available"""
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Quick training on small subset
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train[:1000].reshape(-1, 28, 28, 1) / 255.0
    y_train = y_train[:1000]
    model.fit(x_train, y_train, epochs=3, verbose=0, batch_size=32)
    
    return model

@st.cache_data
def load_data(max_train_size=5000, max_test_size=2000):  # Reduced sizes
    """Load and preprocess MNIST data with reduced memory footprint"""
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # Use smaller dataset sizes
    train_images = train_images[:max_train_size]
    train_labels = train_labels[:max_train_size]
    test_images = test_images[:max_test_size]
    test_labels = test_labels[:max_test_size]

    # Normalize and reshape
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    return (train_images, train_labels), (test_images, test_labels)

def clear_memory():
    """Clear memory and reset TensorFlow session"""
    K.clear_session()
    gc.collect()
    tf.keras.backend.clear_session()

if 'data_loaded' not in st.session_state:
    with st.spinner("Loading model and data..."):
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
        else:
            st.error("❌ Failed to load model")
            st.stop()

def get_model_FEE():
    clear_memory()
    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=(784,)))  # Reduced neurons
    model.add(Dense(10, activation="linear"))
    return model

@st.cache_data
def load_external_dataset(dataset_name, max_samples=2000):  # Reduced max samples
    """Load and preprocess external datasets with memory optimization"""
    try:
        if dataset_name == "CIFAR-10":
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            # Use smaller subset
            x_combined = x_train[:max_samples]  # Only use training set
            
            # Process in batches to save memory
            batch_size = 500
            processed_batches = []
            
            for i in range(0, len(x_combined), batch_size):
                batch = x_combined[i:i+batch_size]
                # Convert to grayscale and resize
                batch_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in batch])
                batch_resized = np.array([cv2.resize(img, (28, 28)) for img in batch_gray])
                processed_batches.append(batch_resized)
                
            x_processed = np.concatenate(processed_batches)
            x_processed = x_processed.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
            
        elif dataset_name == "Fashion-MNIST":
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
            x_combined = x_train[:max_samples]
            x_processed = x_combined.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0

        return x_processed
        
    except Exception as e:
        st.error(f"Error loading {dataset_name}: {str(e)}")
        return None

def get_model(NUM_CLASSES):
    clear_memory()
    model = Sequential()
    # Smaller model architecture
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model

# Memory usage warning
if st.sidebar.button("📊 Check Memory Usage"):
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    st.sidebar.info(f"Memory Usage: {memory_info.rss / 1024 / 1024:.1f} MB")

# Get data from session state
if 'data_loaded' in st.session_state and st.session_state.data_loaded:
    classifier = st.session_state.classifier
    train_images, train_labels = st.session_state.train_data
    test_images, test_labels = st.session_state.test_data
    model = st.session_state.model

    st.sidebar.header("⚔️ Attack Configuration")
    
    # Add memory-friendly mode
    memory_mode = st.sidebar.checkbox("💾 Low Memory Mode", value=True, 
                                     help="Reduces dataset sizes and model complexity for systems with limited RAM")
    
    options = ["CopyCatCNN", "Functionally Equivalent Extraction", "Knockoff Nets"]
    attack_type = st.sidebar.selectbox("Select Attack", options, 
                                       help="""CopyCatCNN: 🔍 Creates a substitute model by querying the target model,
    Functionally Equivalent Extraction: ⚡ Extracts model functionality with different architecture,
    Knockoff Nets: 🎯 Advanced model stealing using adversarial perturbations.""")

    param = {}
    if attack_type == "CopyCatCNN":
        st.sidebar.subheader("🎯CopyCatCNN Parameters ")
        steal_dataset = st.sidebar.selectbox(
            "Select Dataset for Stealing",
            ["MNIST Test Set", "CIFAR-10", "Fashion-MNIST"],
            help="Choose which dataset to use for querying the target model"
        )
        max_samples = 2000 if memory_mode else 5000
        param["batch_size_fit"] = st.sidebar.slider("Batch Size (Training)", 8, 64, 32, step=8)
        param["batch_size_query"] = st.sidebar.slider("Batch Size (Query)", 8, 64, 32, step=8)
        param["nb_epochs"] = st.sidebar.slider("Training Epochs", 1, 15, 5)
        param["nb_stolen"] = st.sidebar.slider("Number of Samples to Steal", 500, max_samples, 1000, step=250)
        param["use_probability"] = st.sidebar.checkbox("Use Probability Output", value=True)
        
    elif attack_type == "Functionally Equivalent Extraction":
        st.sidebar.subheader("⚡ Functionally Equivalent Extraction Parameters")
        st.sidebar.warning("⚠️ **Memory Warning**: This attack is computationally intensive.")
        
        if memory_mode:
            st.sidebar.info("🔧 Low memory mode: Using reduced parameters")
            param["num_neurons"] = st.sidebar.slider("Number of Neurons", 32, 128, 64, step=32)
        else:
            param["num_neurons"] = st.sidebar.number_input("Number of Neurons", min_value=64, max_value=256, value=128, step=32)

        with st.sidebar.expander("🔧 Advanced Parameters"):
            param["delta_0"] = st.number_input("Delta 0", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
            param["fraction_true"] = st.number_input("Fraction True", min_value=0.2, max_value=0.8, value=0.3, step=0.1)
            param["rel_diff_slope"] = st.selectbox("Relative Diff Slope", [1e-4, 1e-5, 1e-6], index=1)
            param["rel_diff_value"] = st.selectbox("Relative Diff Value", [1e-5, 1e-6, 1e-7], index=1)
            param["delta_init_value"] = st.number_input("Delta Init Value", min_value=0.05, max_value=0.5, value=0.1, step=0.05)
            param["delta_value_max"] = st.number_input("Delta Value Max", min_value=20, max_value=50, value=30, step=10)

    elif attack_type == "Knockoff Nets":
        st.sidebar.subheader("🎯 Knockoff Nets Parameters")
        steal_dataset = st.sidebar.selectbox(
            "Select Dataset for Stealing",
            ["MNIST Test Set", "Fashion-MNIST"],  # Removed CIFAR-10 for memory efficiency
            help="Choose which dataset to use for querying the target model"
        )
        max_samples = 2000 if memory_mode else 3000
        param["batch_size_fit"] = st.sidebar.slider("Batch Size (Training)", 8, 32, 16, step=8)
        param["batch_size_query"] = st.sidebar.slider("Batch Size (Query)", 8, 32, 16, step=8)
        param["nb_epochs"] = st.sidebar.slider("Training Epochs", 1, 10, 5)
        param["nb_stolen"] = st.sidebar.slider("Number of Samples to Steal", 500, max_samples, 1000, step=250)
        param["use_probability"] = st.sidebar.checkbox("Use Probability Output", value=True)
        param["sampling_strategy"] = st.sidebar.selectbox("Sampling Strategy", ["random", "adaptive"], index=0)
        param["reward"] = st.sidebar.selectbox("Reward Strategy", ["cert", "div", "loss"], index=0)

    run_button = st.button("🚀 Run Attack", type="primary")
    
    if run_button:
        clear_memory()  # Clear memory before starting attack
        
        if attack_type == "CopyCatCNN":
            with st.spinner("⏳ Running " + attack_type + " attack... Please wait"):
                try:
                    nb_stolen = param["nb_stolen"]
                    if steal_dataset == "MNIST Test Set":
                        x_steal = test_images[:nb_stolen]
                    else:
                        dataset_name = steal_dataset
                        st.write(f"The victim model is reconstructed based on {dataset_name} Dataset")
                        x_steal = load_external_dataset(dataset_name, nb_stolen)
                        if x_steal is None:
                            st.error("Failed to load external dataset")
                            st.stop()
                    
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
                    
                    # Evaluate on smaller test set
                    test_subset = min(1000, len(test_images) - nb_stolen)
                    y_test_cat = to_categorical(test_labels[nb_stolen:nb_stolen+test_subset], nb_classes=10)
                    
                    loss_org, acc_org = classifier.model.evaluate(
                        test_images[nb_stolen:nb_stolen+test_subset], 
                        test_labels[nb_stolen:nb_stolen+test_subset], 
                        verbose=0
                    )
                    loss, acc = classifier_stolen._model.evaluate(
                        test_images[nb_stolen:nb_stolen+test_subset], 
                        y_test_cat, 
                        verbose=0
                    )

                    # Calculate fidelity
                    org_pred = classifier.predict(test_images[nb_stolen:nb_stolen+test_subset])
                    stol_pred = classifier_stolen.predict(test_images[nb_stolen:nb_stolen+test_subset])
                    
                    original_classes = np.argmax(org_pred, axis=1) if len(org_pred.shape) > 1 else org_pred
                    stolen_classes = np.argmax(stol_pred, axis=1) if len(stol_pred.shape) > 1 else stol_pred
                    fidelity = np.mean(original_classes == stolen_classes)
                    
                    st.success("✅ CopyCatCNN attack completed!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Accuracy", f"{acc_org:.3f}", f"{acc_org * 100:.1f}%")
                    with col2:
                        st.metric("Stolen Accuracy", f"{acc:.3f}", f"{acc * 100:.1f}%")
                    with col3:
                        st.metric("Fidelity", f"{fidelity:.3f}", f"{fidelity * 100:.1f}%")
                        
                except Exception as e:
                    st.error(f"❌ Attack failed: {str(e)}")
                    st.info("💡 Try enabling Low Memory Mode or reducing parameter values.")

        elif attack_type == "Functionally Equivalent Extraction":
            with st.spinner("⏳ Running " + attack_type + " attack... Please wait"):
                try:
                    st.info("⚠️ Note: Using dense neural network model for this attack.")
                    
                    # Use smaller dataset - always optimized for memory
                    subset_size = 2000
                    train_images_flat = train_images[:subset_size].reshape(train_images[:subset_size].shape[0], -1)
                    test_images_flat = test_images[:1000].reshape(test_images[:1000].shape[0], -1)
                    
                    target_model = get_model_FEE()
                    target_model.compile(
                        optimizer="adam", 
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=["accuracy"]
                    )
                    
                    # Train with smaller epochs
                    target_model.fit(
                        train_images_flat, train_labels[:subset_size], 
                        epochs=3, verbose=0, batch_size=32
                    )
                    
                    loss, acc_org = target_model.evaluate(test_images_flat, test_labels[:1000], verbose=0)
                    classifier_fee = KerasClassifier(target_model, clip_values=(0, 1), use_logits=True)
                    
                    attack = FunctionallyEquivalentExtraction(classifier_fee, num_neurons=param["num_neurons"])
                    
                    # Use smaller extraction dataset
                    extract_size = 500
                    stolen_classifier = attack.extract(
                        test_images_flat[:extract_size], 
                        test_labels[:extract_size],
                        delta_0=param["delta_0"],
                        fraction_true=param["fraction_true"],
                        rel_diff_slope=param["rel_diff_slope"],
                        rel_diff_value=param["rel_diff_value"],
                        delta_init_value=param["delta_init_value"],
                        delta_value_max=param["delta_value_max"]
                    )
                    
                    loss, acc_stolen = stolen_classifier.model.evaluate(test_images_flat[:500], test_labels[:500], verbose=0)
                    
                    # Calculate fidelity
                    org_pred = classifier_fee.predict(test_images_flat[:500])
                    stol_pred = stolen_classifier.predict(test_images_flat[:500])
                    
                    original_classes = np.argmax(org_pred, axis=1) if len(org_pred.shape) > 1 else org_pred
                    stolen_classes = np.argmax(stol_pred, axis=1) if len(stol_pred.shape) > 1 else stol_pred
                    fidelity = np.mean(original_classes == stolen_classes)
                    
                    st.success("✅ Functionally Equivalent Extraction completed!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Accuracy", f"{acc_org:.3f}", f"{acc_org * 100:.1f}%")
                    with col2:
                        st.metric("Stolen Accuracy", f"{acc_stolen:.3f}", f"{acc_stolen * 100:.1f}%")
                    with col3:
                        st.metric("Fidelity", f"{fidelity:.3f}", f"{fidelity * 100:.1f}%")
                        
                except Exception as e:
                    st.error(f"❌ Attack failed: {str(e)}")
                    st.info("💡 This attack is memory-intensive. Try reducing the number of neurons.")
                    
        elif attack_type == "Knockoff Nets":
            with st.spinner("⏳ Running " + attack_type + " attack... Please wait"):
                try:
                    nb_stolen = param["nb_stolen"]
                    if steal_dataset == "MNIST Test Set":
                        x_steal = test_images[:nb_stolen]
                    else:
                        dataset_name = steal_dataset
                        st.write(f"The victim model is reconstructed based on {dataset_name} Dataset")
                        x_steal = load_external_dataset(dataset_name, nb_stolen)
                        if x_steal is None:
                            st.error("Failed to load external dataset")
                            st.stop()
                    
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
                    
                    classifier_stolen = attack.extract(
                        thieved_classifier=classifier_stolen,
                        x=x_steal,
                        y=y_steal
                    )
                    
                    # Evaluate on smaller test set
                    test_subset = min(1000, len(test_images) - nb_stolen)
                    y_test_cat = to_categorical(test_labels[nb_stolen:nb_stolen+test_subset], nb_classes=10)
                    
                    loss, acc = classifier_stolen._model.evaluate(
                        test_images[nb_stolen:nb_stolen+test_subset], 
                        y_test_cat, 
                        verbose=0
                    )
                    loss_org, acc_org = classifier.model.evaluate(
                        test_images[nb_stolen:nb_stolen+test_subset], 
                        y_test_cat, 
                        verbose=0
                    )

                    # Calculate fidelity
                    org_pred = classifier.predict(test_images[nb_stolen:nb_stolen+test_subset])
                    stol_pred = classifier_stolen.predict(test_images[nb_stolen:nb_stolen+test_subset])
                    
                    original_classes = np.argmax(org_pred, axis=1) if len(org_pred.shape) > 1 else org_pred
                    stolen_classes = np.argmax(stol_pred, axis=1) if len(stol_pred.shape) > 1 else stol_pred
                    fidelity = np.mean(original_classes == stolen_classes)

                    st.success("✅ Knockoff Nets attack completed!")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Accuracy", f"{acc_org:.3f}", f"{acc_org * 100:.1f}%")
                    with col2:
                        st.metric("Stolen Accuracy", f"{acc:.3f}", f"{acc * 100:.1f}%")
                    with col3:
                        st.metric("Fidelity", f"{fidelity:.3f}", f"{fidelity * 100:.1f}%")
                        
                except Exception as e:
                    st.error(f"❌ Attack failed: {str(e)}")
                    st.info("💡 Try reducing batch sizes, number of samples, or enable Low Memory Mode.")

        clear_memory()  # Clean up after attack

else:
    st.warning("⚠️ Data not loaded. Please ensure the model file exists and try refreshing the page.")
