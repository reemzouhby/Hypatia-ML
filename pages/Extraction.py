import streamlit as st
import os
import gc
import uuid
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.datasets import mnist, fashion_mnist, cifar10
import cv2
from art.utils import to_categorical
from art.estimators.classification import KerasClassifier
from art.attacks.extraction import CopycatCNN, FunctionallyEquivalentExtraction, KnockoffNets

# Disable GPU and reduce TensorFlow verbosity
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.set_page_config(
    page_title="Extraction Attacks",
    page_icon="🔓",
    layout="wide"
)

st.title("🔓 Extraction Attacks")
st.markdown("---")


@st.cache_resource
def load_mnist_model():
    """Load the pre-trained MNIST model"""
    try:
        model_path = "mnist_model.h5"
        if not os.path.exists(model_path):
            model_path = "pages/mnist_model.h5"
        if not os.path.exists(model_path):
            st.error("Model file not found. Please ensure mnist_model.h5 exists.")
            return None
        return load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_and_prepare_data():
    """Load and prepare MNIST data"""
    try:
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        # Normalize images
        train_images = train_images.astype('float32') / 255.0
        test_images = test_images.astype('float32') / 255.0

        # Reshape for CNN (add channel dimension)
        train_images = train_images.reshape(-1, 28, 28, 1)
        test_images = test_images.reshape(-1, 28, 28, 1)

        # Limit data size for demo
        max_size = 10000
        train_images = train_images[:max_size]
        train_labels = train_labels[:max_size]

        return (train_images, train_labels), (test_images, test_labels)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


@st.cache_data
def load_external_dataset(dataset_name, max_samples=5000):
    """Load external datasets for stealing"""
    try:
        if dataset_name == "CIFAR-10":
            (_, _), (x_test, _) = cifar10.load_data()
            # Convert to grayscale and resize to 28x28
            x_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_test[:max_samples]])
            x_processed = np.array([cv2.resize(img, (28, 28)) for img in x_gray])
            return x_processed.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        elif dataset_name == "Fashion-MNIST":
            (_, _), (x_test, _) = fashion_mnist.load_data()
            return x_test[:max_samples].reshape(-1, 28, 28, 1).astype('float32') / 255.0
        else:
            return None
    except Exception as e:
        st.error(f"Error loading external dataset: {e}")
        return None


def create_stolen_model(num_classes=10):
    """Create architecture for stolen model - similar to your working code"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_fee_model():
    """Create model for Functionally Equivalent Extraction"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dense(10, activation='linear')  # Linear activation for FEE
    ])
    return model


# Initialize session state
if 'initialized' not in st.session_state:
    with st.spinner("Loading model and data..."):
        # Load model
        model = load_mnist_model()
        if model is None:
            st.stop()

        # Load data
        data_result = load_and_prepare_data()
        if data_result[0] is None:
            st.stop()

        (train_images, train_labels), (test_images, test_labels) = data_result

        # Create classifier
        classifier = KerasClassifier(model=model, clip_values=(0, 1))

        # Store in session state
        st.session_state.model = model
        st.session_state.classifier = classifier
        st.session_state.train_data = (train_images, train_labels)
        st.session_state.test_data = (test_images, test_labels)
        st.session_state.initialized = True

# Get data from session state
classifier = st.session_state.classifier
train_images, train_labels = st.session_state.train_data
test_images, test_labels = st.session_state.test_data

# Sidebar configuration
st.sidebar.header("⚔️ Attack Configuration")
attack_options = ["CopyCatCNN", "Knockoff Nets", "Functionally Equivalent Extraction"]
attack_type = st.sidebar.selectbox("Select Attack", attack_options)

# Attack parameters
if attack_type in ["CopyCatCNN", "Knockoff Nets"]:
    steal_dataset = st.sidebar.selectbox(
        "Dataset for Stealing",
        ["MNIST Test Set", "CIFAR-10", "Fashion-MNIST"]
    )
    batch_size_fit = st.sidebar.slider("Batch Size (Training)", 16, 128, 32, 16)
    batch_size_query = st.sidebar.slider("Batch Size (Query)", 16, 128, 32, 16)
    nb_epochs = st.sidebar.slider("Training Epochs", 5, 20, 10)
    nb_stolen = st.sidebar.slider("Number of Samples to Steal", 500, 3000, 2000, 500)
    use_probability = st.sidebar.checkbox("Use Probability Output", value=True)

    if attack_type == "Knockoff Nets":
        sampling_strategy = st.sidebar.selectbox("Sampling Strategy", ["random", "adaptive"])
        reward = st.sidebar.selectbox("Reward Strategy", ["cert", "div", "loss", "all"])

elif attack_type == "Functionally Equivalent Extraction":
    st.sidebar.subheader("⚡ FEE Parameters")
    st.sidebar.warning("⚠️ This attack can take a long time to complete.")
    num_neurons = st.sidebar.number_input("Number of Neurons", min_value=64, max_value=512, value=128, step=64)
    with st.sidebar.expander("🔧 Advanced Parameters"):
        delta_0 = st.number_input("Delta 0 (Initial step size)", min_value=0.001, max_value=0.1, value=0.05,
                                  step=0.001, format="%.3f")
        fraction_true = st.number_input("Fraction True", min_value=0.1, max_value=0.9, value=0.3, step=0.1)
        rel_diff_slope = st.number_input("Relative Diff Slope", min_value=1e-7, max_value=1e-3, value=1e-5,
                                         format="%.2e")
        rel_diff_value = st.number_input("Relative Diff Value", min_value=1e-8, max_value=1e-4, value=1e-6,
                                         format="%.2e")
        delta_init_value = st.number_input("Delta Init Value", min_value=0.01, max_value=1.0, value=0.1,
                                           step=0.01)
        delta_value_max = st.number_input("Delta Value Max", min_value=10, max_value=100, value=50, step=10)

# Run attack button
if st.button("🚀 Run Attack", type="primary"):
    if attack_type in ["CopyCatCNN", "Knockoff Nets"]:
        with st.spinner(f"⏳ Running {attack_type} attack..."):
            try:
                # Prepare stealing data
                if steal_dataset == "MNIST Test Set":
                    # Use random permutation like your working code
                    indices = np.random.permutation(len(test_images))
                    x_steal = test_images[indices[:nb_stolen]]
                    x_eval = test_images[indices[nb_stolen:nb_stolen + 1000]] 
                    y_eval = test_labels[indices[nb_stolen:nb_stolen + 1000]]
                else:
                    x_steal = load_external_dataset(steal_dataset, nb_stolen)
                    if x_steal is None:
                        st.error("Failed to load external dataset")
                        st.stop()
                    # Use remaining MNIST test data for evaluation
                    x_eval = test_images[:1000]
                    y_eval = test_labels[:1000]

                # Get predictions for stolen data 
                y_steal_predictions = classifier.predict(x_steal)

                # Create stolen model
                stolen_model = create_stolen_model(10)
                classifier_stolen = KerasClassifier(stolen_model, clip_values=(0, 1))

                # Configure and run attack
                if attack_type == "CopyCatCNN":
                    attack = CopycatCNN(
                        classifier,
                        batch_size_fit=batch_size_fit,
                        batch_size_query=batch_size_query,
                        nb_epochs=nb_epochs,
                        nb_stolen=nb_stolen,
                        use_probability=use_probability
                    )

                    classifier_stolen = attack.extract(
                        thieved_classifier=classifier_stolen,
                        x=x_steal
                    )

                elif attack_type == "Knockoff Nets":
                    attack = KnockoffNets(
                        classifier,
                        batch_size_fit=batch_size_fit,
                        batch_size_query=batch_size_query,
                        nb_epochs=nb_epochs,
                        nb_stolen=nb_stolen,
                        use_probability=use_probability,
                        sampling_strategy=sampling_strategy,
                        reward=reward
                    )
                    # For Knockoff Nets, we need to provide both x and y
                    classifier_stolen = attack.extract(
                        thieved_classifier=classifier_stolen,
                        x=x_steal,
                        y=y_steal_predictions
                    )

                # Evaluate models
                # Original model evaluation
                loss_orig, acc_orig = classifier.model.evaluate(
                    x_eval, y_eval, verbose=0
                )

                # Stolen model evaluation (convert labels to categorical)
                y_eval_cat = to_categorical(y_eval, 10)
                loss_stolen, acc_stolen = classifier_stolen._model.evaluate(
                    x_eval, y_eval_cat, verbose=0
                )

                # Calculate fidelity (agreement between models)
                orig_pred = classifier.predict(x_eval)
                stolen_pred = classifier_stolen.predict(x_eval)

                orig_classes = np.argmax(orig_pred, axis=1)
                stolen_classes = np.argmax(stolen_pred, axis=1)
                fidelity = np.mean(orig_classes == stolen_classes)

                
                st.success(f"✅ {attack_type} attack completed!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Original Model Accuracy",
                        f"{acc_orig:.3f}",
                        f"{acc_orig * 100:.1f}%"
                    )
                with col2:
                    st.metric(
                        "Stolen Model Accuracy",
                        f"{acc_stolen:.3f}",
                        f"{acc_stolen * 100:.1f}%"
                    )
                with col3:
                    st.metric(
                        "Model Fidelity",
                        f"{fidelity:.3f}",
                        f"{fidelity * 100:.1f}%"
                    )

            except Exception as e:
                st.error(f"Attack failed: {str(e)}")
                st.exception(e)

    elif attack_type == "Functionally Equivalent Extraction":
        with st.spinner("⏳ Running Functionally Equivalent Extraction..."):
            try:
                st.info("⚠️ Converting to dense model format for FEE attack")

                # Flatten images for dense model
                train_flat = train_images.reshape(train_images.shape[0], -1)
                test_flat = test_images.reshape(test_images.shape[0], -1)

                # Create and train target model
                target_model = create_fee_model()
                target_model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',  # Use sparse since we have integer labels
                    metrics=['accuracy']
                )

                # Train the target model
                with st.spinner("Training target model..."):
                    target_model.fit(
                        train_flat[:5000], train_labels[:5000],
                        epochs=5, verbose=0, batch_size=32
                    )

                # Evaluate original model
                loss_orig, acc_orig = target_model.evaluate(
                    test_flat[:1000], test_labels[:1000], verbose=0
                )

                # Create classifier for FEE
                fee_classifier = KerasClassifier(target_model, clip_values=(0, 1), use_logits=True)

                # Run FEE attack
                attack = FunctionallyEquivalentExtraction(fee_classifier, num_neurons=num_neurons)
                stolen_classifier = attack.extract(
                    test_flat[1000:2000], test_labels[1000:2000],
                    delta_0=delta_0,
                    fraction_true=fraction_true,
                    rel_diff_slope=rel_diff_slope,
                    rel_diff_value=rel_diff_value,
                    delta_init_value=delta_init_value,
                    delta_value_max=delta_value_max
                )

                # Evaluate stolen model
                loss_stolen, acc_stolen = stolen_classifier.model.evaluate(
                    test_flat[:1000], test_labels[:1000], verbose=0
                )

                # Calculate fidelity
                orig_pred = fee_classifier.predict(test_flat[:1000])
                stolen_pred = stolen_classifier.predict(test_flat[:1000])

                orig_classes = np.argmax(orig_pred, axis=1) if len(orig_pred.shape) > 1 else orig_pred
                stolen_classes = np.argmax(stolen_pred, axis=1) if len(stolen_pred.shape) > 1 else stolen_pred
                fidelity = np.mean(orig_classes == stolen_classes)

                st.success("✅ Functionally Equivalent Extraction completed!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Accuracy", f"{acc_orig:.3f}", f"{acc_orig * 100:.1f}%")
                with col2:
                    st.metric("Stolen Accuracy", f"{acc_stolen:.3f}", f"{acc_stolen * 100:.1f}%")
                with col3:
                    st.metric("Fidelity", f"{fidelity:.3f}", f"{fidelity * 100:.1f}%")

            except Exception as e:
                st.error(f"FEE attack failed: {str(e)}")
                st.exception(e)
