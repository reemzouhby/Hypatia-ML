import streamlit as st
import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.datasets import mnist
from art.estimators.classification import KerasClassifier
from art.attacks.extraction import CopycatCNN
from art.utils import to_categorical

# -------------------- Setup --------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.clear_session()

st.set_page_config(page_title="Extraction Attack Demo", layout="wide")
st.title("🔓 Light CopyCatCNN Attack Demo on MNIST")
st.markdown("---")

# -------------------- Load Data --------------------
@st.cache_data
def load_data(max_train=1000, max_test=500):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[:max_train] / 255.0
    y_train = y_train[:max_train]
    x_test = x_test[:max_test] / 255.0
    y_test = y_test[:max_test]
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)

(train_images, train_labels), (test_images, test_labels) = load_data()

# -------------------- Load / Create Victim Model --------------------
@st.cache_resource
def load_victim_model():
    try:
        model = load_model("pages/mnist_model.h5")  # Optional: pre-trained
    except:
        # If no model, create and train a small CNN quickly
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1), padding='same'),
            MaxPooling2D((2,2)),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_images, train_labels, epochs=2, batch_size=32, verbose=0)
    return model

victim_model = load_victim_model()
classifier = KerasClassifier(model=victim_model, clip_values=(0,1))

# -------------------- Sidebar --------------------
st.sidebar.header("⚔️ Attack Configuration")
nb_stolen = st.sidebar.slider("Number of samples to steal", 100, 500, 200, step=50)
batch_size = st.sidebar.slider("Batch Size", 16, 64, 32)
epochs = st.sidebar.slider("Training Epochs", 1, 5, 2)

run_button = st.button("🚀 Run CopyCatCNN Attack")

# -------------------- Run Attack --------------------
if run_button:
    st.info("Running CopyCatCNN attack with small dataset...")

    # Select samples to steal
    x_steal = test_images[:nb_stolen]

    # Build small substitute model
    tf.keras.backend.clear_session()
    stolen_model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1), padding='same'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
    ])
    classifier_stolen = KerasClassifier(stolen_model, clip_values=(0,1))

    # Run CopyCatCNN
    attack = CopycatCNN(classifier, nb_epochs=epochs, batch_size_fit=batch_size,
                        batch_size_query=batch_size, nb_stolen=nb_stolen)
    classifier_stolen = attack.extract(thieved_classifier=classifier_stolen, x=x_steal)

    # Evaluate
    y_test_cat = to_categorical(test_labels[nb_stolen:], nb_classes=10)
    loss_org, acc_org = classifier.model.evaluate(test_images[nb_stolen:], test_labels[nb_stolen:], verbose=0)
    loss_stol, acc_stol = classifier_stolen._model.evaluate(test_images[nb_stolen:], y_test_cat, verbose=0)

    # Fidelity
    org_pred = np.argmax(classifier.predict(test_images[nb_stolen:]), axis=1)
    stol_pred = np.argmax(classifier_stolen.predict(test_images[nb_stolen:]), axis=1)
    fidelity = np.mean(org_pred == stol_pred)

    # Display metrics
    st.success("✅ Attack completed!")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Accuracy", f"{acc_org:.3f}", f"{acc_org*100:.1f}%")
    with col2:
        st.metric("Stolen Accuracy", f"{acc_stol:.3f}", f"{acc_stol*100:.1f}%")
    with col3:
        st.metric("Fidelity", f"{fidelity:.3f}", f"{fidelity*100:.1f}%")
