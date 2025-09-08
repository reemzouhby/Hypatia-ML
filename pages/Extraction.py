import streamlit as st
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from art.estimators.classification import KerasClassifier
from art.attacks.extraction import CopycatCNN
import tensorflow as tf

# 🛠 Clear previous sessions
tf.keras.backend.clear_session()

st.title("🔓 Lightweight CopyCatCNN Attack Demo")

# Load victim model
@st.cache_resource
def load_victim_model():
    model = load_model("pages/mnist_model.h5")
    return model

victim_model = load_victim_model()
classifier_victim = KerasClassifier(model=victim_model, clip_values=(0,1))

# Load MNIST data
@st.cache_data
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test[:100] / 255.0  # Only 100 samples
    x_test = x_test.reshape(-1,28,28,1)
    y_test = y_test[:100]
    return x_test, y_test

x_steal, y_test = load_data()

# Build small stolen model
def get_stolen_model():
    tf.keras.backend.clear_session()
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

stolen_model = get_stolen_model()
classifier_stolen = KerasClassifier(model=stolen_model, clip_values=(0,1))

# Run CopyCatCNN
if st.button("🚀 Run Lightweight CopyCatCNN"):
    st.write("Running attack on 100 samples, batch_size=16, epochs=1...")
    attack = CopycatCNN(
        classifier=classifier_victim,
        nb_epochs=1,
        batch_size_fit=16,
        batch_size_query=16,
        use_probability=True,
        nb_stolen=100
    )
    classifier_stolen = attack.extract(thieved_classifier=classifier_stolen, x=x_steal)
    
    # Evaluate stolen model
    y_test_cat = to_categorical(y_test, num_classes=10)
    loss, acc = classifier_stolen._model.evaluate(x_steal, y_test_cat, verbose=0)
    st.success(f"✅ Attack completed! Stolen model accuracy: {acc*100:.2f}%")
