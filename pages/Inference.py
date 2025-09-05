import  streamlit as st
import os
from sklearn.model_selection import train_test_split

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import pandas as pd
from art.estimators.classification import KerasClassifier
#import attacks Membership
from art.attacks.inference.membership_inference import LabelOnlyDecisionBoundary,MembershipInferenceBlackBox,MembershipInferenceBlackBoxRuleBased,ShadowModels
from art.attacks.inference.attribute_inference import AttributeInferenceBaseline
from art.attacks.inference.model_inversion import MIFace
warnings.filterwarnings('ignore')
from art.utils import to_categorical
st.set_page_config(
    page_title="Inference Attacks on MNIST",
    page_icon="🕵️",
    layout="wide"
)

st.title(" 🕵️ Inference Attacks on MNIST")
st.markdown("---")
def evaluate_attack_results(overall_acc, members_acc, non_members_acc):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Members Accuracy", f"{members_acc:.3f}", f"{members_acc * 100:.1f}%")
    with col2:
        st.metric("Non-Members Accuracy", f"{non_members_acc:.3f}", f"{non_members_acc * 100:.1f}%")
    with col3:
        st.metric("Overall Accuracy", f"{overall_acc:.3f}", f"{overall_acc * 100:.1f}%")
    st.markdown("---")
    if overall_acc >= 0.75: #attack njht y3ni 3erf ymayz inu hul data mch mdarab 3lehun
        st.error("🔴 CRITICAL ATTACK SUCCESS")
        st.error("❌ MODEL VERY VULNERABLE")
        st.error("🚨 URGENT: Apply privacy protection immediately!")
        st.error(f"Attack can distinguish members with {overall_acc:.1%} accuracy")

    elif overall_acc >= 0.65:
        st.warning("🟠 HIGH ATTACK SUCCESS")
        st.warning("⚠️ MODEL VULNERABLE")
        st.warning("🔧 CRITICAL: Apply privacy defenses ")
        st.warning(f"Attack shows strong membership inference capability")

    elif overall_acc >= 0.58:
        st.info("🟡 MODERATE ATTACK SUCCESS")
        st.info("🔸 MODEL SOMEWHAT VULNERABLE")
        st.info("💡 RECOMMENDED: Improve model privacy (dropout, noise)")
        st.info(f"Attack has noticeable success in membership detection")

    elif overall_acc >= 0.53:
        st.success("🔵 WEAK ATTACK SUCCESS")
        st.success("✅ MODEL FAIRLY ROBUST")
        st.info("👍 GOOD: Attack slightly better than random but low threat")
        st.info(f"Minimal privacy leakage detected")

    else:
        st.success("🟢 ATTACK FAILED")
        st.success("🛡️ MODEL HIGHLY ROBUST")
        st.success("🎉 EXCELLENT: Strong privacy protection!")

@st.cache_resource
def load_model():
    """Load model with caching"""
    try:
        model = tf.keras.models.load_model("pages/mnist_model.h5")
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

# Get data from session state
classifier = st.session_state.classifier
train_images, train_labels = st.session_state.train_data
test_images, test_labels = st.session_state.test_data
model=st.session_state.model
st.sidebar.header("⚔️ Attack Configuration")
attack_type = st.sidebar.radio(
    "🎯 Attack Type:",
    ["Membership Inference", "Attribute Inference", "Model Inversion"],
    help="""Choose your inference attack strategy:

🕵️ Membership Inference: Determine if specific data was used in training
🔎 Attribute Inference: Infer sensitive attributes from model outputs  
🔄 Model Inversion: Reconstruct training samples from model behavior
"""
)

#after select each type has a attak type
param={}
if attack_type == "Membership Inference":
    st.sidebar.markdown("### 🕵️ Membership Inference Settings")
    model_type = st.sidebar.radio(
        "🧩 Which model do you want to attack?",
        ["Normal Model", "Shadow Model"],
        help="Choose whether to attack a normal trained model or use Shadow Models"
    )
    if model_type=="Normal Model":
        method = st.sidebar.selectbox("Select Method", options=["Black-box", "Decision Boundary"])
        if method == "Black-box":
            st.sidebar.markdown("#### ⚙️ Black-box Configuration")
            # 2 option balck bozx w rule based
            options = ["MembershipInferenceBlackBox",
                       "MembershipInferenceBlackBoxRuleBased"]
            attack_mode = st.sidebar.selectbox("Select Attack", options, help="""MembershipInferenceBlackBox: ML-based black-box attack
        MembershipInferenceBlackBoxRuleBased: Threshold-based rules""")

            if attack_mode == "MembershipInferenceBlackBox":

                param["input_type"] = st.sidebar.selectbox("📥 Input Type:", ["prediction", "loss"],
                                                           help="prediction: Use model outputs\nloss")

                param["attack_mode_type"] = st.sidebar.selectbox(" Attack Model:",
                                                                 ["nn", "rf", "gb", "dt", "knn", "svm"],
                                                                 help="Neural Network: Flexible\nRandom Forest: Fast\nGradient Boosting: Good performance")

                if param["attack_mode_type"] == "nn":
                    param["nn_epoch"] = st.sidebar.slider("🔄 Epochs:", 3, 10, 3, 1)
                    param["nn_batch"] = st.sidebar.selectbox("📦 Batch Size:", [16, 32, 64, 128], index=1)

            elif attack_mode == "MembershipInferenceBlackBoxRuleBased":
                st.sidebar.write("Membership Inference Black Box Rule Based")
                st.sidebar.info(
                    "✅ No additional parameters needed. This attack uses a simple rule: if the model predicts correctly, the sample is likely a member.")

        elif method == "Decision Boundary":
            st.sidebar.markdown("#### 🎯 Decision Boundary Configuration")
            attack_mode = st.sidebar.selectbox("Select Attack", ["LabelOnlyDecisionBoundary"],
                                               help="LabelOnlyDecisionBoundary: Uses only prediction labels")
            if attack_mode == "LabelOnlyDecisionBoundary":
                decision = st.sidebar.radio("Distance Threshold:",
                                            ["Supervised Calibration", "Unsupervised Calibration"],
                                            help="""
        Choose how to set the decision boundary threshold:
        - Supervised Calibration: Uses some known train/test data to optimize threshold
        - Unsupervised Calibration: Uses randomly generated samples to optimize threshold
        """)
                if decision == "Supervised Calibration":
                    param["dec"] = "super"
                    st.sidebar.write("Will calibrate threshold using known training and test data")
                elif decision == "Unsupervised Calibration":
                    st.sidebar.write("Will calibrate threshold using random samples (unsupervised)")
                    param["dec"] = "unsuper"
                    param["num_samples"] = st.sidebar.slider(
                        "Number of random samples:", 10, 500, 100, 10,
                        help="Generate random samples to estimate threshold"
                    )
                    param["top_t"] = st.sidebar.slider(
                        "Top-t percentile:", 1, 100, 50, 1,
                        help="Percentile of distances to set as threshold. Higher = stricter member detection"
                    )
                    param["max_queries"] = st.sidebar.slider(
                        "Max queries per sample:", 1, 10, 1, 1,
                        help="Maximum HopSkipJump iterations per sample"
                    )

    else: #shadow model
        st.sidebar.markdown("#### 👤 Shadow Model Configuration")
        attack_mode = st.sidebar.selectbox("Select Attack", ["ShadowModels"],
                                           help="ShadowModels: Advanced attack using trained shadow models")
        if attack_mode == "ShadowModels":
            param["num_shadow"] = st.sidebar.slider("Number of Shadow Models", 1, 5, 3)
            param["disjoint"] = st.sidebar.checkbox("Disjoint datasets", True)
            shadow_attack = st.sidebar.selectbox(
                "🎯 Attack on Shadow Model",
                ["Membership Black-box" ,"Decision Boundary"]
            )
            if shadow_attack=="Membership Black-box":
                param["input_type"] = st.sidebar.selectbox("📥 Input Type:", ["prediction", "loss"],
                                                           help="prediction: Use model outputs\nloss")

                param["attack_mode_type"] = st.sidebar.selectbox(" Attack Model:",
                                                                 ["nn", "rf", "gb", "dt", "knn", "svm"],
                                                                 help="Neural Network: Flexible\nRandom Forest: Fast\nGradient Boosting: Good performance")

                if param["attack_mode_type"] == "nn":
                    param["nn_epoch"] = st.sidebar.slider("🔄 Epochs:", 3, 10, 3, 1)
                    param["nn_batch"] = st.sidebar.selectbox("📦 Batch Size:", [16, 32, 64, 128], index=1)
            elif shadow_attack=="Decision Boundary":
                st.sidebar.markdown("#### 🎯 Decision Boundary Configuration")
                attack_mode = st.sidebar.selectbox("Select Attack", ["LabelOnlyDecisionBoundary"],
                                                   help="LabelOnlyDecisionBoundary: Uses only prediction labels")
                if attack_mode == "LabelOnlyDecisionBoundary":
                    decision = st.sidebar.radio("Distance Threshold:",
                                                ["Supervised Calibration", "Unsupervised Calibration"],
                                                help="""
                        Choose how to set the decision boundary threshold:
                        - Supervised Calibration: Uses some known train/test data to optimize threshold
                        - Unsupervised Calibration: Uses randomly generated samples to optimize threshold
                        """)
                    if decision == "Supervised Calibration":
                        param["dec"] = "super"
                        st.sidebar.write("Will calibrate threshold using known training and test data")
                    elif decision == "Unsupervised Calibration":
                        st.sidebar.write("Will calibrate threshold using random samples (unsupervised)")
                        param["dec"] = "unsuper"
                        param["num_samples"] = st.sidebar.slider(
                            "Number of random samples:", 10, 500, 100, 10,
                            help="Generate random samples to estimate threshold"
                        )
                        param["top_t"] = st.sidebar.slider(
                            "Top-t percentile:", 1, 100, 50, 1,
                            help="Percentile of distances to set as threshold. Higher = stricter member detection"
                        )
                        param["max_queries"] = st.sidebar.slider(
                            "Max queries per sample:", 1, 10, 1, 1,
                            help="Maximum HopSkipJump iterations per sample")
if attack_type == "Attribute Inference":
    st.sidebar.markdown("### 🔎 Attribute Inference Settings")
    attack_mode = st.sidebar.selectbox("Select Attack", options=["Baseline"],
                                       help="AttributeInferenceBaseline: Infer hidden features using neural network")

    if attack_mode == "Baseline":
        st.sidebar.markdown("#### ⚙️ Baseline Configuration")

        # Feature selection
        param["attack_feature"] = st.sidebar.slider(
            "🎯 Target Feature (Pixel Position):",
            0, 783, 200, 1,  # 28*28-1 = 783
            help="Which pixel position to try to infer (0-783). This pixel will be hidden from the attacker."
        )

        # Show pixel position info
        row = param["attack_feature"] // 28
        col = param["attack_feature"] % 28
        st.sidebar.info(f"📍 Pixel at position: Row {row}, Column {col}")

        param["is_continuous"] = st.sidebar.checkbox(
            "Continuous Feature", True,
            help="Check if the feature values are continuous (pixel intensities 0-1)"
        )

        param["nn_epochs"] = st.sidebar.slider(
            "🔄 Neural Network Epochs:",
            5, 50, 10, 5,
            help="Number of training epochs for the attack model"
        )

        param["train_size"] = st.sidebar.slider(
            "📊 Training Sample Size:",
            1000, 10000, 5000, 500,
            help="Number of training samples to use for the attack"
        )
elif attack_type=="Model Inversion":
    st.sidebar.markdown("### 🔄 Model Inversion Settings")

    # Parameters
    param["max_iter"] = st.sidebar.slider("🔄 Max Iterations:", 100, 10000, 3000, 100)
    param["threshold"] = st.sidebar.slider("📉 Threshold:", 0.0, 1.0, 0.99, 0.01)
    ignore_threshold = st.sidebar.checkbox("🚫 Ignore Threshold", False)

    if ignore_threshold:
        param["threshold"] = 1.  # force attack to ignore convergence

    # Target class
    param["target_digits"] = st.sidebar.multiselect(
        "🎯 Target Digits (you can select multiple):",
        options=list(range(10)),
        default=[7]
    )

    if len(param["target_digits"]) == 0:
        st.warning("Please select at least one target digit!")

    # Initialization choice
    param["x_option"] = st.sidebar.radio(
        "🖼️ Initialization Image:",
        ["None", "Black", "White", "Gray", "Random"],
        help="""Choose how to initialize the attack:
            - None: let ART handle it
            - Black: start from zeros
            - White: start from ones
            - Gray: start from 0.5
            - Random: uniform noise [0,1]"""
    )









run_button = st.button("🚀 Run  Attack", type="primary")
if run_button:

    if attack_type == "Membership Inference":
        if model_type == "Normal Model":
            st.warning("⚠️ This attack may take 1-3 minutes to complete. Please be patient and don't refresh the page.")

            if attack_mode == "MembershipInferenceBlackBox":
                with st.spinner("⏳ Running " + attack_mode + " attack... Please wait"):

                    attack_train_size = 500
                    attack_test_size = 500
                    attack_args = {
                        "estimator": classifier,
                        "attack_model_type": param["attack_mode_type"],
                        "input_type": param["input_type"]
                    }
                    # only add if  nn
                    if param["attack_mode_type"] == "nn":
                        attack_args["nn_model_epochs"] = param["nn_epoch"]
                        attack_args["nn_model_batch_size"] = param["nn_batch"]

                    attack = MembershipInferenceBlackBox(**attack_args)
                    # train  attck model
                    attack.fit(train_images[:attack_train_size], train_labels[:attack_train_size],
                               test_images[:attack_test_size],
                               test_labels[:attack_test_size])
                    # get infered value
                    # test fir the rest of data in training and for testinf for the training it should give 1
                    infer_train = attack.infer(train_images[attack_train_size:], train_labels[attack_train_size:])
                    infer_test = attack.infer(test_images[attack_test_size:], test_labels[attack_test_size:])
                    # For members (train data): correct prediction = 1, wrong = 0
                    train_acc = np.mean(infer_train)  # Same as np.sum(infer_train) / len(infer_train)

                    # For non-members (test data): correct prediction = 0, wrong = 1
                    test_acc = np.mean(infer_test == 0)  # Fraction of predictions that are 0
                    # Overall accuracy
                    overall_acc = (train_acc * len(infer_train) + test_acc * len(infer_test)) / (
                            len(infer_train) + len(infer_test))
                    evaluate_attack_results(overall_acc, train_acc, test_acc)

            elif attack_mode == "MembershipInferenceBlackBoxRuleBased":
                with st.spinner("⏳ Running " + attack_mode + " attack... Please wait"):
                    attack = MembershipInferenceBlackBoxRuleBased(classifier)
                    train_idx = np.random.choice(len(train_images), size=5000, replace=False)
                    test_idx = np.random.choice(len(test_images), size=5000, replace=False)
                    infer_train = attack.infer(train_images[train_idx], train_labels[train_idx])
                    infer_test = attack.infer(test_images[test_idx], test_labels[test_idx])
                    train_acc = np.mean(infer_train)  # Same as np.sum(infer_train) / len(infer_train)
                    # For non-members (test data): correct prediction = 0, wrong = 1
                    test_acc = np.mean(infer_test == 0)  # Fraction of predictions that are 0
                    # Overall accuracy
                    overall_acc = (train_acc * len(infer_train) + test_acc * len(infer_test)) / (
                            len(infer_train) + len(infer_test))
                    evaluate_attack_results(overall_acc, train_acc, test_acc)

            elif attack_mode == "LabelOnlyDecisionBoundary":
                with st.spinner("⏳ Running " + attack_mode + " attack... Please wait"):
                    attack = LabelOnlyDecisionBoundary(classifier)

                    if param["dec"] == "super":
                        attack.calibrate_distance_threshold(
                            train_images[:200], train_labels[:200],
                            test_images[:200], test_labels[:200]
                        )
                    elif param["dec"] == "unsuper":
                        attack.calibrate_distance_threshold_unsupervised(
                            num_samples=param["num_samples"],
                            top_t=param["top_t"],
                            max_queries=param["max_queries"]
                        )

                    infer_train = attack.infer(train_images[:5000], train_labels[:5000])
                    infer_test = attack.infer(test_images[:5000], test_labels[:5000])

                    train_acc = np.mean(infer_train)
                    test_acc = np.mean(infer_test == 0)
                    overall_acc = (train_acc * len(infer_train) + test_acc * len(infer_test)) / (
                            len(infer_train) + len(infer_test)
                    )

                    evaluate_attack_results(overall_acc, train_acc, test_acc)

        elif model_type == "Shadow Model":
            classifier = KerasClassifier(model=model, clip_values=(0, 1))
            shadow = ShadowModels(classifier, num_shadow_models=param["num_shadow"],
                                  disjoint_datasets=param["disjoint"])
            X = np.concatenate([train_images, test_images])
            y = np.concatenate([train_labels, test_labels])
            # train _test split return X_train, X_test, y_train, y_test
            target_images, shadow_images, target_label, shadow_labels = train_test_split(X, y, test_size=0.7,
                                                                                         random_state=42)
            target_train_size = len(target_images) // 2
            x_target_train = target_images[:target_train_size]
            y_target_train = target_label[:target_train_size]
            x_target_test = target_images[target_train_size:]
            y_target_test = target_label[target_train_size:]
            # disjoint --- bool if true every model trianed with diffferent dataset no overlapping


            shadow_model = ShadowModels(classifier, num_shadow_models=3, disjoint_datasets=True)
            shadow_dataset = shadow_model.generate_shadow_dataset(shadow_images, shadow_labels)
            (member_x, member_y, member_predictions), (
            nonmember_x, nonmember_y, nonmember_predictions) = shadow_dataset

            from keras.utils import to_categorical  # becaue classifier 3ndi softmax

            member_y = to_categorical(member_y, num_classes=10)
            nonmember_y = to_categorical(nonmember_y, num_classes=10)
            if shadow_attack == "Membership Black-box":
                with st.spinner("⏳ Running Membership Black-box with Shadow Models... Please wait"):

                    attack_args = {
                        "estimator": classifier,
                        "attack_model_type": param["attack_mode_type"],
                        "input_type": param["input_type"]
                    }
                    if param["attack_mode_type"] == "nn":
                        attack_args["nn_model_epochs"] = param["nn_epoch"]
                        attack_args["nn_model_batch_size"] = param["nn_batch"]
                    attack = MembershipInferenceBlackBox(**attack_args)
                    attack.fit(member_x,member_y,nonmember_x,nonmember_y,member_predictions,nonmember_predictions)
                    infer_train = attack.infer(x_target_train, y_target_train)
                    infer_test = attack.infer(x_target_test, y_target_test)
                    train_acc = np.mean(infer_train)
                    test_acc = np.mean(infer_test == 0)
                    overall_acc = (train_acc * len(infer_train) + test_acc * len(infer_test)) / (
                            len(infer_train) + len(infer_test)
                    )
                    evaluate_attack_results(overall_acc, train_acc, test_acc)

            elif shadow_attack == "Decision Boundary":
                with st.spinner("⏳ Running Decision Boundary with Shadow Models... Please wait"):

                    attack = LabelOnlyDecisionBoundary(classifier)
                    if param["dec"] == "super":
                        attack.calibrate_distance_threshold(
                            member_x[:200], member_y[:200],
                            nonmember_x[:200], nonmember_y[:200]
                        )
                    elif param["dec"] == "unsuper":
                        attack.calibrate_distance_threshold_unsupervised(
                            x=member_x[:param["num_samples"]],
                            num_samples=param["num_samples"],
                            top_t=param["top_t"],
                            max_queries=param["max_queries"]
                        )

                    infer_train = attack.infer(x_target_train, y_target_train)
                    infer_test = attack.infer(x_target_test, y_target_test)

                    train_acc = np.mean(infer_train)
                    test_acc = np.mean(infer_test == 0)
                    overall_acc = (train_acc * len(infer_train) + test_acc * len(infer_test)) / (
                            len(infer_train) + len(infer_test)
                    )

                    evaluate_attack_results(overall_acc, train_acc, test_acc)
    elif attack_type == "Attribute Inference":
        st.warning("⚠️ This attack may take 2-5 minutes depending on training size. Please be patient.")

        if attack_mode == "Baseline":
            with st.spinner("⏳ Running Attribute Inference Baseline attack... Please wait"):
                # Prepare data - flatten images to 2D
                train_flat = train_images.reshape(train_images.shape[0], -1)
                test_flat = test_images.reshape(test_images.shape[0], -1)

                # Limit training size for performance
                train_size = min(param["train_size"], len(train_flat))
                train_subset = train_flat[:train_size]

                # Create attack
                attack = AttributeInferenceBaseline(
                    attack_feature=param["attack_feature"],
                    is_continuous=param["is_continuous"],
                    nn_model_epochs=param["nn_epochs"]
                )

                # Train attack model
                attack.fit(train_subset)

                # Test on subset of test data
                test_size = min(1000, len(test_flat))
                test_subset = test_flat[:test_size]

                # Get actual values for the target feature
                actual_values = test_subset[:, param["attack_feature"]]

                # Remove the target feature from test data
                test_without_feature = np.delete(test_subset, param["attack_feature"], axis=1)

                # Perform inference
                predicted_values = attack.infer(test_without_feature, values=None)

                # Calculate metrics
                mae = np.abs(actual_values - predicted_values).mean()
                mse = np.mean((actual_values - predicted_values) ** 2)

                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Absolute Error", f"{mae:.4f}")
                with col2:
                    st.metric("Mean Squared Error", f"{mse:.4f}")
                with col3:
                    accuracy = (1 - mae) * 100  # Convert to percentage
                    st.metric("Inference Accuracy", f"{accuracy:.1f}%")

                st.markdown("---")
                if mae < 0.05:  # nsbet la5ta2 alili so nhbt ta3rfa
                    st.error("🔴 CRITICAL ATTACK SUCCESS")
                    st.error("❌ FEATURE HIGHLY VULNERABLE")
                    st.error("🚨 URGENT: The attacker can accurately infer the hidden pixel!")


                elif mae < 0.10:
                    st.warning("🟠 HIGH ATTACK SUCCESS")
                    st.warning("⚠️ FEATURE VULNERABLE")
                    st.warning("🔧 CRITICAL: Apply privacy protection mechanisms")


                elif mae < 0.20:
                    st.info("🟡 MODERATE ATTACK SUCCESS")
                    st.info("🔸 FEATURE SOMEWHAT VULNERABLE")
                    st.info("💡 RECOMMENDED: Consider differential privacy or noise addition")


                elif mae < 0.30:
                    st.success("🔵 WEAK ATTACK SUCCESS")
                    st.success("✅ FEATURE FAIRLY ROBUST")
                    st.info("👍 GOOD: Attack has limited success")

                else:
                    st.success("🟢 ATTACK FAILED")
                    st.success("🛡️ FEATURE HIGHLY ROBUST")
                    st.success("🎉 EXCELLENT: Strong privacy protection!")
    elif attack_type == "Model Inversion":
        with st.spinner("⏳ Running Model Inversion... Please wait"):

            attack = MIFace(
                classifier=classifier,
                max_iter=param["max_iter"],
                threshold=param["threshold"]
            )
            target = param["target_digits"]
            target = to_categorical(np.array(target), nb_classes=10)
            x_option=param["x_option"]
            num_targets = len(param["target_digits"])
            if x_option== "None":
                x=None
            elif x_option== "Black": #1 black 0 white
                x = np.ones((num_targets, 28, 28, 1))
            elif x_option=="White":
                x=np.zeros((num_targets,28,28,1))
            elif x_option=="Gray":
                x = np.zeros((num_targets, 28, 28, 1)) + 0.5
            elif x_option=="Random":
                x=np.random.uniform(0, 1, (num_targets, 28, 28, 1))

            # initialize firt with no image , return is a 1 image
            reconstr = attack.infer(x=x, y=target)
            # visualize result
            plt.figure(figsize=(15, 15))
            num_cols = 5  # images per row

            for i in range(num_targets):
                pred = classifier.predict(np.expand_dims(reconstr[i], axis=0))
                pred = np.argmax(pred, axis=1)[0]


                if i % num_cols == 0:
                    cols = st.columns(num_cols)

                col = cols[i % num_cols]
                with col:
                    st.image(
                        reconstr[i].reshape(28, 28),
                        caption=f"Reconstructed digit {param['target_digits'][i]}, predict is {pred}",
                        width=150,
                        clamp=True
                    )

















