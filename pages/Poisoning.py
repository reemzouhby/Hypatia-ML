import streamlit as st
import os

from art.defences.trainer import AdversarialTrainerMadryPGD

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)           # disable threading
torch.multiprocessing.set_sharing_strategy("file_system")  # safer on Windows

st.set_page_config(
    page_title="Poisoning Attacks on MNIST",
    page_icon="☠️",
    layout="wide"
)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from art.utils import to_categorical
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.checkpoint import load_state_dict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from art.estimators.classification import PyTorchClassifier, KerasClassifier
from art.attacks.poisoning import GradientMatchingAttack, PoisoningAttackCleanLabelBackdoor, PoisoningAttackBackdoor,PoisoningAttackSVM, FeatureCollisionAttack
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import importlib
from art.attacks.poisoning import PoisoningAttackSVM
from art.estimators.classification.scikitlearn import ScikitlearnSVC, SklearnClassifier
from sklearn.svm import SVC
from sklearn.metrics import hinge_loss, accuracy_score
from tensorflow.python.keras.utils.np_utils import to_categorical
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
num_classes = 10


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # raw scores for CrossEntropyLoss
        return x


@st.cache_resource
def load_model_and_data():
    """Cache model and data loading - this is the biggest bottleneck"""
    device = torch.device("cpu")

    # Load model
    model = CNN().to(device)
    try:
        model.load_state_dict(torch.load("pages/mnist_cnn_pytorch.pth", map_location=device))
    except FileNotFoundError:
        st.warning("⚠️ Model file not found. Using randomly initialized model for demo.")
    model.eval()

    # Load MNIST data with reduced size for faster processing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Reduce dataset size for faster processing
    train_size = min(5000, len(train_dataset))
    test_size = min(1000, len(test_dataset))

    # Create smaller datasets
    train_indices = np.random.choice(len(train_dataset), train_size, replace=False)
    test_indices = np.random.choice(len(test_dataset), test_size, replace=False)

    x_train = torch.stack([train_dataset[i][0] for i in train_indices]).to(device)
    y_train = torch.tensor([train_dataset[i][1] for i in train_indices], dtype=torch.long).to(device)

    x_test = torch.stack([test_dataset[i][0] for i in test_indices]).to(device)
    y_test = torch.tensor([test_dataset[i][1] for i in test_indices], dtype=torch.long).to(device)

    # Create classifier
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
        clip_values=(-1, 1)
    )

    return model, classifier, x_train, y_train, x_test, y_test, device


@st.cache_data
def load_mnist_for_svm():
    """Cache MNIST data loading for SVM attack"""
    mnist = fetch_openml('mnist_784')
    X, y = mnist["data"], mnist["target"]
    X = np.array(X)
    y = np.array(y)
    y = y.astype(np.uint8)
    X = X.astype(np.float32) / 255.0
    return X, y

model, classifier, x_train, y_train, x_test, y_test, device = load_model_and_data()

def patch(patch_type, parameters, x):
    """Apply patch to input x - works with both torch tensors and numpy arrays"""
    # Handle both torch tensors and numpy arrays
    if isinstance(x, torch.Tensor):
        x = x.clone()
        is_torch = True
    else:
        x = x.copy()
        is_torch = False

    value = parameters.get("patch_value", 1)
    location = parameters.get("location", "top-left")

    if patch_type == "square":
        h = parameters.get("patch_height", 3)
        w = parameters.get("patch_width", 3)
        if location == "top-left":
            x[:, :, :h, :w] = value
        elif location == "top-right":
            x[:, :, :h, -w:] = value
        elif location == "bottom-left":
            x[:, :, -h:, :w] = value
        elif location == "bottom-right":
            x[:, :, -h:, -w:] = value
        elif location == "center":
            start_x = (28 - h) // 2
            start_y = (28 - w) // 2
            x[:, :, start_x:start_x + h, start_y:start_y + w] = value
        else:  # random
            if is_torch:
                start_x = torch.randint(0, 28 - h + 1, (1,)).item()
                start_y = torch.randint(0, 28 - w + 1, (1,)).item()
            else:
                start_x = np.random.randint(0, 28 - h + 1)
                start_y = np.random.randint(0, 28 - w + 1)
            x[:, :, start_x:start_x + h, start_y:start_y + w] = value

    elif patch_type == "pattern":
        h = parameters.get("patch_height", 4)
        w = parameters.get("patch_width", 4)
        for i in range(min(h, 28)):
            for j in range(min(w, 28)):
                if (i + j) % 2 == 0:
                    x[:, :, i, j] = value

    elif patch_type == "pixel":
        num_pixels = parameters.get("num_pixels", 5)
        for _ in range(num_pixels):
            if is_torch:
                px = torch.randint(0, 28, (1,)).item()
                py = torch.randint(0, 28, (1,)).item()
            else:
                px = np.random.randint(0, 28)
                py = np.random.randint(0, 28)
            x[:, :, px, py] = value

    return x


def create_perturbation_function(patch_type, parameters):
    """Create perturbation function for ART"""

    def perturbation_fn(x):
        return patch(patch_type, parameters, x)

    return perturbation_fn

st.title("☠️ Poisoning Attacks on MNIST")
st.markdown("---")
st.sidebar.header("⚔️ Attack Configuration")
attack_mode = st.sidebar.radio(
    "Attack Mode:",
    ["Untarget", "Target"],
    help="Untargeted: Try to misclassify to any wrong class\nTargeted: Try to misclassify to a specific target class"
)

# Set available attacks based on mode
if attack_mode == "Target":
    available = ["GradientMatchingAttack", "PoisoningAttackCleanLabelBackdoor", "PoisoningAttackBackdoor",
                 "FeatureCollisionAttack"]
else:
    available = ["Poisoning SVM Attack"]

# Attack type selection
attack_type = st.sidebar.selectbox("Select the type of Attack", options=available)


st.sidebar.subheader(f"{attack_type} Parameters")
parameters = {}
patch_choose = ["square", "pattern", "pixel"]

# Show parameters based on attack type
if attack_type == "PoisoningAttackBackdoor":
    patch_type = st.sidebar.selectbox("Select patch type", options=patch_choose)

    st.sidebar.subheader("📐 Patch Size Configuration")

    if patch_type == "square":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            parameters["patch_height"] = st.slider("Patch Height", min_value=1, max_value=10, value=3, key="height")
        with col2:
            parameters["patch_width"] = st.slider("Patch Width", min_value=1, max_value=10, value=3, key="width")

    elif patch_type == "pattern":
        parameters["patch_height"] = st.sidebar.slider("Pattern Height", min_value=2, max_value=8, value=4)
        parameters["patch_width"] = st.sidebar.slider("Pattern Width", min_value=2, max_value=8, value=4)

    elif patch_type == "pixel":
        parameters["num_pixels"] = st.sidebar.slider("Number of Pixels", min_value=1, max_value=20, value=5)

    # Patch location
    st.sidebar.subheader("📍 Patch Location")
    location_options = ["random", "top-left", "top-right", "bottom-left", "bottom-right", "center"]
    parameters["location"] = st.sidebar.selectbox("Patch Location:", options=location_options)

    # Other backdoor parameters
    st.sidebar.subheader("🎯 Attack Parameters")
    parameters["target_class"] = st.sidebar.selectbox("Target Class:", options=list(range(10)))
    parameters["percent_poison"] = st.sidebar.slider("Poison Percentage (%)", min_value=1, max_value=50, value=10)

    # Intensity of the patch
    parameters["patch_value"] = st.sidebar.slider("Patch Intensity", min_value=-1.0, max_value=1.0, value=1.0, step=0.1)

elif attack_type == "GradientMatchingAttack":
    parameters["source_class"] = st.sidebar.selectbox("Source Class (to be poisoned):", options=list(range(10)),
                                                      index=7)
    parameters["target_class"] = st.sidebar.selectbox("Target Class (desired prediction):", options=list(range(10)),
                                                      index=2)

    parameters["percent_poison"] = st.sidebar.slider("Poison Percentage (%)", min_value=1, max_value=20, value=10)
    parameters["epsilon"] = st.sidebar.slider("Epsilon (perturbation strength)", min_value=0.1, max_value=1.0,
                                              value=0.3, step=0.1)


elif attack_type == "PoisoningAttackCleanLabelBackdoor":
    patch_type = st.sidebar.selectbox("Select patch type", options=patch_choose)

    st.sidebar.subheader("📐 Patch Size Configuration")

    if patch_type == "square":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            parameters["patch_height"] = st.slider("Patch Height", min_value=1, max_value=10, value=3, key="height")
        with col2:
            parameters["patch_width"] = st.slider("Patch Width", min_value=1, max_value=10, value=3, key="width")

    elif patch_type == "pattern":
        parameters["patch_height"] = st.sidebar.slider("Pattern Height", min_value=2, max_value=8, value=4)
        parameters["patch_width"] = st.sidebar.slider("Pattern Width", min_value=2, max_value=8, value=4)

    elif patch_type == "pixel":
        parameters["num_pixels"] = st.sidebar.slider("Number of Pixels", min_value=1, max_value=20, value=5)

    # Patch location
    st.sidebar.subheader("📍 Patch Location")
    location_options = ["random", "top-left", "top-right", "bottom-left", "bottom-right", "center"]
    parameters["location"] = st.sidebar.selectbox("Patch Location:", options=location_options)

    # Other backdoor parameters
    st.sidebar.subheader("🎯 Attack Parameters")
    parameters["target_class"] = st.sidebar.selectbox("Target Class:", options=list(range(10)))
    parameters["percent_poison"] = st.sidebar.slider("Poison Percentage (%)", min_value=1, max_value=50, value=10)

    # Intensity of the patch
    parameters["patch_value"] = st.sidebar.slider("Patch Intensity", min_value=-1.0, max_value=1.0, value=1.0, step=0.1)

elif attack_type == "FeatureCollisionAttack":
# targetclass, baseclass,max iteration
    st.sidebar.subheader("🎯 Attack Parameters")
    parameters["target_class"] = st.sidebar.selectbox("Target Class(to be missclassify):", options=list(range(10)))
    parameters["base_class"] = st.sidebar.selectbox("Base Class(to be predict):", options=list(range(10)))
    parameters["max_iter"] = st.sidebar.slider("Max Iterations", 10, 100, 50, 10)
    parameters["percent_poison"] = st.sidebar.slider("Poison Percentage (%)", min_value=1, max_value=50, value=10)

elif attack_type == "Poisoning SVM Attack":
    st.sidebar.info("this attack work with binary classification so if u want to test it should choose 2 nb  ")
    parameters["nb1"]=st.sidebar.selectbox("choose nb1 ",options=list(range(10)))
    parameters["nb2"] = st.sidebar.selectbox("choose nb2 ", options=list(range(10)))
    parameters["percent_poison"] = st.sidebar.slider("Poison Percentage (%)", min_value=1, max_value=20, value=10)
    parameters["epsilon"] = st.sidebar.slider("Epsilon (perturbation strength)", min_value=1, max_value=10,
                                                                                 value=1, step=1)
    parameters["max_iter"] = st.sidebar.slider("Max Iterations", 10, 100, 50, 10)





# Run button
run_button = st.button("🚀 Run " + attack_type + " Attack", type="primary")

# Attack execution
if run_button:
    if attack_type == "PoisoningAttackBackdoor":
        with st.spinner("⏳ Running " + attack_type + " attack... Please wait"):
            # Create perturbation function
            perturbation_fn = create_perturbation_function(patch_type, parameters)

            # Initialize attack
            attack = PoisoningAttackBackdoor(perturbation=perturbation_fn)

            target_class = parameters.get("target_class")
            poison_fraction = parameters.get("percent_poison") / 100.0
            nb_poisoning = int(len(x_train) * poison_fraction)
            st.info(f"🧪 Creating {nb_poisoning} poisoned samples targeting class {target_class}...")

            # Select random training images to poison
            idx = np.random.choice(len(x_train), nb_poisoning, replace=False)
            x_poison = x_train[idx].cpu().numpy()
            y_poison = np.zeros((nb_poisoning, num_classes))
            y_poison[:, target_class] = 1

            # Apply poisoning
            x_poison_modified, _ = attack.poison(x_poison, y_poison)
            x_poison_torch = torch.tensor(x_poison_modified, dtype=torch.float32).to(device)
            y_poison_labels = torch.full((nb_poisoning,), target_class, dtype=torch.long).to(device)

            # Combine clean + poisoned data
            x_train_poisoned = torch.cat([x_train, x_poison_torch], dim=0)
            y_train_poisoned = torch.cat([y_train, y_poison_labels], dim=0)

            # Retrain the classifier with poisoned data
            classifier.fit(x_train_poisoned.cpu().numpy(), y_train_poisoned.cpu().numpy(), batch_size=128,
                           nb_epochs=10)

            # Test accuracy on clean data
            clean_predictions = classifier.predict(x_test.cpu().numpy())
            clean_acc = (clean_predictions.argmax(1) == y_test.cpu().numpy()).mean()

            # Test backdoor success rate
            mask_non_target = (y_test.cpu().numpy() != target_class)
            x_test_subset = x_test[mask_non_target]

            # Apply patch to test images
            x_triggered = patch(patch_type, parameters, x_test_subset)
            triggered_predictions = classifier.predict(x_triggered.cpu().numpy())
            asr = (triggered_predictions.argmax(1) == target_class).mean()

            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Clean Test Accuracy", f"{clean_acc:.3f}")
            with col2:
                st.metric("Attack Success Rate (ASR)", f"{asr:.3f}")

            # Visualization
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))

            for i in range(5):
                # Original image
                orig_img = x_test_subset[i].cpu().numpy().squeeze()
                axes[0, i].imshow(orig_img, cmap='gray')
                axes[0, i].set_title(f"Original\nTrue: {y_test[mask_non_target][i].item()}")
                axes[0, i].axis('off')

                # Triggered image
                trig_img = x_triggered[i].cpu().numpy().squeeze()
                pred_label = triggered_predictions[i].argmax()
                color = 'green' if pred_label == target_class else 'red'
                axes[1, i].imshow(trig_img, cmap='gray')
                axes[1, i].set_title(f"Triggered\nPred: {pred_label}", color=color)
                axes[1, i].axis('off')

            plt.tight_layout()
            st.pyplot(fig)

            if asr > 0.8:
                st.success(f"🎉 Attack successful! ASR: {asr:.1%}")
            elif asr > 0.5:
                st.warning(f"⚠️ Partial success. ASR: {asr:.1%}")
            else:
                st.error(f"❌ Attack failed. ASR: {asr:.1%}")

    elif attack_type=="FeatureCollisionAttack":
        with st.spinner("⏳ Running FeatureCollisionAttack... Please wait"):
            max_it = parameters.get("max_iter", 50)
            target_class = parameters.get("target_class")
            base_class = parameters.get("base_class")
            if target_class == base_class:
                st.error("❌ Target class and base class cannot be the same!")
                st.stop()
            target_index = np.where(y_train.cpu().numpy() == target_class)[0][0]
            x_target = x_train[target_index].unsqueeze(0).to(device)  # [1, 1, 28, 28]
            y_target = y_train[target_index].unsqueeze(0).to(device)

            base_indices = np.where(y_train.cpu().numpy() == base_class)[0][:100]
            x_base = x_train[base_indices].to(device)  # [5, 1, 28, 28]
            y_base = y_train[base_indices].to(device)

            print(f"Target sample: class {target_class}")
            print(f"Base samples: {len(x_base)} samples of class {base_class}")

            # Convert to numpy arrays and ensure proper format for ART
            x_target_np = x_target.cpu().numpy()
            x_base_np = x_base.cpu().numpy()
            y_base_np = y_base.cpu().numpy()

            # For ART, we need one-hot encoded labels for the base samples
            y_base_onehot = np.eye(num_classes)[y_base_np]

            attack = FeatureCollisionAttack(classifier, target=x_target_np, feature_layer=-2, max_iter=max_it)
            x_poison, y_poison = attack.poison(x_base_np, y_base_onehot)

            # Convert back to torch tensors
            x_poison_torch = torch.from_numpy(x_poison).to(device)
            y_poison_torch = torch.from_numpy(y_poison).to(device)

            x_train_poisoned = torch.cat([x_train, x_poison_torch], dim=0)
            y_train_poisoned = torch.cat([F.one_hot(y_train, num_classes=10).float(), y_poison_torch], dim=0)

            # Retrain the classifier with poisoned data
            classifier.fit(x_train_poisoned.cpu().numpy(), y_train_poisoned.cpu().numpy(), batch_size=128, nb_epochs=10)

            # Get multiple target samples (class 3) for testing
            target_indices = np.where(y_train.cpu().numpy() == target_class)[0][:40]  # Get 10 samples of class 3
            x_targets_test = x_train[target_indices].to(device)
            y_targets_test = y_train[target_indices].to(device)

            # Convert to numpy for ART
            x_targets_test_np = x_targets_test.cpu().numpy()

            # Get predictions after the attack
            predictions = classifier.predict(x_targets_test_np)
            predicted_classes = np.argmax(predictions, axis=1)

            fig, axes = plt.subplots(4, 5, figsize=(12, 8))
            axes = axes.flatten()

            successful_misclassifications = 0
            for i in range(len(x_targets_test)):
                axes[i].imshow(x_targets_test[i].cpu().numpy().squeeze(), cmap="gray")
                true_class = target_class
                pred_class = predicted_classes[i]

                if pred_class == base_class:
                    color = 'green'
                    successful_misclassifications += 1
                    title = f"T:{true_class} → P:{pred_class} ✓"
                else:
                    color = 'red'
                    title = f"T:{true_class} → P:{pred_class} ✗"

                axes[i].set_title(title, color=color, fontsize=9)
                axes[i].axis("off")

            plt.tight_layout()
            st.pyplot(fig)

            asr = successful_misclassifications / len(x_targets_test)
            st.metric("Attack Success Rate (ASR)", f"{asr:.3f}")

            if asr > 0.8:
                st.success(f"🎉 Attack successful! ASR: {asr:.1%}")
            elif asr > 0.5:
                st.warning(f"⚠️ Partial success. ASR: {asr:.1%}")
            else:
                st.error(f"❌ Attack failed. ASR: {asr:.1%}")



    elif attack_type == "Poisoning SVM Attack":
        st.info("""💡 Hint: The success of the poisoning attack depends on the chosen digits.
If you pick **similar-looking digits** (e.g., 1 vs 7, or 3 vs 5), the attack is usually more effective because the classifier’s decision boundary is less clear.
If you pick **well-separated digits** (e.g., 0 vs 8), the attack will likely be weaker.
""")
        with st.spinner("⏳ Running PoisonningSVM... Please wait"):
            # load data  from sklearn
            mnist = fetch_openml('mnist_784')
            X, y = mnist["data"], mnist["target"]
            X = np.array(X)
            y = np.array(y)
            y = y.astype(np.uint8)
            X = X.astype(np.float32) / 255.0

            # note svm only work with binary classfication so make sure that you work on binary classification so  we ppick 2 labls for exp 7 and 1 to test and do poisoning into it
            # choose two digits for binary classification so nb predict as 2 or 7 only so we take the nb has label 7 and 2
            target_digit1 = parameters["nb1"]
            target_digit2 = parameters["nb2"]
            if target_digit1 == target_digit2:
                st.error("❌ the 2 nb  cannot be the same!")
                st.stop()

            X1, y1 = X[y == target_digit1], y[y == target_digit1]
            X2, y2 = X[y == target_digit2], y[y == target_digit2]
            # because on art w e us eone hot encodeing we should transform 2 and 7 as 2--0 and 7--1
            y1 = np.zeros_like(y1)
            y2 = np.ones_like(y2)
            # now if nb ==7 predict 1 and if nb ==2 predict 0
            # Part 2: Train/Val/Test split 50 , 200, 1000 for testing
            X_train = np.concatenate([X1[:500], X2[:500]])
            y_train = np.concatenate([y1[:500], y2[:500]])

            X_val = np.concatenate([X1[50:250], X2[50:250]])
            y_val = np.concatenate([y1[50:250], y2[50:250]])

            X_test = np.concatenate([X1[250:1250], X2[250:1250]])
            y_test = np.concatenate([y1[250:1250], y2[250:1250]])

            # shuffle , reorder the nb without order for make the training more robust
            perm1 = np.random.permutation(X_train.shape[0])
            perm2 = np.random.permutation(X_val.shape[0])
            perm3 = np.random.permutation(X_test.shape[0])

            X_train, y_train = X_train[perm1], y_train[perm1]
            X_val, y_val = X_val[perm2], y_val[perm2]
            X_test, y_test = X_test[perm3], y_test[perm3]

            # Part 3: Train baseline SVM

            svm_clf = SVC(kernel="rbf", C=1.0, random_state=42)
            svm_clf.fit(X_train, y_train)
            # for test the acc on testing
            y_pred = svm_clf.predict(X_test)
            acc_clean = accuracy_score(y_test, y_pred)
            print(f"[Baseline] Accuracy: {acc_clean:.2%}")
            # classifier fr start the attack

            # make the y-train and y-val onehot encode for the art
            y_train_oh = to_categorical(y_train, num_classes=2)
            y_val_oh = to_categorical(y_val, num_classes=2)
            y_test_oh = to_categorical(y_test, num_classes=2)
            classfier = ScikitlearnSVC(model=SVC(kernel="linear", C=1.0), clip_values=(0.0, 1.0))
            classfier.fit(X_train, y_train_oh)
            attack = PoisoningAttackSVM(classifier=classfier, eps=parameters.get("epsilon"), x_train=X_train, y_train=y_train_oh, x_val=X_val,
                                        y_val=y_val_oh, max_iter=parameters.get("max_iter"), step=0.1)
            # attack.generateattack hyd btal2 bas sample wehde poisoning
            percent=parameters.get("percent_poison")/100.0

            pois_sample=int(len(X_train) *percent)
            x_poison, y_poison = attack.poison(X_train[:pois_sample], y_train_oh[:pois_sample])
            # combine poisoning with training
            x_train_poisoned = np.concatenate([X_train, x_poison])
            y_train_poisoned = np.concatenate([y_train_oh, y_poison])
            # retrain model
            classfier.fit(x_train_poisoned, y_train_poisoned)
            # show predict
            predict = classfier.predict(X_test)
            predict = np.argmax(predict, axis=1)
            acc_po = accuracy_score(y_test, predict)
            print(acc_po * 100)
            # show 20 images  with labels
            class_names = [str(target_digit1), str(target_digit2)]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Clean Accuracy", f"{acc_clean:.3%}")
            with c2:
                st.metric("Poisoned Accuracy", f"{acc_po:.3%}")
            with c3:
                st.metric(" Accuracy drop", f"{(acc_po - acc_clean):+.2%}")
            fig, axes = plt.subplots(4, 5, figsize=(12, 8))
            axes = axes.flatten()

            for i in range(20):
                axes[i].imshow(X_test[i].reshape(28, 28), cmap="gray")
                true_class = y_test[i]
                pred_class = predict[i]

                if pred_class == true_class:
                    color = "green"
                    title = f"T:{true_class} P:{pred_class} ✓"
                else:
                    color = "red"
                    title = f"T:{true_class} P:{pred_class} ✗"

                axes[i].set_title(title, color=color, fontsize=9)
                axes[i].axis("off")

            plt.tight_layout()
            st.pyplot(fig)
    elif attack_type == "GradientMatchingAttack":
        with st.spinner("⏳ Running Gradient Matching Attack... Please wait"):
            source_class = parameters.get("source_class", 7)
            target_class = parameters.get("target_class", 2)
            percent_poison = parameters.get("percent_poison", 10) / 100.0
            epsilon = parameters.get("epsilon", 0.3)

            attack = GradientMatchingAttack(
                classifier=classifier,
                percent_poison=percent_poison,
                epsilon=epsilon,
                max_trials=3,
                max_epochs=50,
                learning_rate_schedule=([0.1, 0.01, 0.001], [20, 40, 45]),
                batch_size=64,
                clip_values=(-1.0, 1.0),
                verbose=True
            )


            # Choose trigger samples (from source_class)
            x_trigger = x_test[y_test == source_class][:5].cpu().numpy()
            y_trigger = np.full(len(x_trigger), target_class)

            # Poison the training set
            x_poison, y_poison = attack.poison(
                x_trigger,
                y_trigger,
                x_train.cpu().numpy(),
                y_train.cpu().numpy(),

            )

            # Convert back to torch tensors
            x_train_poisoned = torch.from_numpy(x_poison).to(device)
            y_train_poisoned = torch.from_numpy(y_poison).to(device)

            # Retrain classifier with poisoned dataset
            classifier.fit(
                x_train_poisoned.cpu().numpy(),
                y_train_poisoned.cpu().numpy(),
                batch_size=128,
                nb_epochs=5
            )


            # Test on some target class samples
            idx = np.where(y_test.cpu().numpy() == target_class)[0][:20]
            x_eval = x_test[idx]
            y_eval = y_test[idx]
            predictions = classifier.predict(x_eval.cpu().numpy())
            predicted_classes = predictions.argmax(1)

            # Visualization
            fig, axes = plt.subplots(4, 5, figsize=(15, 8))
            axes = axes.ravel()
            successful = 0
            for i in range(len(x_eval)):
                axes[i].imshow(x_eval[i].cpu().numpy().squeeze(), cmap="gray")
                true_class = y_eval[i].item()
                pred_class = predicted_classes[i]
                if pred_class != true_class:
                    successful += 1
                    color = 'green'
                    title = f"TRUE:{true_class} → PRED:{pred_class} ✓"
                else:
                    color = 'red'
                    title = f"TRUE:{true_class} → PRED:{pred_class} ✗"
                axes[i].set_title(title, color=color, fontsize=9)
                axes[i].axis("off")

            plt.tight_layout()
            st.pyplot(fig)

            asr = successful / len(x_eval)
            st.metric("Attack Success Rate (ASR)", f"{asr:.2%}")
    elif attack_type == "PoisoningAttackCleanLabelBackdoor":
        with st.spinner("⏳ Running " + attack_type + " attack... Please wait"):
            # Create perturbation function
            perturbation_fn = create_perturbation_function(patch_type, parameters)

            # Initialize attack
            target_class = parameters.get("target_class")
            poison_fraction = parameters.get("percent_poison") / 100.0
            backdoor = PoisoningAttackBackdoor(perturbation=perturbation_fn)
            proxy = AdversarialTrainerMadryPGD(classifier)
            proxy.fit(x_train.cpu().numpy(), y_train.cpu().numpy())
            attack= PoisoningAttackCleanLabelBackdoor(backdoor=backdoor,target=target_class,proxy_classifier=classifier, pp_poison=poison_fraction)
            nb_poisoning = int(len(x_train) * poison_fraction)

            non_target_indices = np.where(y_train.cpu().numpy() != target_class)[0]
            if len(non_target_indices) < nb_poisoning:
                st.error(
                    f"Not enough non-target samples. Only {len(non_target_indices)} available, need {nb_poisoning}.")
                st.stop()
            st.info(f"🧪 Creating {nb_poisoning} poisoned samples targeting class {target_class}...")

            idx = np.random.choice(non_target_indices, nb_poisoning, replace=False)

            x_poison = x_train[idx].cpu().numpy()
            y_poison = y_train[idx].cpu().numpy()
            from art.utils import to_categorical
            y_poison_oh = to_categorical(y_poison, nb_classes=num_classes)
            x_poisoned, y_poisoned = attack.poison(x_poison, y_poison_oh)

            # Convert back to torch tensors
            x_poison_torch = torch.tensor(x_poisoned, dtype=torch.float32).to(device)
            y_poison_labels = torch.tensor(y_poisoned.argmax(axis=1), dtype=torch.long).to(device)

            # Combine clean + poisoned data
            x_train_poisoned = torch.cat([x_train, x_poison_torch], dim=0)
            y_train_poisoned = torch.cat([y_train, y_poison_labels], dim=0)

            # Retrain the classifier with poisoned data
            classifier.fit(
                x_train_poisoned.cpu().numpy(),
                y_train_poisoned.cpu().numpy(),
                batch_size=128,
                nb_epochs=10
            )

            # Test accuracy on clean data
            clean_predictions = classifier.predict(x_test.cpu().numpy())
            clean_acc = (clean_predictions.argmax(1) == y_test.cpu().numpy()).mean()

            # Test backdoor success rate
            mask_non_target = (y_test.cpu().numpy() != target_class)
            x_test_subset = x_test[mask_non_target]

            # Apply patch to test images
            x_triggered = patch(patch_type, parameters, x_test_subset)
            triggered_predictions = classifier.predict(x_triggered.cpu().numpy())
            asr = (triggered_predictions.argmax(1) == target_class).mean()

            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Clean Test Accuracy", f"{clean_acc:.3f}")
            with col2:
                st.metric("Attack Success Rate (ASR)", f"{asr:.3f}")

            # Visualization
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))

            for i in range(min(5, len(x_test_subset))):
                # Original image
                orig_img = x_test_subset[i].cpu().numpy().squeeze()
                axes[0, i].imshow(orig_img, cmap='gray')
                axes[0, i].set_title(f"Original\nTrue: {y_test[mask_non_target][i].item()}")
                axes[0, i].axis('off')

                # Triggered image
                trig_img = x_triggered[i].cpu().numpy().squeeze()
                pred_label = triggered_predictions[i].argmax()
                color = 'green' if pred_label == target_class else 'red'
                axes[1, i].imshow(trig_img, cmap='gray')
                axes[1, i].set_title(f"Triggered\nPred: {pred_label}", color=color)
                axes[1, i].axis('off')

            plt.tight_layout()
            st.pyplot(fig)

            if asr > 0.8:
                st.success(f"🎉 Attack successful! ASR: {asr:.1%}")
            elif asr > 0.5:
                st.warning(f"⚠️ Partial success. ASR: {asr:.1%}")
            else:
                st.error(f"❌ Attack failed. ASR: {asr:.1%}")
