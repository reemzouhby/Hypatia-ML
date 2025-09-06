# 🧠 HYPATIA - Adversarial Machine Learning Testing Platform
![Python](https://img.shields.io/badge/python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-orange)
[![ART](https://img.shields.io/badge/ART-Adversarial%20Robustness%20Toolbox-blue)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
[![Live Demo](https://img.shields.io/badge/Live-Demo-blue?style=for-the-badge&logo=streamlit)](https://hypatia-ml-app.streamlit.app/)


>*"Reserve your right to think, for even to think wrongly is better than not to think at all."*
>**— Hypatia of Alexandria**

Named after **Hypatia of Alexandria**, the renowned ancient mathematician, astronomer, and philosopher, this platform embodies her spirit of rigorous inquiry and fearless pursuit of knowledge. Just as Hypatia challenged conventional thinking in her time, **HYPATIA** challenges your machine learning models to reveal their vulnerabilities and strengthen their defenses against adversarial threats.
## 🌟 Overview

HYPATIA is a comprehensive adversarial machine learning testing platform built with Streamlit that provides researchers and practitioners with tools to evaluate ML model robustness through systematic adversarial testing. The platform supports four major categories of attacks across multiple threat models.
**Model & Dataset Used in the App:**

* Model: CNN (Neural Network)

* Dataset: MNIST

**All attacks in the platform are applied to this model and dataset unless specified otherwise (e.g., SVM-specific attacks).**
## 🌐 Live Demo

Try HYPATIA online via Streamlit: [https://hypatia-ml-app.streamlit.app/](https://hypatia-ml-app.streamlit.app/)

## ✨ Features
![This is an alt text.](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/_images/adversarial_threats_attacker.png)
**Hypatia** provides a comprehensive suite of adversarial testing tools to help you:

## **🎯Evasion Attacks**
Test model robustness against adversarial examples designed to fool classifiers at inference time.

**⚪White-box Attacks:**

Attacker has **full access** to the model (architecture, parameters, gradients).

**⚫ Black-box Attacks:**

Attacker can **only query** the model (no internal details).

**White-box Attacks:**
- **FGSM** (Fast Gradient Sign Method) - Single-step gradient-based attack
- **PGD** (Projected Gradient Descent) - Iterative adversarial attack
- **BIM** (Basic Iterative Method) - Multi-step variant of FGSM  
- **DeepFool** - Minimal perturbation attack
- **C&W** (Carlini & Wagner) - Optimization-based attack with L0, L2, L∞ variants
- **ElasticNet** - Sparse adversarial attack
- **NewtonFool** - Newton-based attack method
- **JSMA** (Jacobian-based Saliency Map Attack) - Targeted pixel manipulation

**Black-box Attacks:**
- **Boundary Attack** - Decision boundary exploration
- **HopSkipJump** - Query-efficient boundary attack

**📌 Targeted vs Untargeted**

**Untargeted:** The goal is just to make the model misclassify into any wrong class.
Example: Turning a “3” into anything except “3”.

**Targeted:** The adversary tries to force the model to misclassify into a specific target class.
Example: Turning a “3” into exactly a “7”.
### 🎯 Attack Categories

The following table summarizes which attacks are **white-box/black-box** and whether they support **targeted** or **untargeted** modes:

| Attack              | White-box | Black-box | Untargeted | Targeted |
|---------------------|:---------:|:---------:|:----------:|:--------:|
| **FGSM**            | ✅        | ❌        | ✅         | ✅       |
| **PGD**             | ✅        | ❌        | ✅         | ✅       |
| **BIM**             | ✅        | ❌        | ✅         | ✅       |
| **DeepFool**        | ✅        | ❌        | ✅         | ❌       |
| **NewtonFool**      | ✅        | ❌        | ✅         | ❌       |
| **C&W (L0, L2, L∞)**| ✅        | ❌        | ✅         | ✅       |
| **ElasticNet**      | ✅        | ❌        | ✅         | ✅       |
| **JSMA**            | ✅        | ❌        | ❌         | ✅       |
| **Boundary Attack** | ❌        | ✅        | ✅         | ✅       |
| **HopSkipJump**     | ❌        | ✅        | ✅         | ✅       |

✅ = Supported  
❌ = Not supported


## ☠️ **Poisoning Attacks**
**A poisoning attack** targets **the training phase** of a machine learning model. Instead of manipulating inputs at inference time (like evasion), the attacker **injects malicious samples** into the training data. These poisoned samples cause the model to learn in a way that reduces accuracy or biases predictions in favor of the attacker’s goal.
| Attack              |  Targeted / Untargeted| Description  |
|---------------------|:---------:|:----------:|
| **Backdoor Attack**            |   Targeted   | Adds **a specific patch or trigger** to some training samples so that the model will misclassify any test sample containing the same trigger into a chosen target class.      
| **Clean-label Backdoor**       | Targeted       | Similar to backdoor, but uses **correctly labeled samples**. Poisoning is subtle and harder to detect, making the model learn the backdoor without obvious mislabeled examples.   
| **Gradient Matching Attack**   | Targeted      | Chooses poison samples so that their **gradients match those of a target sample**, forcing the model to misclassify the target into a desired class.       
| **Feature Collision Attack**  | Targeted | Generates poisoned samples that **collide in feature space with a target sample**, causing the model to misclassify the target sample into a specific base class.   
| **Poisoning SVM Attack**      | Untargeted  | Designed for **two selected MNIST digits.** Modifies training points **to shift the SVM decision boundary**, reducing overall accuracy without targeting a specific misclassification.
#### Patch Types for Backdoor Attacks

**For Backdoor and Clean-Label Backdoor attacks**, poisoned samples include a **patch or trigger**. HYPATIA supports three patch styles:

 * **Square patch:** Solid square applied to a corner or center.

* **Pattern patch:** Checkerboard-like repeated pattern.

* **Pixel patch:** Random individual pixels altered.

These patches are what the model learns to associate with the target class during training, enabling the backdoor attack.

## 🕵️ Inference Attacks

**Inference attacks** target the **trained model** to extract sensitive information about the training data rather than modifying it. HYPATIA implements three main inference attacks on the CNN trained on MNIST:

| Attack Type             | Description | Parameters / Options |
|--------------------------|------------|--------------------|
| **Membership Inference** | Determines if a specific data sample was part of the training dataset. | - **Normal Model** vs **Shadow Model**<br>- Black-box or Decision Boundary methods<br>- Attack classifier: NN, SVM, RF, GB, DT, KNN<br>- Decision thresholds: Supervised / Unsupervised |
| **Attribute Inference**  | Infers hidden or sensitive attributes of a sample, e.g., missing pixel values. | - Target feature (pixel index 0–783)<br>- Continuous / discrete feature<br>- Attack model epochs<br>- Training subset size |
| **Model Inversion**      | Attempts to reconstruct training samples from model outputs. | - Initialization image: None, Black, White, Gray, Random<br>- Target digits (0–9)<br>- Max iterations & threshold for convergence |

**Evaluation Metrics:**  

- **Membership Inference:** Accuracy for members, non-members, and overall attack success.  
- **Attribute Inference:** Mean Absolute Error (MAE) and Mean Squared Error (MSE) for predicted attributes.  
- **Model Inversion:** Visual reconstruction of target samples with predicted labels.  

**Severity Levels (Membership / Attribute Inference):**  

| Overall Accuracy / MAE | Severity | Notes |
|------------------------|---------|------|
| ≥ 0.75 / < 0.05        | 🔴 Critical | Strong privacy leakage; urgent mitigation required |
| 0.65–0.75 / 0.05–0.10 | 🟠 High | Model vulnerable; apply privacy defenses |
| 0.58–0.65 / 0.10–0.20 | 🟡 Moderate | Some vulnerability; consider adding noise or dropout |
| 0.53–0.58 / 0.20–0.30 | 🔵 Weak | Minimal threat; model fairly robust |
| < 0.53 / ≥ 0.30        | 🟢 Failed | Strong privacy protection; negligible leakage |

**Notes:**  
- These attacks demonstrate how model outputs can reveal information about training data even without modifying it.  
- The app allows configuring each attack with flexible parameters to explore different privacy threats.
## 🔓 Extraction (Model Stealing) Attacks

**Extraction attacks** aim to replicate a target model’s behavior by training a substitute (stolen) model that mimics the victim’s predictions. Unlike evasion or poisoning, extraction attacks **do not modify the training or test data of the victim model**; they leverage queries and outputs to reconstruct functionality.

HYPATIA supports three main extraction attacks:

| Attack Type                           | Description |
|--------------------------------------|------------|
| **CopyCatCNN**                        | 🔍 Creates a substitute model by querying the target model with synthetic or external data and training a neural network to replicate its predictions and decision boundaries. Supports probability-based output for better fidelity. |
| **Functionally Equivalent Extraction** | ⚡ Extracts model functionality without replicating internal structure. Focuses on achieving similar input-output behavior using a dense neural network, even if architecture differs. |
| **Knockoff Nets**                      | 🎯 Advanced model stealing using adversarial perturbations and transfer learning to create functional copies with minimal queries to the target model. Supports adaptive sampling and reward strategies to maximize stolen model performance. |

### ⚙️ Parameters and Options

**CopyCatCNN**
- Dataset for querying: MNIST Test Set, CIFAR-10, Fashion-MNIST
- Batch size for training and query
- Number of epochs
- Number of samples to steal
- Probability output toggle

**Functionally Equivalent Extraction**
- Number of neurons in dense substitute model
- Delta values for iterative extraction
- Fraction of true labels used
- Relative difference slope and value
- Maximum delta value

**Knockoff Nets**
- Dataset for querying: MNIST Test Set, CIFAR-10, Fashion-MNIST
- Batch size for training and query
- Number of epochs
- Number of samples to steal
- Probability output toggle
- Sampling strategy: random / adaptive
- Reward strategy: certainty (cert), diversity (div), loss-based (loss), combination (all)

### 📊 Evaluation Metrics

- **Original Accuracy:** Accuracy of the victim model on test set  
- **Stolen Accuracy:** Accuracy of the stolen model on test set  
- **Fidelity:** Percentage of test predictions where stolen model matches the original model  

**Note:** The app allows configuring each attack with flexible parameters and datasets to explore different model stealing scenarios. Visual metrics and success rates are displayed after each attack to measure effectiveness.
## 🚀 Quick Start
### Prerequisites

- Python 3.8 or higher
### Installation

1. **Clone the repository**
```bash
git clone https://github.com/reemzouhby/Hypatia-ML.git
cd Hypatia-ML
```
2. **Create a virtual environment**
```bash
# Create a new environment named "hypatia_env" with Python 3.10 (or your preferred version)
conda create -n hypatia_env python=3.10

# Activate the environment
conda activate hypatia_env
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Dependencies

The repository includes a  `requirements.txt` file containing all necessary packages:
**Included Packages:**
```txt
streamlit>=1.28.0
tensorflow>=2.10.0
torch>=1.12.0
torchvision>=0.13.0
adversarial-robustness-toolbox>=1.15.0
numpy>=1.21.0
matplotlib>=3.5.0
pandas>=1.3.0
scikit-learn>=1.0.0
pillow>=8.3.0
opencv-python>=4.5.0
seaborn>=0.11.0
```
>**Note**: Use pip inside your Conda environment. Do not install globally.
### Running the Application

```bash
streamlit run Hypatia.py
```
This will launch the Streamlit interface in your browser where you can interact with the platform.
You should see output like:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.xxx:8501
```
## 📖 Usage

1. Select the type of attack from the sidebar.

2. Configure attack parameters.

3. Click Run Attack to see results like accuracy drop and fidelity metrics.

## 🤝 Contributing

We welcome contributions to expand HYPATIA's capabilities! You can contribute by:

1. **Implementing New Attack Methods** – Add additional adversarial techniques.
2. **Extending Dataset Support** – Integrate datasets beyond MNIST.
3. **Adding Defense Mechanisms** – Enhance model robustness features.
4. **Optimizing Performance** – Speed up attack execution and reduce memory usage.

To contribute, please fork the repository, make your changes, and submit a pull request. You can also open an issue for suggestions or bugs.
## 🔗 References

[Adversarial Robustness Toolbox (ART)](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/)

[Adversarial Robustness Toolbox (ART) GitHub](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

[Supporting materials](Supporting_material.bib)

## 🙏 Acknowledgments

- **ART Team**: Adversarial Robustness Toolbox for attack implementations
- **Versifai CEO**: Guidance and support during this project 

## 📞 Support

For questions or suggestions, you can open an issue on GitHub or reach out directly.









