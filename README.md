# 🧠 HYPATIA - Adversarial Machine Learning Testing Platform
<div align="center">

![Python](https://img.shields.io/badge/python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-orange)
[![ART](https://img.shields.io/badge/ART-Adversarial%20Robustness%20Toolbox-blue)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

[![Live Demo](https://img.shields.io/badge/Live-Demo-blue?style=for-the-badge&logo=streamlit)](https://hypatia-ml-app.streamlit.app/)

</div>

---


## 🌟 About HYPATIA

> *"Reserve your right to think, for even to think wrongly is better than not to think at all."*  
> **— Hypatia of Alexandria**

Named after **Hypatia of Alexandria**, the renowned ancient mathematician, astronomer, and philosopher, this platform embodies her spirit of rigorous inquiry and fearless pursuit of knowledge. Just as Hypatia challenged conventional thinking in her time, **HYPATIA** challenges your machine learning models to reveal their vulnerabilities and strengthen their defenses.

**HYPATIA** is a comprehensive adversarial machine learning testing platform that provides researchers and practitioners with powerful tools to evaluate ML model robustness through systematic adversarial testing. Built with Streamlit, it offers an intuitive interface for conducting sophisticated security assessments across multiple threat models.

### 🎯 **Platform Specifications**
- **Model:** Convolutional Neural Network (CNN)
- **Dataset:** MNIST handwritten digits
- **Coverage:** 4 major attack categories with 20+ attack variants
- **Interface:** Web-based Streamlit application

---

## 🌐 Try HYPATIA Online

**Live Demo:** [https://hypatia-ml-app.streamlit.app/](https://hypatia-ml-app.streamlit.app/)

Experience the full power of adversarial testing without any installation required.

---

## 🚀 What HYPATIA Can Do

<div align="center">

![Adversarial Threats](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/_images/adversarial_threats_attacker.png)

</div>

HYPATIA provides a comprehensive suite of adversarial testing tools across four critical security dimensions:

| Attack Category | Purpose | Impact |
|----------------|---------|--------|
| 🎯 **Evasion** | Test inference-time robustness | Model misclassification |
| ☠️ **Poisoning** | Evaluate training data security | Compromised model behavior |
| 🕵️ **Inference** | Assess privacy vulnerabilities | Information leakage |
| 🔓 **Extraction** | Test model stealing resilience | Intellectual property theft |

---

**Hypatia** provides a comprehensive suite of adversarial testing tools to help you:

## **🎯Evasion Attacks**
Test model robustness against adversarial examples designed to fool classifiers at inference time.

#### ⚪ **White-box Attacks**
- **Full model access** (architecture, parameters, gradients)
- **High success rate**
- **Computationally efficient**
- **Realistic for insider threats**

#### ⚫ **Black-box Attacks**
- **Query-only access** (no internal details)
- **More realistic threat model**
- **Requires more queries**
- **Harder to detect and defend**



### 🎯 **Targeting Strategies**

| Strategy | Goal | Example |
|----------|------|---------|
| **Untargeted** | Force any misclassification | Turn "3" into anything except "3" |
| **Targeted** | Force specific misclassification | Turn "3" into exactly "7" |

### 📊 **Available Attacks**

#### **White-box Methods**
| Attack | Type | Targeted | Untargeted | Description |
|--------|------|:--------:|:----------:|-------------|
| **FGSM** | Single-step | ✅ | ✅ | Fast gradient-based perturbations |
| **PGD** | Iterative | ✅ | ✅ | Projected gradient descent optimization |
| **BIM** | Multi-step | ✅ | ✅ | Basic iterative FGSM variant |
| **DeepFool** | Minimal | ❌ | ✅ | Smallest perturbation to decision boundary |
| **C&W** | Optimization | ✅ | ✅ | Advanced L0, L2, L∞ norm attacks |
| **ElasticNet** | Sparse | ✅ | ✅ | Sparse adversarial perturbations |
| **NewtonFool** | Newton-based | ❌ | ✅ | Second-order optimization method |
| **JSMA** | Saliency | ✅ | ❌ | Jacobian-based pixel manipulation |

#### **Black-box Methods**
| Attack | Type | Targeted | Untargeted | Description |
|--------|------|:--------:|:----------:|-------------|
| **Boundary Attack** | Decision boundary | ✅ | ✅ | Explores classification boundaries |
| **HopSkipJump** | Query-efficient | ✅ | ✅ | Advanced boundary exploration |

---

## ☠️ Poisoning Attacks

**Poisoning attacks** compromise models during training by injecting malicious samples into the training dataset, causing models to learn incorrect behaviors or hidden backdoors.

### 🎯 **Attack Types**

| Attack | Strategy | Target | Description |
|--------|----------|---------|-------------|
| **Backdoor Attack** | Targeted | Specific trigger | Embeds hidden triggers that cause misclassification |
| **Clean-label Backdoor** | Targeted | Subtle poisoning | Uses correctly labeled samples for stealthy backdoors |
| **Gradient Matching** | Targeted | Gradient alignment | Matches poison gradients with target sample gradients |
| **Feature Collision** | Targeted | Feature space | Forces feature space collisions for misclassification |
| **SVM Poisoning** | Untargeted | Decision boundary | Shifts SVM boundaries to reduce overall accuracy |

### 🔧 **Backdoor Trigger Options**

HYPATIA supports multiple patch styles for backdoor attacks:

| Patch Type | Description | Visibility |
|------------|-------------|------------|
| **Square** | Solid square patches | High visibility, reliable trigger |
| **Pattern** | Checkerboard designs | Medium visibility, pattern-based |
| **Pixel** | Random pixel alterations | Low visibility, subtle trigger |

---

## 🕵️ Inference Attacks

**Inference attacks** exploit trained models to extract sensitive information about training data, threatening privacy without modifying the model or data.

### 🔍 **Attack Categories**

#### **Membership Inference**
Determines if specific samples were used in model training.

**Configuration Options:**
- **Model Types:** Normal vs Shadow models
- **Attack Methods:** Black-box, Decision Boundary
- **Classifiers:** Neural Network, SVM, Random Forest, Gradient Boosting, Decision Tree, KNN
- **Thresholds:** Supervised/Unsupervised decision boundaries

#### **Attribute Inference**  
Infers hidden or sensitive attributes from partial information.

**Configuration Options:**
- **Target Features:** Any pixel position (0-783)
- **Feature Types:** Continuous/Discrete values
- **Training Parameters:** Epochs, subset sizes
- **Attack Models:** Configurable neural networks

#### **Model Inversion**
Reconstructs training samples from model outputs.

**Configuration Options:**
- **Initialization:** None, Black, White, Gray, Random images
- **Targets:** Specific digits (0-9)
- **Convergence:** Max iterations and thresholds

### 📈 **Privacy Risk Assessment**

| Accuracy/MAE | Risk Level | Action Required |
|--------------|------------|-----------------|
| ≥ 0.75 / < 0.05 | 🔴 **Critical** | Urgent mitigation required |
| 0.65-0.75 / 0.05-0.10 | 🟠 **High** | Apply privacy defenses |
| 0.58-0.65 / 0.10-0.20 | 🟡 **Moderate** | Consider noise/dropout |
| 0.53-0.58 / 0.20-0.30 | 🔵 **Weak** | Monitor and assess |
| < 0.53 / ≥ 0.30 | 🟢 **Safe** | Strong privacy protection |

---

## 🔓 Model Extraction Attacks

**Extraction attacks** create functional copies of target models by querying them strategically, enabling intellectual property theft and creating substitute models for further attacks.

### 🎯 **Extraction Methods**

#### **CopyCatCNN**
🔍 **Query-based replication** using synthetic data to train substitute models.

**Features:**
- Multiple query datasets (MNIST, CIFAR-10, Fashion-MNIST)
- Probability-based outputs for higher fidelity
- Configurable batch sizes and training epochs

#### **Functionally Equivalent Extraction**
⚡ **Behavior-focused copying** without replicating internal structure.

**Features:**
- Dense neural network architecture
- Iterative extraction with delta parameters
- Adaptive relative difference slopes

#### **Knockoff Nets**
🎯 **Advanced adversarial extraction** with minimal queries.

**Features:**
- Transfer learning integration
- Adaptive sampling strategies
- Multiple reward functions (certainty, diversity, loss-based)

### 📊 **Evaluation Metrics**

| Metric | Description | Significance |
|--------|-------------|--------------|
| **Original Accuracy** | Victim model performance | Baseline comparison |
| **Stolen Accuracy** | Substitute model performance | Extraction quality |
| **Fidelity** | Agreement between models | Functional similarity |

---

## 🚀 Quick Start
### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM recommended
- CUDA support (optional, for GPU acceleration)
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
---

## 📖 Using HYPATIA

### 🎮 **Basic Workflow**

1. **Select Attack Category** from the sidebar menu
2. **Configure Parameters** using the intuitive controls
3. **Run Attack** and observe real-time progress
4. **Analyze Results** with detailed metrics and visualizations
5. **Export Data** for further analysis (optional)

### 🎯 **Best Practices**

- **Start with simple attacks** (FGSM) before advanced methods
- **Compare multiple attack types** for comprehensive assessment
- **Document parameters** used for reproducible results
- **Monitor resource usage** for large-scale experiments

---
## 🤝 Contributing

We welcome contributions to expand HYPATIA's capabilities! You can contribute by:

1. **Implementing New Attack Methods** – Add additional adversarial techniques.
2. **Extending Dataset Support** – Integrate datasets beyond MNIST.
3. **Adding Defense Mechanisms** – Enhance model robustness features.
4. **Optimizing Performance** – Speed up attack execution and reduce memory usage.

To contribute, please fork the repository, make your changes, and submit a pull request. You can also open an issue for suggestions or bugs.


### 📝 **How to Contribute**

1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** your changes
4. **Test** thoroughly
5. **Submit** a pull request



---
## 🔗 References

[Adversarial Robustness Toolbox (ART)](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/)

[Adversarial Robustness Toolbox (ART) GitHub](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

[Supporting materials](Supporting_material.bib)
   


   ---

## 🙏 Acknowledgments

- **ART Team**: Adversarial Robustness Toolbox for attack implementations
- **Versifai CEO**: Guidance and support during this project 

## 📞 Support

For questions or suggestions, you can open an issue on GitHub or reach out directly.


---

<div align="center">

**Built with ❤️ for the adversarial machine learning community**

*Empowering researchers to build more robust and secure AI systems*

</div>





