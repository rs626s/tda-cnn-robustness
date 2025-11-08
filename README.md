# Topology-Enhanced Image Classification Framework

This project explores how **topological data analysis (TDA)** can enhance the **robustness and interpretability of deep learning models** for image classification. The framework integrates a custom **Persistence Landscape Layer (PLLay)** with standard CNN backbones such as **CNN**, **ResNet18**, and **VGG11**, and compares them against classical ML baselines (SVM, Random Forest) on datasets like **MNIST**, **Fashion-MNIST**, and **CIFAR-10**.

---

## 🎯 Project Goals

- Introduce a **topology-inspired persistence layer (PLLay)** that extracts global structural features from images.  
- Compare **baseline models** (CNN, ResNet18, VGG11) with their **topology-augmented counterparts** (`*_topology`).  
- Evaluate and visualize:
  - Accuracy on clean and noisy data  
  - Model robustness under Gaussian perturbations  
  - Training and inference efficiency  
  - Generalization gap (train vs. test)  
  - Feature-space differences introduced by topological preprocessing  

The design is influenced by  
*PLLay: Efficient Topological Layer based on Persistent Landscapes* (arXiv:2002.02778)

---

## 🧠 What Is Implemented?

This framework implements both **standard** and **topology-enhanced** model variants for comparative analysis:

1. **Baseline Models:** `cnn`, `resnet18`, `vgg11`  
2. **Topology-Augmented Models:** `cnn_topology`, `resnet18_topology`, `vgg11_topology`

Each topology model introduces a **Persistence Landscape Layer (PLLay)** that computes topological descriptors from input images, transforms them into compact persistence landscapes, and concatenates these as an **extra channel** to the network input before convolution.

In addition, the project includes:
- **Classical ML baselines** (SVM, Random Forest) with **HOG** and **PCA** features  
- **Noise robustness analysis** evaluating performance under varying Gaussian noise levels  
- **Training and evaluation pipelines** for reproducible experiments  

---

## ⚙️ Requirements and Setup

These libraries are required as prerequisites to execute this code: 
`pip install torch torchvision torchaudio numpy pandas matplotlib seaborn scikit-learn tqdm gudhi scipy Pillow`

**Core dependencies:**  
- torch, torchvision — deep learning framework  
- numpy, pandas, scikit-learn — data processing and classical ML  
- matplotlib, seaborn — visualization  
- tqdm — progress tracking  
- gudhi — topological persistence (optional but recommended)  
- scipy, Pillow — image and math utilities  

---

## 🚀 How to Run

**Train a baseline model:**  
`python train.py --model cnn --dataset mnist --epochs 10`

**Train a topology-enhanced model:**  
`python train.py --model cnn_topology --dataset fashion --useGudhi --epochs 5`

**Evaluate a trained model:**  
`python evaluate.py --model resnet18 --dataset cifar10 --checkpoint outputs/resnet18_cifar10/model_best.pt`

**Run classical ML baselines:**  
`python classical_ml.py --dataset mnist --feature pca --clf svm`

**Test robustness to Gaussian noise:**  
`python evaluate.py --model cnn --dataset mnist --checkpoint outputs/cnn_mnist/model_best.pt --testRobustness --robustnessLevels 0.0 0.1 0.2 0.3`

---

## 📈 Evaluation Metrics and Outputs

After training or evaluation, the following files are automatically generated:

- **metrics.json** — accuracy, loss, parameters, and runtime statistics  
- **confusion_matrix.png** — confusion matrix visualization  
- **learning_curves.png** — training and validation accuracy trends  
- **robustness_curves.png** — accuracy vs. Gaussian noise level  
- **topology_analysis.png** — visualization of persistence-based feature space  

---

## 🧮 Example Results

| Model Type | Dataset | Accuracy | Notes |
|-------------|----------|-----------|--------|
| CNN | MNIST | ~99% | Standard CNN baseline |
| CNN + PLLay | Fashion-MNIST | ~91% | Improved noise robustness |
| ResNet18 | CIFAR-10 | ~93% | Deep model baseline |
| SVM (PCA) | MNIST | ~97% | Classical ML baseline |

---

## 🧰 Key Insights

- Topological preprocessing improves **robustness** and **feature stability** with minimal accuracy loss.  
- Persistence landscapes encode **global shape features**, complementing local pixel-based CNN filters.  
- The framework supports **reproducible experiments** and **systematic robustness evaluation**.

---

*This repository unifies classical machine learning and topology-enhanced deep learning to advance research in robust image classification.*
