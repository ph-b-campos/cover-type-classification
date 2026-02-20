# 🌲 Cover Type Classification with Deep Learning

**Executive Summary:**
In this Deep Learning project, I built a Multilayer Perceptron (MLP) using PyTorch to predict forest cover types based strictly on cartographic variables. By implementing a custom Neural Network architecture, efficient DataLoaders, and utilizing Optuna for hyperparameter tuning, the model successfully achieved **over 93% accuracy** on the test set, demonstrating strong generalization on a large-scale dataset.

---

## 1. The Technical Challenge

Land management tasks often require identifying tree species and forest cover types across vast geographic areas. While satellite imagery is common, this project relies solely on tabular cartographic data (elevation, slope, soil type, etc.) to make predictions.

**Objective:** Build a Deep Learning model to accurately classify 30x30 meter cells into one of 7 distinct forest cover types.
* **Problem Type:** Multiclass Classification (7 classes)
* **Dataset:** CoverType dataset (approx. 581,000 instances, 54 attributes)
* **Target Metric:** Accuracy (> 93%)

## 2. Data Pipeline & Neural Architecture

Handling a dataset of nearly 600,000 instances requires an efficient data pipeline to feed the GPU/CPU without bottlenecking the training process.

* **Data Preprocessing:** Built a Scikit-Learn `ColumnTransformer` pipeline to apply `StandardScaler` to continuous features (elevation, slope, etc.) while passing through the pre-encoded binary data (wilderness areas and soil types).
* **PyTorch Integration:** Created a custom `torch.utils.data.Dataset` class and utilized `DataLoader` to manage batching (batch size = 1024) and memory efficiency.
* **Network Topology:** Designed a custom `nn.Module` (Multilayer Perceptron). The network utilizes sequential `Linear` layers, `ReLU` activation functions, and `BatchNorm1d` to stabilize and accelerate training.

## 3. Training & Hyperparameter Optimization (Optuna)

The training process was structured to ensure the model converged effectively without overfitting:

1. **Batch Overfitting:** Initially verified the network's capacity to learn by successfully overfitting a single batch of data.
2. **Hyperparameter Tuning:** Integrated **Optuna** to systematically search for the optimal architecture and learning rate. The search space included the number of hidden neurons and the learning rate, guided by a Median Pruner to halt unpromising trials early.
3. **Early Stopping:** Implemented a custom training loop with an early stopping mechanism based on validation accuracy to preserve the best model weights.

🛠️ **Tech Stack:**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-252A42?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

> 💡 **The full training loop, learning curve visualizations, and final test metrics are documented directly inside the Jupyter Notebook.**
