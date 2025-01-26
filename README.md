# QSAR_drug
# Quantum Machine Learning for QSAR Prediction 🧪⚛️

*Improving Drug Discovery with Quantum Classifiers on Limited Data*

---

## 📌 Overview

This project explores the use of **quantum machine learning (QML)** to predict Quantitative Structure-Activity Relationships (QSAR) in drug discovery. The goal is to demonstrate how quantum classifiers can outperform classical models when working with small datasets or reduced features—a common challenge in medical research. Inspired by [this paper](QSAR_drug.pdf), the code compares classical neural networks with hybrid quantum-classical models using real-world datasets like BACE, BBBP, and HIV.

---

## 🔑 Key Features

- **Hybrid Quantum-Classical Workflow**: Combines quantum circuits with classical neural networks.
- **Robust to Limited Data**: Quantum models show better generalization with fewer training samples.
- **Multiple Molecular Embeddings**: Supports Morgan fingerprints and ImageMol (placeholder) for feature extraction.
- **PCA for Dimensionality Reduction**: Mimics real-world data incompleteness.
- **Reproducible Results**: Modular code structure for easy experimentation.

---

## 🛠️ Installation

### Dependencies
- Python 3.8+
- [Poetry](https://python-poetry.org/) (recommended) or `pip`

### Setup
1. **Clone the repo**:
   ```bash
   git clone https://github.com/yourusername/QSAR_Quantum_ML.git
   cd QSAR_Quantum_ML
