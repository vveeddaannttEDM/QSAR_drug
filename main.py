import numpy as np
from src.data_loading import load_data, preprocess_data
from src.embedding import morgan_fingerprint
from src.feature_selection import apply_pca
from src.models import ClassicalClassifier, QuantumClassifier
from src.train_evaluate import train_model, evaluate_model
from src.visualization import plot_accuracy_vs_features
from src.config import *

# Load data
smiles, labels = load_data("BACE")
X_train, X_test, y_train, y_test = preprocess_data(smiles, labels)

# Generate embeddings
X_train_emb = np.array([morgan_fingerprint(s) for s in X_train])
X_test_emb = np.array([morgan_fingerprint(s) for s in X_test])

# Apply PCA for feature selection
n_components = 2**N_QUBITS
X_train_pca = apply_pca(X_train_emb, n_components)
X_test_pca = apply_pca(X_test_emb, n_components)

# Train classical model
model_classical = ClassicalClassifier(n_components)
loss_classical = train_model(model_classical, X_train_pca, y_train, epochs=N_EPOCHS)
acc_classical, _ = evaluate_model(model_classical, X_test_pca, y_test)

# Train quantum model
model_quantum = QuantumClassifier(N_QUBITS)
loss_quantum = train_model(model_quantum, X_train_pca, y_train, epochs=N_EPOCHS)
acc_quantum, _ = evaluate_model(model_quantum, X_test_pca, y_test)

# Plot results
plot_accuracy_vs_features([acc_classical], [acc_quantum], [N_QUBITS])
