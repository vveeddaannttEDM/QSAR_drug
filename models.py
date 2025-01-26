import torch
import torch.nn as nn
import pennylane as qml

# Classical MLP
class ClassicalClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

# Quantum Circuit (simulated with PennyLane)
n_qubits = 3  # Example: 3 qubits for 8 features (2^3)
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # Amplitude encoding
    qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True)
    # Entangling layers
    for i in range(2):
        for j in range(n_qubits):
            qml.Rot(*weights[i][j], wires=j)
        qml.broadcast(qml.CNOT, wires=range(n_qubits), pattern="all")
    # Measure all qubits
    return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

# Hybrid Quantum-Classical Model
class QuantumClassifier(nn.Module):
    def __init__(self, n_qubits, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        weight_shape = (n_layers, n_qubits, 3)
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes={"weights": weight_shape})
        self.post_process = nn.Linear(n_qubits, 1)
    
    def forward(self, x):
        x = self.qlayer(x)
        return torch.sigmoid(self.post_process(x))
