import matplotlib.pyplot as plt

def plot_accuracy_vs_features(acc_classical, acc_quantum, n_features_list):
    plt.plot(n_features_list, acc_classical, label="Classical")
    plt.plot(n_features_list, acc_quantum, label="Quantum")
    plt.xlabel("Number of Features (log2)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("results/accuracy_vs_features.png")
