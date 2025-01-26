import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

def load_data(dataset_name):
    """Load BACE, BBBP, or HIV dataset."""
    df = pd.read_csv(f"data/{dataset_name}.csv")
    # Example preprocessing: Assume SMILES in 'smiles' column and labels in 'label'
    smiles = df['smiles'].values
    labels = df['label'].values
    return smiles, labels

def preprocess_data(smiles, labels, test_size=0.2, random_state=42):
    """Split data into train/test and undersample to handle imbalance."""
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        smiles, labels, test_size=test_size, random_state=random_state
    )
    # Undersample majority class
    rus = RandomUnderSampler(random_state=random_state)
    X_train_res, y_train_res = rus.fit_resample(X_train.reshape(-1, 1), y_train)
    return X_train_res.ravel(), X_test, y_train_res, y_test
