import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, recall_score

def train_model(model, X_train, y_train, epochs=100, lr=0.01):
    """Train a model and return losses."""
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                           torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
    return losses

def evaluate_model(model, X_test, y_test):
    """Evaluate accuracy and recall."""
    with torch.no_grad():
        outputs = model(torch.tensor(X_test, dtype=torch.float32)).squeeze()
        predictions = (outputs > 0.5).float()
    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    return acc, recall
