import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import shuffle
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
import pdb

class DNN(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rate, output_dim):
        super(DNN, self).__init__()
        layers = []
        in_features = input_dim

        for units in hidden_units:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = units

        layers.append(nn.Linear(in_features, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train(best_params, X_train, X_test, y_train, y_test):
    final_model = DNN(
        input_dim=X_train.shape[1],
        hidden_units=list(map(int, best_params["architecture"].split("_"))),
        dropout_rate=best_params["dropout_rate"],
        output_dim=len(np.unique(y_train))
        
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(final_model.parameters(), lr=best_params["lr"])
    epochs = 50

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=int(best_params["batch_size"]), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=int(best_params["batch_size"]), shuffle=False)

    final_model.train()
    for epoch in range(epochs): 
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = final_model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

    return final_model, test_loader

def evaluate(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            outputs = model(xb)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(yb.numpy())
    
    return all_labels, all_preds

def objective(trial, X_train, X_val, y_train, y_val, le, args=None):
    architecture = trial.suggest_categorical(
        "architecture",
        [
            "64_32",
            "64_32_16",
            "128_64",
            "128_64_32",
            "256_128",
            "256_128_64",
            "256_128_64_32"
        ]
    )
    hidden_units = list(map(int, architecture.split("_")))
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    lr = trial.suggest_categorical("lr", [1e-4, 5e-4, 1e-3])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    epochs = 30

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = DNN(input_dim=X_train.shape[1],
                hidden_units=hidden_units,
                dropout_rate=dropout_rate,
                output_dim=len(np.unique(y_train)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    all_labels, all_preds, model = evaluate(model, val_loader)
    report_dict = classification_report(all_labels, all_preds, target_names=le.classes_, output_dict=True, zero_division=0)
    
    row = {
        "trial": trial.number,
        "architecture": architecture,
        "dropout_rate": dropout_rate,
        "lr": lr,
        "batch_size": batch_size,
        "accuracy": report_dict["accuracy"],
        "precision": report_dict["macro avg"]["precision"],
        "recall": report_dict["macro avg"]["recall"],
        "f1": report_dict["macro avg"]["f1-score"]
    }
    
    os.makedirs(f"results/tmp", exist_ok=True)
    result_file = f"results/tmp/trial_results.csv"

    df_row = pd.DataFrame([row])
    if not os.path.exists(result_file):
        df_row.to_csv(result_file, index=False)
    else:
        df_row.to_csv(result_file, mode="a", header=False, index=False)


    f1 = report_dict["macro avg"]["f1-score"]
    return f1