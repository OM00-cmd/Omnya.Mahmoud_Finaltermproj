import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim

# Load dataset
df = pd.read_csv('diabetes.csv')
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])
        return self.sigmoid(out)

# Initialize model, loss function, and optimizer
input_size = X_train.shape[1]
hidden_size = 50
output_size = 1
lstm_model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

# Train the LSTM model
def train_lstm_model(model, X_train, y_train, epochs=10, batch_size=32):
    model.train()
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size].unsqueeze(1)
            y_batch = y_train[i:i + batch_size]
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

train_lstm_model(lstm_model, X_train_tensor, y_train_tensor)

# Predict using the LSTM model
lstm_model.eval()
with torch.no_grad():
    X_test_tensor = X_test_tensor.unsqueeze(1)
    y_pred_lstm_tensor = lstm_model(X_test_tensor)
    y_pred_lstm = (y_pred_lstm_tensor > 0.5).float().numpy().flatten()

# SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Calculate performance metrics
def calculate_performance_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"False Positive Rate (FPR): {fpr}")
    print(f"False Negative Rate (FNR): {fnr}")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

print("Performance Metrics for Random Forest:")
calculate_performance_metrics(y_test, y_pred_rf)

print("\nPerformance Metrics for LSTM:")
calculate_performance_metrics(y_test, y_pred_lstm)

print("\nPerformance Metrics for SVM:")
calculate_performance_metrics(y_test, y_pred_svm)
