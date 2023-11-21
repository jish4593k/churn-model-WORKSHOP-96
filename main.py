import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# Check if GPU is available and use it for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
dataset = pd.read_csv('/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Python/Churn_Modelling.csv')

X = dataset.iloc[:, 3:-1].values
Y = dataset.iloc[:, -1].values

# Label encoding for Gender column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# Encoding for country
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)

# Build the model
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.layer1 = nn.Linear(12, 6)
        self.layer2 = nn.Linear(6, 6)
        self.output_layer = nn.Linear(6, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.sigmoid(self.output_layer(x))
        return x

# Initialize the model
ann = ANN().to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(ann.parameters(), lr=0.001)

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    ann.train()
    optimizer.zero_grad()
    outputs = ann(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# Make predictions on the test set
ann.eval()
with torch.no_grad():
    y_pred_tensor = ann(X_test_tensor)
    y_pred = (y_pred_tensor.cpu().numpy() > 0.5).astype(int)

# Convert tensors to numpy arrays for evaluation
y_test_np = np.array(y_test)
y_pred_np = np.squeeze(y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test_np, y_pred_np)
print(cm)

# Accuracy Score
accuracy = accuracy_score(y_test_np, y_pred_np)
print("Accuracy:", accuracy)
