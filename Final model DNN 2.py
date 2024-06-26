# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:15:39 2024

@author: kpourmah
"""

#Optimized on train data 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'C:/Users/kpourmah/OneDrive - Åbo Akademi O365/Big Data/data spilit/vegetarian_or_vegan_data.csv'
df = pd.read_csv(file_path)

# Select relevant columns
selected_columns = ['month', 'age', 'gender', 'morma_outlet_id', 'discount_class', 'product_name', 'brand_name', 
                    'partner_name', 'avg_kg_price', 'vegan', 'organic', 'consumer_package_size','volume_kg']
df_selected = df[selected_columns].copy()

# Encode categorical variables
label_encoders = {}
for col in ['age', 'gender', 'morma_outlet_id', 'discount_class', 'product_name', 'brand_name', 
                    'partner_name','vegan', 'organic', 'month']:
    label_encoders[col] = LabelEncoder()
    df_selected[col] = label_encoders[col].fit_transform(df_selected[col])



# Split the data into features (X) and target (y)
X = df_selected.drop('avg_kg_price', axis=1)
y = df_selected['avg_kg_price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Set a seed for reproducibility
seed_value = 42
torch.manual_seed(seed_value)
np.random.seed(seed_value)

# Define a new neural network architecture
class ImprovedNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(ImprovedNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)

        # Add batch normalization layers
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(16)

        # Add dropout layers
        self.dropout = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))
        x = F.relu(self.bn4(self.dropout(self.fc4(x))))
        x = self.fc5(x)
        return x

# Instantiate the improved model
input_size = X_train_tensor.shape[1]
improved_model = ImprovedNeuralNetwork(input_size)

# Define a new optimizer with a different learning rate
optimizer = optim.SGD(improved_model.parameters(), lr=0.01, momentum=0.9)

# Define the loss function
criterion = nn.MSELoss()
# Train the improved model
num_epochs = 1300
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    improved_model.train()
    optimizer.zero_grad()
    outputs = improved_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    improved_model.eval()
    with torch.no_grad():
        outputs = improved_model(X_test_tensor)
        loss = criterion(outputs, y_test_tensor)
        test_losses.append(loss.item())

    if epoch % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')

# Plot the MSE convergence curve
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('1000 Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Evaluate the improved model
improved_model.eval()
with torch.no_grad():
    outputs = improved_model(X_test_tensor)
    test_loss = criterion(outputs, y_test_tensor)
    print(f'Improved Test Loss: {test_loss.item():.4f}')

from sklearn.metrics import mean_squared_error, r2_score

# Make predictions on the test set
improved_model.eval()
with torch.no_grad():
    outputs = improved_model(X_test_tensor)
    test_loss = criterion(outputs, y_test_tensor)
    print(f'Improved Test Loss: {test_loss.item():.4f}')
        
# Plot the MSE convergence curve
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Evaluate the improved model
improved_model.eval()
with torch.no_grad():
    outputs = improved_model(X_test_tensor)
    test_loss = criterion(outputs, y_test_tensor)
    print(f'Improved Test Loss: {test_loss.item():.4f}')
# Convert PyTorch tensor to numpy array for evaluation metrics
predictions = outputs.numpy()
# Calculate Absolute Percentage Error (APE)
absolute_percentage_error = np.abs((y_test_tensor.numpy() - predictions) / y_test_tensor.numpy()) * 100
# Calculate Percentage Error (PE)
percentage_error = ((y_test_tensor.numpy() - predictions) / y_test_tensor.numpy()) * 100
# Calculate Mean Percentage Error (MPE)
mpe = np.mean(percentage_error)
print(f'Mean Percentage Error (MPE): {mpe:.2f}%')
# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(absolute_percentage_error)
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test_tensor.numpy(), predictions))
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
# Calculate R-squared
r_squared = r2_score(y_test, predictions)
print(f'R-squared (R²): {r_squared:.4f}')
# Plot regression plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.title('Test Set Regression Plot')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

import os
print(os.getcwd())

# Store the predictions in the DataFrame
df_test_with_predictions = X_test.copy()
df_test_with_predictions['actual_avg_kg_price'] = y_test.values
df_test_with_predictions['avg_kg_price'] = predictions  # Use the correct variable name
# Save the DataFrame with predictions to a CSV file
df_test_with_predictions.to_csv('test_data_with_predictions.csv', index=False)



