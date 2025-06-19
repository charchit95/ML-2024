import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
train_file = "./data/ML-CUP24-TR.csv"
test_file = "./data/ML-CUP24-TS.csv"

df_train = pd.read_csv(train_file, comment='#', header=None)
df_test = pd.read_csv(test_file, comment='#', header=None)

# Extract Features (X) and Targets (y)
X = df_train.iloc[:, 1:-3].values  # Features (Columns 1 to -3)
y = df_train.iloc[:, -3:].values   # Targets (Last 3 columns: x, y, z)

# Normalize Features
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X = (X - X_mean) / X_std

y_mean, y_std = y.mean(axis=0), y.std(axis=0)
y = (y - y_mean) / y_std  # Normalize targets

# Split Data (Train/Validation)
np.random.seed(42)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
split = int(0.8 * X.shape[0])
X_train, X_val = X[indices[:split]], X[indices[split:]]
y_train, y_val = y[indices[:split]], y[indices[split:]]

# Neural Network Architecture
input_size = X.shape[1]  # Number of input features
hidden_size = 64         # Number of hidden neurons
output_size = y.shape[1]  # Number of target variables (3)

# Initialize Weights and Biases
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# Activation Functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Training Parameters
learning_rate = 0.01
epochs = 500
batch_size = 16

# Training Loop
train_losses = []
val_losses = []

for epoch in range(epochs):
    epoch_train_loss = 0
    num_batches = 0

    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        # Forward Propagation
        Z1 = np.dot(X_batch, W1) + b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = Z2  # Linear activation for regression

        # Compute Loss (MSE)
        loss = mean_squared_error(y_batch, A2)
        epoch_train_loss += loss
        num_batches += 1

        # Backpropagation
        dZ2 = (A2 - y_batch) / batch_size
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = np.dot(X_batch.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update Parameters
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    # Average training loss for the epoch
    avg_train_loss = epoch_train_loss / num_batches
    train_losses.append(avg_train_loss)

    # Validation loss
    Z1_val = np.dot(X_val, W1) + b1
    A1_val = relu(Z1_val)
    Z2_val = np.dot(A1_val, W2) + b2
    val_loss = mean_squared_error(y_val, Z2_val)
    val_losses.append(val_loss)

    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Plot Training and Validation MSE
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training MSE')
# plt.plot(val_losses, label='Validation MSE')
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Training and Validation MSE Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Load and Normalize Test Data
X_test = df_test.iloc[:, 1:].values  # Exclude ID column
X_test = (X_test - X_mean) / X_std  # Normalize using training statistics

# Predict on Test Data
Z1_test = np.dot(X_test, W1) + b1
A1_test = relu(Z1_test)
Z2_test = np.dot(A1_test, W2) + b2
y_test = (Z2_test * y_std) + y_mean  # Convert back to original scale

# Save Predictions to CSV
output_df = pd.DataFrame(y_test, columns=["output_x", "output_y", "output_z"])
output_df.insert(0, "id", np.arange(1, len(output_df) + 1))

output_filename = "./CUP24-Out.csv"

with open(output_filename, 'w', newline='') as f:
    f.write("# Charchit Bansal, Sounak Mukopadhyay\n")
    f.write("# The Traders\n")
    f.write("# ML-CUP24 V1\n")
    f.write("# 20/06/2025\n")
    output_df.to_csv(f, index=False, header=False)

print(f"\nPredictions saved to: {output_filename}")
