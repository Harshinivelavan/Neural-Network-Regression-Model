# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

<img width="1209" height="883" alt="image" src="https://github.com/user-attachments/assets/97df1656-0992-47d0-80a8-dd506bd05022" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Harshini V
### Register Number: 212224040109
~~~
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# -------------------------------------------------
# Load dataset from CSV
dataset1 = pd.read_csv("k.csv")   # make sure k.csv is in the same folder

# Check columns (optional but useful)
print(dataset1.head())

X = dataset1[['Input']].values
y = dataset1[['Output']].values

# -------------------------------------------------
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=33
)

# -------------------------------------------------
# Scaling (important!)
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

# -------------------------------------------------
# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# -------------------------------------------------
# Neural Network Model
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------------------------------
# Initialize model, loss, optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(ai_brain.parameters(), lr=0.01)

# -------------------------------------------------
# Training function
def train_model(model, X_train, y_train, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        model.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f"Epoch [{epoch}/{epochs}]  Loss: {loss.item():.6f}")

# Train the model
train_model(ai_brain, X_train_tensor, y_train_tensor)

# -------------------------------------------------
# Testing
with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f"Test Loss: {test_loss.item():.6f}")

# -------------------------------------------------
# Loss Plot
plt.plot(ai_brain.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# -------------------------------------------------
# Example prediction
with torch.no_grad():
    sample = torch.tensor([[5.0]])
    sample = x_scaler.transform(sample)
    sample = torch.tensor(sample, dtype=torch.float32)

    pred = ai_brain(sample)
    pred = y_scaler.inverse_transform(pred.numpy())

    print(f"Predicted output for input 5: {pred[0][0]:.2f}")


~~~
## Dataset Information


<img width="186" height="245" alt="image" src="https://github.com/user-attachments/assets/972bbd6b-c3e4-4160-9edc-d6b2121e195c" />





## OUTPUT


<img width="542" height="376" alt="Screenshot 2026-02-10 160254" src="https://github.com/user-attachments/assets/54a127bc-b224-4f75-bf30-9f03235b14b8" />


### Training Loss Vs Iteration Plot



<img width="695" height="559" alt="Screenshot 2026-02-10 160314" src="https://github.com/user-attachments/assets/a4785afd-04ed-4668-a23b-37eca0aa7ac7" />



### New Sample Data Prediction

<img width="414" height="43" alt="Screenshot 2026-02-10 160321" src="https://github.com/user-attachments/assets/b48f2ec7-79ff-43ac-9b96-55edf6bc3d22" />




## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
