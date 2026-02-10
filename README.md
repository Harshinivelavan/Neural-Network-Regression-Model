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

# ----------------------------------------
# Load Dataset
# ----------------------------------------
dataset1 = pd.read_csv('MyMLData.csv')

X = dataset1[['Input']].values
y = dataset1[['Output']].values

# ----------------------------------------
# Train-Test Split
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=33
)

# ----------------------------------------
# Scaling
# ----------------------------------------
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------
# Convert to PyTorch Tensors
# ----------------------------------------
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ----------------------------------------
# Neural Network Model
# ----------------------------------------
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 12)
        self.fc3 = nn.Linear(12, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Regression output
        return x

# ----------------------------------------
# Initialize Model, Loss, Optimizer
# ----------------------------------------
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)

# ----------------------------------------
# Training Function
# ----------------------------------------
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        outputs = ai_brain(X_train)
        loss = criterion(outputs, y_train)
        
        loss.backward()
        optimizer.step()
        
        ai_brain.history['loss'].append(loss.item())
        
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

# ----------------------------------------
# Train the Model
# ----------------------------------------
train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)

# ----------------------------------------
# Test the Model
# ----------------------------------------
with torch.no_grad():
    predictions = ai_brain(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

# ----------------------------------------
# Plot Loss Curve
# ----------------------------------------
loss_df = pd.DataFrame(ai_brain.history)

plt.figure()
plt.plot(loss_df['loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()


~~~
## Dataset Information

<img width="186" height="346" alt="image" src="https://github.com/user-attachments/assets/c497cc89-ea23-423e-8094-0dc53b6a5a95" />



## OUTPUT

### Training Loss Vs Iteration Plot

<img width="937" height="702" alt="image" src="https://github.com/user-attachments/assets/d24b79ca-97a2-4aeb-9427-ceb9acdab7ad" />


### New Sample Data Prediction

<img width="788" height="99" alt="image" src="https://github.com/user-attachments/assets/3e5d03ee-a26b-4bba-b357-d06f8a68b651" />



## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
