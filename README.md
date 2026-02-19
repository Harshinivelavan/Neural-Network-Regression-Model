# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model


<img width="1081" height="788" alt="image" src="https://github.com/user-attachments/assets/24fd7f08-9f60-4afe-8dc7-1a8d0767db20" />



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


~~~
#Harshini V
#212224040109

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

dataset1 = pd.read_csv("k.csv")   
print(dataset1.head())

X = dataset1[['Input']].values
y = dataset1[['Output']].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=33
)

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

#Name:V.Harshini
#Register number:212224040109
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


ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(ai_brain.parameters(), lr=0.01)

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


train_model(ai_brain, X_train_tensor, y_train_tensor)


with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f"Test Loss: {test_loss.item():.6f}")


# Loss Plot
plt.plot(ai_brain.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

with torch.no_grad():
    sample = torch.tensor([[5.0]])
    sample = x_scaler.transform(sample)
    sample = torch.tensor(sample, dtype=torch.float32)

    pred = ai_brain(sample)
    pred = y_scaler.inverse_transform(pred.numpy())

    print(f"Predicted output for input 5: {pred[0][0]:.2f}")



~~~
## Dataset Information


<img width="172" height="265" alt="image" src="https://github.com/user-attachments/assets/af19fba1-6d62-4070-814b-988b6b1a2ce0" />






## OUTPUT


<img width="448" height="375" alt="image" src="https://github.com/user-attachments/assets/35021d4c-0a71-4313-9453-a30e77be50cf" />






### Training Loss Vs Iteration Plot


<img width="707" height="555" alt="image" src="https://github.com/user-attachments/assets/e4e2f0fc-7094-41c7-aa23-c21922d3d92b" />





### New Sample Data Prediction



<img width="370" height="38" alt="image" src="https://github.com/user-attachments/assets/ccce0bde-88b5-47f6-be02-df7b1e65309b" />







## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
