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
        x= self.relu(self.fc2(x))
        x= self.fc3(x) # No activation here since it's a regression task
        return x



# Initialize the Model, Loss Function, and Optimizer

ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim. RMSprop(ai_brain.parameters(), lr=0.001)

def train_model (ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
  for epoch in range(epochs):
    optimizer.zero_grad()
    loss = criterion (ai_brain (X_train), y_train)
    loss.backward()
    optimizer.step()

    ai_brain.history['loss'].append(loss.item())
    if epoch % 200 == 0:
      print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


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
