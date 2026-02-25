# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and dataset

To develop a Convolutional Neural Network (CNN) model to classify grayscale images into 10 distinct categories using deep learning techniques.
To train, evaluate, and verify the model’s performance using confusion matrix, classification report, and prediction on new sample images.
<img width="1042" height="223" alt="image" src="https://github.com/user-attachments/assets/7338f580-9587-4e5b-b0cd-e622d613509d" />

## Neural Network Model
<img width="1038" height="526" alt="image" src="https://github.com/user-attachments/assets/14b2fef3-4d5b-4071-978e-2d9064d319ee" />


## DESIGN STEPS

### STEP 1:
Import required libraries such as PyTorch, Torchvision, and Matplotlib in Google Colab.
Load the dataset and apply necessary transformations like tensor conversion and normalization.

### STEP 2:
Create a CNN model using convolution layers, ReLU activation, and max pooling.
Add fully connected layers for classification and define the forward propagation function.

### STEP 3:
Initialize the model, loss function (CrossEntropyLoss), and optimizer (Adam).
Perform forward propagation, compute loss, backpropagate errors, and update weights for multiple epochs.

### STEP 4:
Test the trained model using test data and compute performance metrics.
Generate confusion matrix, classification report, and predict output for new sample images.

## PROGRAM

### Name: V KAMALESH VIJAYAKUMAR
### Register Number:212224110028
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(128*3*3,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)
    def forward(self, x):
      x = self.pool(torch.relu(self.conv1(x)))
      x = self.pool(torch.relu(self.conv2(x)))
      x = self.pool(torch.relu(self.conv3(x)))
      x=x.view(x.size(0),-1)
      x=torch.relu(self.fc1(x))
      x=torch.relu(self.fc2(x))
      x=self.fc3(x)
      return x



```

```python
# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

```python
# Train the Model
## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=3):
  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
    print('Name:V KAMALESH VIJAYAKUMAR')
    print('Register Number:212224110028')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


```

## OUTPUT
### Training Loss per Epoch
<img width="534" height="217" alt="image" src="https://github.com/user-attachments/assets/a8392269-7b2d-4332-868c-9b6f61b5df41" />


### Confusion Matrix
<img width="713" height="611" alt="image" src="https://github.com/user-attachments/assets/bfe64b87-6a71-49bd-8cab-636d4034d3e3" />


### Classification Report
<img width="665" height="442" alt="image" src="https://github.com/user-attachments/assets/6e85c2a6-1b81-40ed-a4f7-8b0632f1f14a" />



### New Sample Data Prediction
<img width="627" height="625" alt="image" src="https://github.com/user-attachments/assets/3675d7c0-a587-469e-873e-33e66a2239cb" />


## RESULT
The Convolutional Neural Network model was successfully trained for image classification and achieved satisfactory performance on the test dataset.
The model correctly classified new sample images, validating its effectiveness and accuracy.
