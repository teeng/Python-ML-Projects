# Introduction to Neural Networks
## 5/8/23

## Abstract
This assignment involved evaluating two variations of neural networks against previously-explored classification models on the MNIST dataset. This dataset contains 70,000 images of 28x28 images of a single handwritten digit, with the images labeled for what number they represent. Each of the images therefore have 784 features (28x28).

Classification methods include a three-layer feed forward neurel network, an LMST, SVM, and Decision Tree classifier.

## Table of Contents
•&emsp;[Introduction and Overview](#introduction-and-overview)

•&emsp;[Theoretical Background](#theoretical-background)

•&emsp;[Algorithm Implementation and Development](#algorithm-implementation-and-development)


&emsp;•&emsp;[Problem 1i](#problem-1i)
&emsp;•&emsp;[Problem 1ii](#problem-1ii)
&emsp;•&emsp;[Problem 1iii](#problem-1iii)
&emsp;•&emsp;[Problem 1iv](#problem-1iv)
&emsp;•&emsp;[Problem 2i](#problem-2i)
&emsp;•&emsp;[Problem 2ii](#problem-2ii)

•&emsp;[Computational Results](#computational-results)

&emsp;•&emsp;[Problem 1i](#problem-1i-1)
&emsp;•&emsp;[Problem 1ii](#problem-1ii-1)
&emsp;•&emsp;[Problem 1iii](#problem-1iii-1)
&emsp;•&emsp;[Problem 1iv](#problem-1iv-1)
&emsp;•&emsp;[Problem 2i](#problem-2i-1)
&emsp;•&emsp;[Problem 2ii](#problem-2ii-1)

•&emsp;[Summary and Conclusions](#summary-and-conclusions)

## Introduction and Overview
Classifying images through neural network generally requires a large dataset, which can be satisfied with the MNIST dataset. Since this dataset was previously analyzed in other homeworks, we can compare the classification of a neural network to these previous classification models.

Classifiers applied were a three-layer feed-forward neural network, as well as a network implementing LMST, a support vector machine (SVM), and a decision tree classifier. These were compared on how accurately they classified the ten digits from the MNIST dataset.

A sample of the MNIST dataset is given in Fig. 1 with their corresponding labels.

![MNIST](https://media.discordapp.net/attachments/1096628827762995220/1105292433987747840/image.png)

> Fig. 1. Sample images from the "MNIST" dataset

## Theoretical Background
Neural networks involve taking an input and determining the weights between that input and several hidden layers of various sizes. The translation from one layer involves minimizing the loss from chosen activation functions, therefore changing the weights over several iterations of training the neural network.
These activation functions are preferred to be simple to compute the first derivatives of as minimizing the loss function requires taking this derivative. Therefore, the more complex the activation function, the more computationally heavy neural networks are compared to other classification models.

To make processing the MNIST dataset somewhat less computationally heavy, the PCA for 20 components was computed to reduce the dimensionality of the data to 20 features, instead of 784 like before. This requires minimizing less parameters across each layer and therefore eases the burden of building the neural network.

The other classification models involved in this assignment are the LSTM, the SVM, and the decision tree classifier. These last two classifiers were previously explored in homework 3, and so their explanations are repeated here.

The LSTM (Long Short Term Memory) is a neural network that not only references current data points to predict future ones, but also references previous data points to fit the model.

The SVM splits data by approximating a plane through all of the features (so in this case, a 784 dimension plane) that will split the data. By having a plane through this many dimensions, the SVM also has a buffer space for where the separation line could be, and wants to maximize the distance from the line to the edge of the buffer (finding the "most middle spot" the line could be between two different classes).

The decision tree classifier does not perform optimization and instead selects the principle components that best split the data and iterates downward, finding the next best feature to split the data into different classes.

## Algorithm Implementation and Development
The procedure is discussed in this section. For the results, see [Computational Results](#computational-results).

### Problem 1i
Firstly, a sample three-layer neural network was fitted to the input data below:

```py
X = np.arange(0, 31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
```

The neural network was defined below, with the first layer containing three neurons, and the second layer containing two. This was due to observations of this data made from the first homework, where not many dimensions were needed to fit the data. Therefore, the layers do not have to be large.

```py
# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 3)
        self.fc2 = nn.Linear(3, 2)
        self.fc3 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

After defining the network architecture, it is instantiated and fitted to the data. The loss function is the least square error, and the optimizer is Stochastic Gradient Descent (SGD).

```py
# Initialize the network and define the loss function and optimizer
net = Net()

# Loss function and optimization
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
```

Finally the network is train on 3000 epochs (due to the simplicity of the data).
```py
# Train the network
for epoch in range(num_epochs):
    # Forward pass
    outputs = net(X)
    loss = criterion(outputs, Y)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # print loss every 50 epochs
    if (epoch+1) % 100 == 0:
      print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
```

After training, the accuracy of the model. However, since we trained this model using the entirety of the dataset (which is often not recommended but was requested by this problem), the test dataset is the same as the train dataset. Therefore, we visualize the accuracy of the model by printing its output, or predicted y-values, for each of the inputs.

```py
# Test the model
with torch.no_grad():
    y_pred = net(X)
    print('Predicted Y: ', y_pred.numpy().flatten())
```


### Problem 1ii
In this problem, the dataset was properly split where the first 20 points were used as training data, and the rest were used for the test dataset. The process in problem 1i was therefore replicated but with the split datasets.

The data was split through the code below:
```py
# Load the MNIST dataset and apply transformations
# Data to tensors
X_train = torch.Tensor(X_raw[:20].reshape(-1, 1))
Y_train = torch.Tensor(Y_raw[:20].reshape(-1, 1))
train_dataset = TensorDataset(X_train, Y_train)

X_test = torch.Tensor(X_raw[-10:].reshape(-1, 1))
Y_test = torch.Tensor(Y_raw[-10:].reshape(-1, 1))
test_dataset = TensorDataset(X_test, Y_test)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, shuffle=False)
```

And the network from problem 1i was trained using the new dataset.

```py
# Train the network
for epoch in range(num_epochs):
  for i, (inp, lbl) in enumerate(train_loader):
    # Forward pass
    outputs = net(inp)
    loss = criterion(outputs, lbl)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # print loss every 100 epochs
  if (epoch+1) % 100 == 0:
    print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
```

The least square error on the test dataset is printed through the code below:
```py
# Compute the least square error on the test data
y_pred = net(X_test)
test_error = (((Y_test - y_pred)**2).mean())**1/2
print('Test error: {:.4f}'.format(test_error))
```

### Problem 1iii
This problem was similar to problem 1ii, but using the first and last ten datapoints as training and the rest as testing. Therefore, the format for the code is generally the same.

```py
# Load the MNIST dataset and apply transformations
# Data to tensors
X_train = torch.Tensor(np.concatenate((X_raw[:10], X_raw[-10:])).reshape(-1, 1))
Y_train = torch.Tensor(np.concatenate((Y_raw[:10], Y_raw[-10:])).reshape(-1, 1))
train_dataset = TensorDataset(X_train, Y_train)

X_test = torch.Tensor(X_raw[10:21].reshape(-1, 1))
Y_test = torch.Tensor(Y_raw[10:21].reshape(-1, 1))
test_dataset = TensorDataset(X_test, Y_test)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, shuffle=False)
```
Once splitting the training and test dataset, the model was fit to the training data.

```py
# Train the network
for epoch in range(num_epochs):
  for i, (inp, lbl) in enumerate(train_loader):
    # Forward pass
    outputs = net(inp)
    loss = criterion(outputs, lbl)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # print loss every 50 epochs
  if (epoch+1) % 100 == 0:
    print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
```

Results were evaluated by computing the least square error through the following code:

```py
# Compute the least square error on the test data
y_pred = net(X_test)
test_error = (((Y_test - y_pred)**2).mean())**1/2
print('Test error: {:.4f}'.format(test_error))
```

### Problem 1iv
For this task, the accuracy of the neural networks constructed were compared to homework 1's performance on the same dataset using curve fitting. The discussion is given in the [results section](#problem-1iv-1).

### Problem 2i
For problem 2, the 70,000 image MNIST dataset was used to build the neural network and several other classification models.

The objective for problem 2 in general was to replicate problem 1 but on the MNIST dataset and using different classification models, with 60,000 images as the training dataset and 10,000 images as the test dataset.

First the MNIST dataset was loaded.

```py
# Load the MNIST data
mnist = fetch_openml('mnist_784', parser="auto")
y = mnist.target
X = mnist.data / 255.0  # Scale the data to [0, 1]
```
For part (i), the first 20 components of PCA was computed to reduce the dimensionality of the MNIST data, which originally has 784 features.

The PCA was computed below:
```py
# Apply PCA to reduce the dimensionality of the data
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X)
```

To visualize the 20 components, they were also plotted. Results are given [below](#problem-2i-1)

```py
# Plot the 20 principal components
fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(5, 5))
axs = axs.ravel()

for i in range(20):
    axs[i].imshow(pca.components_[i].reshape(28, 28), cmap='gray')
    axs[i].axis('off')
    axs[i].set_title(f'Mode {i+1}')

plt.suptitle("PCA Modes for 20 Dimensions")
plt.tight_layout()
plt.show()
```

### Problem 2i
The next task involved building several models for classifying the digits using the new MNIST dataset that was reduced to 20 dimensions.

The dataset was still split with 60,000 training images and 10,000 images, but now transformed using the 20 component PCA calculated in [problem 2i](#problem-2i).

The first model is a three-layer feed-forward network. This was done by loading keras API. The optimizer for the neural network is SGD, with the Cross Entropy loss function, using batch size of 64 over 5 epochs.

```py
# Define and train a neural network on the normalized and PCA-transformed data
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=20))
model.add(Dense(units=10, activation='softmax'))
model.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.1)

# Evaluate the performance of the neural network on the testing data
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc * 100)
```

The next model used LSTM, which is given below with the same hyperparameters, optimization, and loss function.

```py
# Reshape the data into sequences for the LSTM layer
n_timesteps, n_features = 20, 1
X_lstm = X_pca.reshape((X_pca.shape[0], n_timesteps, n_features))

# Split the data into training and testing sets
train_size = 60000
X_train, X_test = X_lstm[:train_size], X_lstm[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define and train a neural network with LSTM layers
model = Sequential()
model.add(LSTM(units=64, input_shape=(n_timesteps, n_features)))
model.add(Dense(units=10, activation='softmax'))
model.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.1)

# Evaluate the performance of the neural network on the testing data
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
```

The next model is a decision tree classifier, which was built below:
```py
# Train decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate performance on test set
acc = clf.score(X_test, y_test)
print(f"Accuracy on test set: {acc*100:.2f}%")
```

Along with an SVM (Support Vector Machine)
```py
# use SVC (support vector machine classifier) for classification
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

# evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy * 100, "%", sep="")
```


# Computational Results
## Problem 1i
The results of each 100th epoch over 1000 epochs are given below.

```
Epoch [100/1000], Loss: 292.1619
Epoch [200/1000], Loss: 122.9426
Epoch [300/1000], Loss: 199.0020
Epoch [400/1000], Loss: 175.4856
Epoch [500/1000], Loss: 144.8939
Epoch [600/1000], Loss: 164.5882
Epoch [700/1000], Loss: 119.8296
Epoch [800/1000], Loss: 82.0596
Epoch [900/1000], Loss: 59.7308
Epoch [1000/1000], Loss: 45.8013
```
The resulting predictions of the neural network are printed:
```
Predicted Y:
[33.17927  34.110886 35.042507 35.97412  36.905735 37.837357 38.76897
 39.700584 40.62548  41.545757 42.46603  43.386303 44.306583 45.226856
 46.14713  47.067406 47.987682 48.907955 49.82823  50.74851  51.668785
 52.589058 53.509335 54.429615 55.349884 56.270157 57.190437 58.110718
 59.030987 59.951263 60.871536]
```

### Problem 1ii


### Problem 1iii

The results of each 100th epoch over 1000 epochs are given below.
```
Epoch [100/1000], Loss: 6.1781
Epoch [200/1000], Loss: 1.2824
Epoch [300/1000], Loss: 6.0643
Epoch [400/1000], Loss: 5.7211
Epoch [500/1000], Loss: 44.5242
Epoch [600/1000], Loss: 0.8790
Epoch [700/1000], Loss: 0.2357
Epoch [800/1000], Loss: 0.0001
Epoch [900/1000], Loss: 15.3755
Epoch [1000/1000], Loss: 7.8360
```
The general loss at each epoch seems to be much less than before. By using the first 20 datapoints, the test error is found to be `28.9862`

For using the first and last ten datapoints for training, the results of each 100th epoch over 1000 epochs are given below.

```
Epoch [100/1000], Loss: 185.4745
Epoch [200/1000], Loss: 61.3059
Epoch [300/1000], Loss: 14.2544
Epoch [400/1000], Loss: 11.0838
Epoch [500/1000], Loss: 0.0008
Epoch [600/1000], Loss: 0.6088
Epoch [700/1000], Loss: 22.1057
Epoch [800/1000], Loss: 16.0067
Epoch [900/1000], Loss: 4.2800
Epoch [1000/1000], Loss: 0.2705
```
The general loss at each epoch seems to be much less than before. By using the first 20 datapoints, the test error is found to be `9.6754`

### Problem 1iv
Similar to the results of homework 1, using the first 10 and last 10 data points as training dataset had greater accuracy than using the first 20 data points.

Comparing classification methods, using the neural network had a greater least square error than the Line and Parabola fit, meaning that the prediction for y were worse than with the classification methods in homework 1. Only the 19th degree polynomial had a greater least square error, however this is due to the polynomial's extreme overfitting on the training data.


### Problem 2i
The MNIST dataset is PCA-transformed to have 20 components. These 20 modes are shown in Fig. 2.

![Alt text](https://media.discordapp.net/attachments/1096628827762995220/1105263402940645376/image.png)
> Fig. 2. PCA Modes for 20 Dimensions

### Problem 2ii
First, the three-layer neural network was built on the PCA-transformed dataset. The loss function results of each 100th batch over five epochs is given below as well. The resulting test accuracy is `92.25%`

```
Epoch 1/5, batch: 844
loss: 0.6706 - accuracy: 0.7984 - val_loss: 0.3469 - val_accuracy: 0.8998
Epoch 2/5, batch: 844
loss: 0.3765 - accuracy: 0.8868 - val_loss: 0.2810 - val_accuracy: 0.9190
Epoch 3/5, batch: 844
loss: 0.3225 - accuracy: 0.9018 - val_loss: 0.2496 - val_accuracy: 0.9250
Epoch 4/5, batch: 844
loss: 0.2913 - accuracy: 0.9103 - val_loss: 0.2288 - val_accuracy: 0.9325
Epoch 5/5, batch: 844 + 313
loss: 0.2700 - accuracy: 0.9164 - val_loss: 0.2161 - val_accuracy: 0.9362
loss: 0.2518 - accuracy: 0.9225
```

Next an LSTM layer was used in a neural network. Results are also given below, and test accuracy was found to be: `47.24%`, which is very poor compared to the previous neural network.

```
Epoch 1/5, batch: 844
loss: 2.2575 - accuracy: 0.2054 - val_loss: 2.1956 - val_accuracy: 0.2512
Epoch 2/5, batch: 844
loss: 2.1175 - accuracy: 0.2890 - val_loss: 2.0160 - val_accuracy: 0.3357
Epoch 3/5, batch: 844
loss: 1.9350 - accuracy: 0.3434 - val_loss: 1.8246 - val_accuracy: 0.3745
Epoch 4/5, batch: 844
loss: 1.7827 - accuracy: 0.3923 - val_loss: 1.6942 - val_accuracy: 0.4223
Epoch 5/5, batch: 844 + 313
loss: 1.6721 - accuracy: 0.4293 - val_loss: 1.5822 - val_accuracy: 0.4765
loss: 1.5958 - accuracy: 0.4725
```

Next, a Decision Tree Classifier was built with the PCA dataset. The resulting test accuracy is: `84.11%`
The tree is also saved as `tree_1.dot` and the code to visualize it is saved in the `HW4_TJ.ipynb` file, however because it is an extremely large image and cannot be condensed without maintaining text readability, it will not be pasted here.

Finally, an SVM was made to classify the digits. Its resulting accuracy was `96.22%`.

Overall, the SVM performed the best, with the neural network without LSTM ranking second with an accuracy above 90%, the decision tree classifier being third with a moderate 84%, and LSTM being and atrocious fourth with a deplorable accuracy of 47.24% on the test dataset.

## Summary and Conclusions
Analyzing the differences between the classifiers, the neural network seems to perform better on a larger dataset, struggling more on a smaller dataset such as in problem 1, where other classifiers would perform better (as seen in homework 1). In problem 2 with a much larger dataset, the neural network and SVM both performed very well comparatively. However, both the SVM and neural network are somewhat computationally heavy, while the decision tree classifier gives a good representation of how to interpret the model.

