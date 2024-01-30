# Predicting Next State of a System with Different Neural Networks
## 5/12/23

## Abstract
This assignment involved evaluating several neural network models to predict the next state of the Lorenz equations, depending on the value of rho.

Neural Network models include a Feed Forward Neural Network (FNN), Long Short-Term Memory network (LSTM), Recurrent Neural Network (RNN), and Echo State Network (ESN).

## Table of Contents
•&emsp;[Introduction and Overview](#introduction-and-overview)

•&emsp;[Theoretical Background](#theoretical-background)

•&emsp;[Algorithm Implementation and Development](#algorithm-implementation-and-development)


&emsp;•&emsp;[Problem 1i](#problem-1i)
&emsp;•&emsp;[Problem 1ii](#problem-1ii)
&emsp;•&emsp;[Problem 2i](#problem-2i)
&emsp;•&emsp;[Problem 2ii](#problem-2ii)
&emsp;•&emsp;[Problem 2iii](#problem-2iii)

•&emsp;[Computational Results](#computational-results)

&emsp;•&emsp;[Problem 1i](#problem-1i-1)
&emsp;•&emsp;[Problem 1ii](#problem-1ii-1)
&emsp;•&emsp;[Problem 2i](#problem-2i-1)
&emsp;•&emsp;[Problem 2ii](#problem-2ii-1)
&emsp;•&emsp;[Problem 2iii](#problem-2iii-1)

•&emsp;[Summary and Conclusions](#summary-and-conclusions)

## Introduction and Overview
The Lorenz system, when initialized and parameterized with different values, gives various chaotic and seemingly random solutions for a three-variable system. Therefore, it is expected to be difficult to predict the future state of the system when just examining the current or previous solutions.

The future state of the Lorenz system can be predicted using neural networks. This is done by examining the current state of the system and predicting the future, or by referencing previous states and the current to predict the future. These methods are explored with FNN, LSTM, RNN, and ESN.

A visualization of the Lorenz system is given in Fig. 1.

![The Lorenz System](https://thumbs.gfycat.com/TangibleSeparateGrison-max-1mb.gif)

> Fig. 1. The Lorenz System

## Theoretical Background
Neural networks involve taking an input and determining the weights between that input and several hidden layers of various sizes. The translation from one layer involves minimizing the loss from chosen activation functions, therefore changing the weights over several iterations of training the neural network.
These activation functions are preferred to be simple to compute the first derivatives of as minimizing the loss function requires taking this derivative. Therefore, the more complex the activation function, the more computationally heavy neural networks are compared to other classification models.

The FNN maps from the current state of the system to the next state, and therefore bases its predictions on solutions made currently. Neural networks can also be customized to retain some memory of its previous solutions, or stepping through time.

The LSTM (Long Short Term Memory) is a neural network that not only references current data points to predict future ones, but also references previous data points to fit the model. The RNN performs well for time-dependent systems since it keeps  a memory of past inputs, therefore letting it predict future states based off of the past. An ESN is a variant of an RNN and randomly initialize a large number of neurons, transforms the data into this higher-dimension space, and then uses that transformed data to predict the future data. The ESN therefore can be computationally heavier.

## Algorithm Implementation and Development
The procedure is discussed in this section. For the results, see [Computational Results](#computational-results).

Since many parts of this assignment were repeated, a few major functions were defined that control most of the output.

Below initializes the Lorenz system data for a given set of rhos.

```py
# Define the Lorenz system equations
def lorenz(x, y, z, sigma=10, rho=28, beta=8/3):
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return dx_dt, dy_dt, dz_dt

# Generate Initial Lorenz system data
x0 = np.random.uniform(-20, 20)
y0 = np.random.uniform(-20, 20)
z0 = np.random.uniform(0, 50)
init = np.zeros((time_steps, 3))
init[0, :] = [x0, y0, z0]

def gen_lorenz(x, rhos):
    X = []
    Y = []

    for rho in rhos:
        for i in range(1, time_steps):
            dx_dt, dy_dt, dz_dt = lorenz(x[i-1, 0], x[i-1, 1], x[i-1, 2], rho=rho)
            x[i, 0] = x[i-1, 0] + dx_dt * dt
            x[i, 1] = x[i-1, 1] + dy_dt * dt
            x[i, 2] = x[i-1, 2] + dz_dt * dt

        # Create the input and output data for the model
        X_rho = np.zeros((time_steps - 10, 10, 3))
        Y_rho = np.zeros((time_steps - 10, 3))
        for i in range(10, time_steps):
            X_rho[i-10,:,:] = x[i-10:i,:]
            Y_rho[i-10,:] = x[i,:]
        
        # Append the data for this rho value to the overall dataset
        X.append(X_rho)
        Y.append(Y_rho)

    # Combine the data for different rho values
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    return X, Y
```
The below function uses the initialization of x and the given rhos to plot a prediction for the future state of the Lorenz system using the trained model.

```py
# Function to plot on rhos
def predict_on(x, rhos, net):
    figsize = (10,5)
    if (len(rhos) == 3):
        figsize=(20,5)

    # Create the figure for plotting the results
    fig, axs = plt.subplots(1, len(rhos), figsize=figsize, subplot_kw={'projection': '3d'})

    # Loop over different values of rho
    for j in range(len(rhos)):
        rho = rhos[j]
        for i in range(1, time_steps):
            dx_dt, dy_dt, dz_dt = lorenz(x[i-1, 0], x[i-1, 1], x[i-1, 2], rho=rho)
            x[i, 0] = x[i-1, 0] + dx_dt * dt
            x[i, 1] = x[i-1, 1] + dy_dt * dt
            x[i, 2] = x[i-1, 2] + dz_dt * dt

        # Create the input and output data for the model
        X = np.zeros((time_steps - 10, 10, 3))
        Y = np.zeros((time_steps - 10, 3))
        for i in range(10, time_steps):
            X[i-10,:,:] = x[i-10:i,:]
            Y[i-10,:] = x[i,:]

        # Make predictions on the testing set
        y_pred = net.predict(X, verbose=0)

        # Evaluate the model on the testing set
        mse = net.evaluate(X, Y, verbose=0)
        print("Mean squared error:", mse, "for rho = {}".format(rho))   
        
        # Plot the actual and predicted trajectories
        axs[j].plot(Y[:,0], Y[:,1], Y[:,2], label='Actual')
        axs[j].plot(y_pred[:,0], y_pred[:,1], y_pred[:,2], label='Predicted')
        axs[j].set_xlabel('X')
        axs[j].set_ylabel('Y')
        axs[j].set_zlabel('Z')
        axs[j].legend()
        axs[j].set_title('rho = {}'.format(rho))

    plt.show()
```
This code was reused to simplify future tasks.


### Problem 1i
For this task, a simple neural network was defined to predict the next state of the lorenz system.

The model was built as an FNN using the keras API.
```py
# Define the FNN model
net_ff = Sequential()
net_ff.add(Flatten(input_shape=(10, 3)))
net_ff.add(Dense(32, activation='relu'))
net_ff.add(Dense(32, activation='relu'))
net_ff.add(Dense(3, activation=None))

net_ff.compile(optimizer='adam', loss='mse')

# Compile the model
net_ff.compile(loss='mse', optimizer='adam')
```

Next, the training and testing datasets were initialized based off of runing `gen_lorenz` which returns the results of the lorenz system with the given `rho` and `init`, both in the functions above.

```py
X, Y = gen_lorenz(init, [10, 28, 40])

# Split the data into training and testing sets
train_size = int(0.9 * len(X))
train_X, test_X = X[:train_size,:,:], X[train_size:,:,:]
train_Y, test_Y = Y[:train_size,:], Y[train_size:,:]

# Train the model
net_ff.fit(train_X, train_Y, epochs=num_epochs, batch_size=32, verbose=2)

# Evaluate the model on the testing set
mse = net_ff.evaluate(test_X, test_Y, verbose=0)
print("Mean squared error:", mse)

# Make predictions on the testing set
y_pred = net_ff.predict(test_X, verbose=0)
```

To visually verify, the model was evaluated on the test dataset and plotted using the function `predict_on(init, [10, 28, 40], net_ff)`.

### Problem 1ii
In this problem, the trained model was used to predict on `rho = [17, 35]`. This was done again through the `predict_on()` function, this time specifying the rho values to the new given ones.

### Problem 2i
For problem 2, different models were built to predict the lorenz system. For part 2i, the LSTM network was built, as an FNN was previously built in problem 1.

For the LSTM, the model was defined as:
```py
# Define the LSTM model
net_lstm = Sequential()
net_lstm.add(LSTM(64, input_shape=(10, 3)))
net_lstm.add(Dense(3))

# Compile the model
net_lstm.compile(loss='mse', optimizer='adam')
```
Using the same training and test dataset as in problem 1, the network was fitted and the evaluated.

```py
# Train the model
net_lstm.fit(train_X, train_Y, epochs=num_epochs, batch_size=32, verbose=2)

# Evaluate the model on the testing set
mse = net_lstm.evaluate(test_X, test_Y, verbose=0)
print("Mean squared error:", mse)

# Make predictions on the testing set
y_pred = net_lstm.predict(test_X, verbose=0)
```

Again, results were visualized by plotting them.
```py
predict_on(init, [10, 28, 40], net_lstm)
```

And finally, the network predicted for `rho = [17, 35]` the same as in problem 1
```
predict_on(init, [17, 35], net_lstm)
```


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

### Problem 2ii
The same format from the previous parts was used again for building an RNN.

The model was built using the SimpleRNN module, and compiled with the same loss function and optimizer as before.

```py
# Define the model architecture
net_rnn = Sequential()
net_rnn.add(SimpleRNN(50, input_shape=(10, 3)))
net_rnn.add(Dense(3))

# Compile the model
net_rnn.compile(loss='mse', optimizer='adam')
```
Again, the model was fitted, evaluated, and then plotted using the same training and testing dataset.
```py
# Train the model
net_rnn.fit(train_X, train_Y, epochs=num_epochs, batch_size=32, verbose=2)

# Evaluate the model on the testing set
mse = net_rnn.evaluate(test_X, test_Y, verbose=0)
print("Mean squared error:", mse)

predict_on(init, [10, 28, 40], net_rnn)

# Make predictions on the testing set
y_pred = net_rnn.predict(test_X, verbose=0)
```

Before eventually being used to predict for `rho=[17, 35]`.
```py
predict_on(init, [17, 35], net_rnn)
```

### Problem 2iii
A similar format to the previous parts was used, but now using the reservoirpy API to build the ESN model, which uses nodes to represent the manipulation of data through the network. The architecture is given below.

```py
# Create the ESN model
data = Input(input_dim=3)
reservoir_node = Reservoir(units=100, size=100, leak_rate=.3, spectral_radius=.9)
readout_node = Ridge(output_dim=3, ridge=1e-2)

net_esn = data >> reservoir_node >> readout_node
```
For `rho = [10, 28, 40]`, the model was fitted and then evaluated with the MSE.

```py
for j in range(len(rhos)):
  rho = rhos[j]

  # Generate the training data
  np.random.seed(42)
  t = np.arange(0, time_steps, dt)
  x = np.zeros_like(t)
  y = np.zeros_like(t)
  z = np.zeros_like(t)
  x[0], y[0], z[0] = x0, y0, z0
  [x[0], y[0], z[0]] = init[0, :]  

  for i in range(1, time_steps):
      x_dot, y_dot, z_dot = lorenz(x[i-1], y[i-1], z[i-1], rho=rho)
      x[i] = x[i-1] + x_dot * dt
      y[i] = y[i-1] + y_dot * dt
      z[i] = z[i-1] + z_dot * dt

  # Split the data into input/output pairs
  input_data = np.column_stack([x, y, z])
  output_data = np.zeros_like(input_data)
  for i in range(len(t)):
      x_dot, y_dot, z_dot = lorenz(x[i], y[i], z[i], rho=rho)
      output_data[i, :] = [x_dot, y_dot, z_dot]

  # Split the data into training and testing sets
  train_size = int(0.9 * len(input_data))
  train_X, test_X = input_data[:train_size, :], input_data[train_size:, :]
  train_Y, test_Y = output_data[:train_size, :], output_data[train_size:, :]

  # Train the ESN
  net_esn.fit(train_X, train_Y)

  # Evaluate the ESN
  y_pred = net_esn.run(test_X)
  
  mse = mean_squared_error(test_Y, y_pred)
  print(f"Mean squared error: {mse}", "for rho = {}".format(rho))
```
Before being used to predict for `rho = [17, 35]`

```py
# Create the figure for plotting the results
fig, axs = plt.subplots(1, len(rhos), figsize=(10, 5), subplot_kw={'projection': '3d'})

for j in range(len(rhos)):
  rho = rhos[j]
  [x[0], y[0], z[0]] = init[0, :]

  for i in range(1, len(t)):
      x_dot, y_dot, z_dot = lorenz(x[i-1], y[i-1], z[i-1], rho=rho)
      x[i] = x[i-1] + x_dot * dt
      y[i] = y[i-1] + y_dot * dt
      z[i] = z[i-1] + z_dot * dt

  # Split the data into input/output pairs
  input_data = np.column_stack([x, y, z])
  output_data = np.zeros_like(input_data)
  for i in range(len(t)):
      x_dot, y_dot, z_dot = lorenz(x[i], y[i], z[i])
      output_data[i, :] = [x_dot, y_dot, z_dot]

  # Evaluate the ESN
  y_pred = net_esn.run(input_data)
  mse = mean_squared_error(output_data, y_pred)
  print(f"Mean squared error: {mse}")

  # Plot the actual and predicted trajectories
  axs[j].plot(output_data[:,0], output_data[:,1], output_data[:,2], label='Actual')
  axs[j].plot(y_pred[:,0], y_pred[:,1], y_pred[:,2], label='Predicted')
  axs[j].set_xlabel('X')
  axs[j].set_ylabel('Y')
  axs[j].set_zlabel('Z')
  axs[j].set_title(f'Lorenz System with Rho = {rho}')
  axs[j].legend()
plt.show()
```


# Computational Results
## Problem 1i
Training for the FNN ran for 100 epochs and therefore results are truncated below.

```
Epoch 1/100
84/84 - loss: 42.0250
Epoch 2/100
84/84 - loss: 4.2466
Epoch 3/100
84/84 - loss: 2.0703
Epoch 4/100
84/84 - loss: 1.3220
Epoch 5/100
84/84 - loss: 0.8810
Epoch 6/100
84/84 - loss: 0.5186
Epoch 7/100
84/84 - loss: 0.3439
Epoch 8/100
84/84 - loss: 0.2601
Epoch 9/100
84/84 - loss: 0.2323
Epoch 10/100
84/84 - loss: 0.1921
Epoch 11/100
84/84 - loss: 0.1720
Epoch 12/100
84/84 - loss: 0.1640
Epoch 13/100
...
84/84 - loss: 0.0256
Epoch 100/100
84/84 - loss: 0.0182
```
The MSE for the test dataset is given as: `0.06890258938074112`

Running the trained model on each rho value (among 10, 28, and 40) individual gave the following MSEs:
```
Mean squared error: 0.0059122247621417046 for rho = 10
Mean squared error: 0.028158457949757576 for rho = 28
Mean squared error: 0.0655554011464119 for rho = 40
```
Additionally, results were visualized in Fig. 2. where the model predicted on the entire dataset (which includes the training data, however this is to visualize overall results). 

![Alt text](https://media.discordapp.net/attachments/1096628827762995220/1106770383794864208/image.png?width%3D1440%26height%3D408)
> Fig. 2. Predictions on rho = 10, 28, 40 with FNN

Predictions on new given rho = 17, 35 in problem 1ii.

### Problem 1ii
Prediction results for rho = 17, 35 are given in Fig. 3, which the MSE for each rho given below.
```
Mean squared error: 0.05537094175815582 for rho = 17
Mean squared error: 0.045406144112348557 for rho = 35
```

![Alt text](https://cdn.discordapp.com/attachments/1096628827762995220/1106770687705759855/image.png)
> Fig. 3. Predictions on rho = 17, 35 with FNN


### Problem 2i
Training for the LSTM ran for 100 epochs and therefore results are truncated below.

```
Epoch 1/100
84/84 - loss: 201.9676
Epoch 2/100
84/84 - loss: 111.5716
Epoch 3/100
84/84 - loss: 75.0263 
Epoch 4/100
84/84 - loss: 54.0072 
Epoch 5/100
84/84 - loss: 40.6446 
Epoch 6/100
84/84 - loss: 31.4211 
Epoch 7/100
84/84 - loss: 24.7960 
Epoch 8/100
84/84 - loss: 20.0167 
Epoch 9/100
84/84 - loss: 16.4701 
Epoch 10/100
84/84 - loss: 13.7667 
Epoch 11/100
84/84 - loss: 11.6711 
Epoch 12/100
84/84 - loss: 10.0492
Epoch 13/100
...
84/84 - loss: 0.0389
Epoch 100/100
84/84 - loss: 0.0200
```
The MSE for the test dataset is given as: `0.034510694444179535`

Running the trained model on each rho value (among 10, 28, and 40) individual gave the following MSEs:
```
Mean squared error: 0.0009619670454412699 for rho = 10
Mean squared error: 0.011930308304727077 for rho = 28
Mean squared error: 0.03743388503789902 for rho = 40
```
Additionally, results were visualized in Fig. 4. where the model predicted on the entire dataset (which includes the training data, however this is to visualize overall results). 


![Alt text](https://media.discordapp.net/attachments/1096628827762995220/1106771074219245579/image.png?width%3D893%26height%3D253)
> Fig. 4. Predictions on rho = 10, 28, 40 with LSTM


Prediction results for rho = 17, 35 are given in Fig. 5, which the MSE for each rho given below.
```
Mean squared error: 0.050436992198228836 for rho = 17
Mean squared error: 0.03178843855857849 for rho = 35
```

![Alt text](https://media.discordapp.net/attachments/1096628827762995220/1106771215877668935/image.png)
> Fig. 5. Predictions on rho = 17, 35 with LSTM


### Problem 2ii
Training for the RNN ran for 100 epochs and therefore results are truncated below.

```
Epoch 1/100
84/84 - loss: 220.7189
Epoch 2/100
84/84 - loss: 154.2753
Epoch 3/100
84/84 - loss: 120.8752
Epoch 4/100
84/84 - loss: 95.0876 
Epoch 5/100
84/84 - loss: 75.0748 
Epoch 6/100
84/84 - loss: 60.2182 
Epoch 7/100
84/84 - loss: 49.0127 
Epoch 8/100
84/84 - loss: 40.4247 
Epoch 9/100
84/84 - loss: 33.7737 
Epoch 10/100
84/84 - loss: 28.5686 
Epoch 11/100
84/84 - loss: 24.3828 
Epoch 12/100
84/84 - loss: 21.0165 
Epoch 13/100
...
84/84 - loss: 0.0284
Epoch 100/100
84/84 - loss: 0.0290
```
The MSE for the test dataset is given as: `0.049022331833839417`

Running the trained model on each rho value (among 10, 28, and 40) individual gave the following MSEs:
```
Mean squared error: 0.0005062147392891347 for rho = 10
Mean squared error: 0.013675092719495296 for rho = 28
Mean squared error: 0.07739883661270142 for rho = 40
```

Additionally, results were visualized in Fig. 6. where the model predicted on the entire dataset (which includes the training data, however this is to visualize overall results). 


![Alt text](https://media.discordapp.net/attachments/1096628827762995220/1106771980415422484/image.png?width%3D893%26height%3D253)
> Fig. 6. Predictions on rho = 10, 28, 40 with RNN


Prediction results for rho = 17, 35 are given in Fig. 7, which the MSE for each rho given below.
```
Mean squared error: 0.07241684198379517 for rho = 17
Mean squared error: 0.02801322750747204 for rho = 35
```

![Alt text](https://media.discordapp.net/attachments/1096628827762995220/1106772109482524772/image.png)
> Fig. 7. Predictions on rho = 17, 35 with RNN


### Problem 2iii
Due to using a different API to build the ESN, training results are of a different format. The MSE for each rho, however, is output.

```
Mean squared error: 0.02733702962525365 for rho = 10
Mean squared error: 1.5139517335993407 for rho = 28    
Mean squared error: 4.475234828698539 for rho = 40
```

Prediction results for rho = 17, 35 are given in Fig. 7, which the MSE for each rho given below.
```
Mean squared error: 2468.304189011738 for rho = 17   
Mean squared error: 9098.272112835088 for rho = 35
```
![Alt text](https://media.discordapp.net/attachments/1096628827762995220/1106772496277061642/image.png)
> Fig. 7. Predictions on rho = 17, 35 with ESN



## Summary and Conclusions
Comparing MSE with `rho = 17`:
FNN: `0.05537094175815582`
LSTM: `0.050436992198228836`
RNN: `0.07241684198379517`
ESN: `2468.304189011738`

Comparing MSE with `rho = 35`:
FNN: `0.045406144112348557`
LSTM: `0.03178843855857849`
RNN: `0.02801322750747204`
ESN: `9098.272112835088`

For `rho = 17`, the LSTM performed best, with the FNN ranking second. For `rho = 35`, the RNN ranked best, with the LSTM second. The ESN struggled to predict on the model even though test results MSE were small. Overall, the FNN, LSTM, and RNN are preferable to the ESN as they have the smallest MSE for both predicted states of the Lorenz system, while the LSTM ranked well for both `rho = [17, 35]`.