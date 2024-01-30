# Determining Sea-Surface Temperature from Minimal Sensor Data using SHRED Neural Network
## 5/16/23

## Abstract
This assignment involved analyzing the neural network, SHRED, built for the paper ["Sensing with shallow recurrent decoder networks"](https://arxiv.org/abs/2301.12011) [1], which uses a combination of long short term memory (LSTM) and a shallow decoder (SDN) to reconstruct sea-surface temperatures requiring far fewer data points (sensors) based the time-history on global sea-surface temperature.
Analysis included measuring performance based on varying the time-history (trajectory of sensory measurements), added noise, and number of sensors.

## Table of Contents
•&emsp;[Introduction and Overview](#introduction-and-overview)

•&emsp;[Theoretical Background](#theoretical-background)

•&emsp;[Algorithm Implementation and Development](#algorithm-implementation-and-development)


&emsp;•&emsp;[Problem 1](#problem-1)
&emsp;•&emsp;[Problem 2](#problem-2)
&emsp;•&emsp;[Problem 3](#problem-3)
&emsp;•&emsp;[Problem 4](#problem-4)
&emsp;•&emsp;[Problem 5](#problem-5)

•&emsp;[Computational Results](#computational-results)

&emsp;•&emsp;[Problem 1](#problem-1-1)
&emsp;•&emsp;[Problem 2](#problem-2-1)
&emsp;•&emsp;[Problem 3](#problem-3-1)
&emsp;•&emsp;[Problem 4](#problem-4-1)
&emsp;•&emsp;[Problem 5](#problem-5-1)

•&emsp;[Summary and Conclusions](#summary-and-conclusions)

## Introduction and Overview
Global sea-surface temperatures are a complex system that may be seemingly very difficult to predict with very few sensors, such as three or four. However, when training a neural network based on measurements in time and transforming the data to as small a dimension as possible while still retaining an optimal representation of the data, it is possible to reconstruct the sea-surface temperatures from this minimal rank of data. This reconstruction can be evaluated by changing the number of sensors, lengthening or shortening the trajectory of the sensors, or even with additional noise, and still performs with surprisngly small error. This model was developed in the paper, ["Sensing with shallow recurrent decoder networks" by Jan P. Williams, Olivia Zahn, and J. Nathan Kutz](https://github.com/Jan-Williams/pyshred). Here, the performance of the model is evaluated with a varied number of sensors, length of time measurements, and with added noise.

A visualization of true and reconstructed global sea-surface temperatures from the referenced paper is given in Fig. 1.

![True vs Reconstructed Sea-Surface Temperatures](https://media.discordapp.net/attachments/1096628827762995220/1108545196754481162/AdKpNKo9Qj8NAAAAAElFTkSuQmCC.png)

> Fig. 1. True vs Reconstructed Sea-Surface Temperatures [1]

## Theoretical Background
The neural network architecture, SHRED, constructed in [1] relies on both an LSTM and SDN. An LSTM references current and previous data points to process, and then this data is sent to an autoencoder, which reduces the dimensionality of the data to as small as it can be, a rank. This transformation ideally, when decoded back to the larger dimensions, will still retain the same information in the data as before. In other words, the difference between the data before and after going through the encoder and decoder should be minimal.

Reducing the dimensionality allows for cheap and quick computation of neural networks containing a large amount of neurons, as reducing the dimensionality of the data greatly reduces the amount of calculations made per neuron.

However, reducing the dimensionality of the data can also provide insight on what the essential or principle components of the data are. In this way, with only a few components, the majority of the data can still be reconstructed using these principle components.

## Algorithm Implementation and Development
The procedure is discussed in this section. For the results, see [Computational Results](#computational-results).

Since many parts of this assignment were repeated, a few major functions were defined that control most of the output.

Below is the function to construct the training, validation, and test dataset. These datasets are dependent on lags, num_sensors, and noise_std, which represent the trajectory, number of sensors, and noise level (Gaussian), respectively.

```py
def train_test_split(lags=52, num_sensors=3, noise_std=0.0):
  sensor_locations = np.random.choice(m, size=num_sensors, replace=False)
  train_indices = np.random.choice(n - lags, size=1000, replace=False)
  mask = np.ones(n - lags)
  mask[train_indices] = 0
  valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]
  valid_indices = valid_test_indices[::2]
  test_indices = valid_test_indices[1::2]

  sc = MinMaxScaler()
  sc = sc.fit(load_X[train_indices])
  transformed_X = sc.transform(load_X)

  ### Generate input sequences to a SHRED model
  all_data_in = np.zeros((n - lags, lags, num_sensors))
  for i in range(len(all_data_in)):
      all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

  if (noise_std != 0.0):
    # Add Gaussian noise to the training data
    noise_std = noise_std  # Standard deviation of the Gaussian noise
    noisy_train_data_in = add_gaussian_noise(all_data_in[train_indices], noise_std)
    train_data_in = torch.tensor(noisy_train_data_in, dtype=torch.float32).to(device)
    train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
  else:
    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
  
  valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
  test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

  ### -1 to have output be at the same time as final sensor measurements
  train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
  valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
  test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

  train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
  valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
  test_dataset = TimeSeriesDataset(test_data_in, test_data_out)

  return train_dataset, valid_dataset, test_dataset, sc, test_indices
```

After separating the training, valdiation, and test dataset, as well as constructing the standard scalar for the data, this information is later passed through to plot the model after it has been trained. This code is shown below.

```py
def train_and_plot(train_dataset, valid_dataset, test_dataset, sc, test_indices, lags=52, num_sensors=3, noise_std=0.0):
  shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
  validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=num_epochs, lr=1e-3, verbose=True, patience=5)

  test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
  test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
  mse =  np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
  print(f"MSE: {mse} for {lags} Trajectory Length and {num_sensors} Sensors with Noise Level of {noise_std}")

  # SST data with world map indices for plotting
  full_SST, sst_locs = load_full_SST()
  full_test_truth = full_SST[test_indices, :]

  # replacing SST data with our reconstruction
  full_test_recon = full_test_truth.copy()
  full_test_recon[:,sst_locs] = test_recons

  # reshaping to 2d frames
  for x in [full_test_truth, full_test_recon]:
      x.resize(len(x),180,360)

  plotdata = [full_test_truth, full_test_recon]
  labels = ['truth','recon']
  fig, ax = plt.subplots(1,2,constrained_layout=True,sharey=True)
  fig.suptitle(f'{lags} Trajectory Length and {num_sensors} Sensors with Noise Level of {noise_std}', y=.78)
  for axis,p,label in zip(ax, plotdata, labels):
      axis.set_axis_off()
      axis.imshow(p[0])
      axis.set_aspect('equal')
      axis.text(0.1,0.1,label,color='w',transform=axis.transAxes)
      
  return mse
  ```

These functions were reused to simplify future tasks.

### Problem 1

For this task, the model from [1] is to uploaded from the github repository.

```py
!git clone https://github.com/shervinsahba/pyshred
```

### Problem 2
In this problem, the model was to be trained, which follows the same code as given in the repository's example. Using the functions above, training the model was done with the code below:

```py
train_dataset, valid_dataset, test_dataset, sc, test_indices = train_test_split()
mse_default = train_and_plot(train_dataset, valid_dataset, test_dataset, sc, test_indices)
```

Once the model is trained, it is evaluated with the parameter changes in problems 3 through 5.

### Problem 3
For this problem, the performance as a function of the time lag variable, or the trajectory of sensor measurements. This was done by defining the range of values the time lag variable would be, then iterating through the model and collecting the MSE from each reconstruction.

Additionally, a few of the results from the individual iterations are plotted to visualize their performance.

```py
lags = np.arange(26, 260) # 6 months to 5 years

mse_lags = np.ndarray((len(lags)))
for i in range(len(lags)):
  lag = lags[i]
  train_dataset_lags, valid_dataset_lags, test_dataset_lags, sc_lags, test_indices_lags = train_test_split(lags=lag)
  mse_lags[i] = train_and_plot(train_dataset_lags, valid_dataset_lags, test_dataset_lags, sc_lags, test_indices_lags, lag)
```
The collected MSEs in mse_lags are then plotted against the increasing time lag variable.
```py
plt.plot(lags, mse_lags)
plt.xlabel("Time Lag")
plt.ylabel("MSE")
plt.title("MSE of Model over Increasing Time Lag")
plt.show()
```


### Problem 4
A similar format to the previous problem was used, but now with added Gaussian noise.

Noise was added to the training dataset at increasing levels, results of the MSE vs the noise level are then plotted.

```py
noise = np.arange(1, 3, .1)
mse_noise = np.ndarray((len(noise)))
for i in range(len(noise)):
  noise_std = noise[i]
  train_dataset_noisy, valid_dataset_noisy, test_dataset_noisy, sc_noisy, test_indices_noisy = train_test_split(noise_std=noise_std)
  mse_noise[i] = train_and_plot(train_dataset_noisy, valid_dataset_noisy, test_dataset_noisy, sc_noisy, test_indices_noisy, noise_std=noise_std)

  plt.plot(noise, mse_noise)
  plt.xlabel("Gaussian Noise Level")
  plt.ylabel("MSE")
  plt.title("MSE of Model over Increasing Noise Level")
  plt.show()
```

### Problem 5
In this problem, the performance of the model was to be plotted against the number of sensors. This was done in a similar format to the previous problems.

First the range of the number of sensors is defined, before splitting the train, validation, and test dataset and training the model to evaluate at greater number of sensors.

```py
sensors = np.arange(1, 20)

mse_sens = np.ndarray((len(sensors)))
for i in range(len(sensors)):
  sens = sensors[i]
  train_dataset_sens, valid_dataset_sens, test_dataset_sens, sc_sens, test_indices_sens = train_test_split(num_sensors=sens)
  mse_sens[i] = train_and_plot(train_dataset_sens, valid_dataset_sens, test_dataset_sens, sc_sens, test_indices_sens, num_sensors=sens)

  plt.plot(sensors, mse_sens)
  plt.xlabel("Number of Sensors")
  plt.ylabel("MSE")
  plt.title("MSE of Model over Increasing Number of Sensors")
  plt.show()
```

# Computational Results
### Problem 1
The github repository was uploaded successfully.

### Problem 2
Due to patience set to 5, the model trained for 899 epochs before beginning validation and testing.

Reconstruction was compared to the ground truth using MSE, and visualized in Fig. 2.
```
MSE: 0.019510094076395035 for 52 Trajectory Length and 3 Sensors with Noise Level of 0.0
```
![Alt text](https://media.discordapp.net/attachments/1096628827762995220/1108611109025489017/image.png)
> Fig. 2. Ground Truth vs Reconstruction for 52 Trajectory Length and 3 Sensors with Noise Level of 0.0

### Problem 3
While the uploaded notebook contains all the MSEs and plots for each value of time lag defined, only a few will be shown here to stay concise. Like in problem 2, the reconstruction based on the new time lag was compared to the ground truth using MSE and was visualized, such as in Fig. 3, 4, and 5.

```
epoch: 1000 valid_error: 0.1123
MSE: 0.01978457160294056 for 26 Trajectory Length and 3 Sensors with Noise Level of 0.0
epoch: 740 valid_error: 0.1104
MSE: 0.020631009712815285 for 39 Trajectory Length and 3 Sensors with Noise Level of 0.0
epoch: 840 valid_error: 0.1102
MSE: 0.019823167473077774 for 52 Trajectory Length and 3 Sensors with Noise Level of 0.0
```

![Alt text](https://media.discordapp.net/attachments/1096628827762995220/1108612554323927080/image.png)
> Fig. 3. Ground Truth vs Reconstruction for 26 Trajectory Length and 3 Sensors with Noise Level of 0.0

![Alt text](https://media.discordapp.net/attachments/1096628827762995220/1108612790119321640/image.png)
> Fig. 4. Ground Truth vs Reconstruction for 39 Trajectory Length and 3 Sensors with Noise Level of 0.0

![Alt text](https://media.discordapp.net/attachments/1096628827762995220/1108612887225835530/image.png)
> Fig. 5. Ground Truth vs Reconstruction for 52 Trajectory Length and 3 Sensors with Noise Level of 0.0


Finally, the performance of the model, measured by MSE, was plotted against the increasing time lag in Fig. 6.

![Alt text](https://media.discordapp.net/attachments/1096628827762995220/1108613429536764054/image.png)
> Fig. 6. Performance of Model as Function of Time Lag

### Problem 4
Again, full results are in the Jupyter Notebook while a few are shown here to stay concise. For this problem, the reconstruction based on added noise to the training dataset was compared to the ground truth using MSE and visualized, such as in Fig. 7, 8, and 9.

```
epoch: 200 valid_error: 0.2617
MSE: 0.05066714808344841 for 52 Trajectory Length and 3 Sensors with Noise Level of 1.0
epoch: 160 valid_error: 0.2421
MSE: 0.047627225518226624 for 52 Trajectory Length and 3 Sensors with Noise Level of 1.1
epoch: 160 valid_error: 0.2592
MSE: 0.047426238656044006 for 52 Trajectory Length and 3 Sensors with Noise Level of 1.2000000000000002
```

![Alt text](https://media.discordapp.net/attachments/1096628827762995220/1108614193546006548/image.png)
> Fig. 7. Ground Truth vs Reconstruction for 52 Trajectory Length and 3 Sensors with Noise Level of 1.0

![Alt text](https://media.discordapp.net/attachments/1096628827762995220/1108614205218762762/image.png)
> Fig. 8. Ground Truth vs Reconstruction for 52 Trajectory Length and 3 Sensors with Noise Level of 1.1

![Alt text](https://media.discordapp.net/attachments/1096628827762995220/1108614222608347197/image.png)
> Fig. 9. Ground Truth vs Reconstruction for 52 Trajectory Length and 3 Sensors with Noise Level of 1.2000000000000002

Finally, the performance of the model, measured by MSE, was plotted against the increasing time lag in Fig. 10.

![Alt text](https://media.discordapp.net/attachments/1096628827762995220/1108614805176193145/image.png)
> Fig. 10. Performance of Model as Function of Noise Level

### Problem 5
Again, full results are in the Jupyter Notebook while a few are shown here to stay concise. For this problem, the reconstruction based on changing the number of sensors was compared to the ground truth using MSE and visualized, such as in Fig. 11, 12, and 13.

```
epoch: 1000 valid_error: 0.1134
MSE: 0.020394427701830864 for 52 Trajectory Length and 1 Sensors with Noise Level of 0.0
epoch: 1000 valid_error: 0.1083
MSE: 0.020075317472219467 for 52 Trajectory Length and 2 Sensors with Noise Level of 0.0
epoch: 1000 valid_error: 0.1102
MSE: 0.019647957757115364 for 52 Trajectory Length and 3 Sensors with Noise Level of 0.0
```

![Alt text](https://media.discordapp.net/attachments/1096628827762995220/1108615061460758598/image.png)
> Fig. 11. Ground Truth vs Reconstruction for 52 Trajectory Length and 1 Sensors with Noise Level of 0.0

![Alt text](https://media.discordapp.net/attachments/1096628827762995220/1108615075167748146/image.png)
> Fig. 12. Ground Truth vs Reconstruction for 52 Trajectory Length and 2 Sensors with Noise Level of 0.0

![Alt text](https://media.discordapp.net/attachments/1096628827762995220/1108615083849953331/image.png)
> Fig. 13. Ground Truth vs Reconstruction for 52 Trajectory Length and 3 Sensors with Noise Level of 0.0

Finally, the performance of the model, measured by MSE, was plotted against the increasing time lag in Fig. 10.

![Alt text](https://media.discordapp.net/attachments/1096628827762995220/1108615110605410387/image.png)
> Fig. 14. Performance of Model as Function of Number of Sensors

## Summary and Conclusions
Observing the performance of the model with increasing trajectory length, the time lag variable, the MSE generally seems to follow a pattern, where around every 60 weeks, the MSE drops, and within this 60 week period the MSE reaches a peak. Possibly, the model may be repeating similar annual patterns in its reconstructions. Each peak is smaller than the previous, so overall, a greater trajectory length improves the performance of the model. This does, however, require more computation to analyze a greater number of previous data points.

For the performance of the model with increasing noise level, the MSE very clearly worsens at higher noise levels, and very quickly. Noise disrupts reconstruction due to additional randomization of the data, which is difficult to fit to. Therefore, it makes sense that at increasing noise levels, the model performs worse.

Increasing the number of sensors shows overall decrease in the MSE. This is likely due to possibly having more unique data from each of the sensors which can represent certain regions of the global sea-surface temperatures better, however it does not decrease the MSE significantly much from the three sensors that was originally tested in [1]. Therefore, including more sensors does not contribute significantly much to the performance, though still improving it.

[1] J. Williams, O. Zahn, and N. Kutz, “Sensing with shallow recurrent decoder networks,” arXiv.org, https://arxiv.org/abs/2301.12011. 