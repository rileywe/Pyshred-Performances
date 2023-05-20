# Pyshred-Performances
This program tests the performance of the PySHRED LSTM/decoder model as it sweeps through different values and input parameters. 

##### Author: 
Riley Estes

### Abstract
This program tests the performance of the SHRED model consisting of an LSTM and a 1 hidden layer decoder in reconstructing sea-surface temperatures. The performance with different data lags, data noise, and number of sensors will be tested. After doing such, there is minimal change in accuracy above at most a lag of 26 and when there are at least 3 sensors, and there is a linear relationship between accuracy and the standard deviation of added noise. 

### Introduction and Overview
This program explores the performance of the model described in the paper "Sensing with shallow recurrent decoder networks" by Jan P. Williams, Olivia Zahn, and J. Nathan Kutz, known as a SHallow REcurrent Decoder (SHRED). Using a sea-surface temperatures from the NOAA Optimum Interpolation SST V2 dataset, the trajectory length (lags), noise, and number of sensors are swept through as parameters to see the effect each one has on the model's performance. This allows the reader to gain more insight into hyperparameter tuning and machine learning model training, and also discovers the best hyperparameters to use for this particular model. The code in this project is adapted from the example code in https://github.com/shervinsahba/pyshred

### Theoretical Background

#### Neural Network
A Neural Network, also known as a Multi-Layer Perceptron (MLP) is a machine learning algorithm where data passes through a series of layers of nodes connected with each other (fully or partially) by weights. This creates a nonlinear and very complicated network because each node in a layer is generally connected to all the nodes in the next layer, each with its own weight. That means that each node in a layer is the sum of all of the nodes in the last layer multiplied by each connection's particular weight. These networks require training with a training set, and are then tested on a test set of data. In training, the values of all the weights are updated (using backpropagation) based on the incoming data (and its labels for supervised learning). The model can then be tested on the test data to see how well it processed the training data, and how well its weights are set to achieve the data processing task. Neural Networks often perform very well on complicated tasks, but require huge amounts of data to do so. 

#### Shallow Recurrent Decoder (SHRED)
A SHRED is a type of neural network architecture that combines an LSTM (encoder) with a decoder with one hidden layer. The LSTM excels at capturing temporal dependencies and patterns in the input data, and the decoder takes these patterns and dependencies and reconstructs the input from the encoded representation. Here is a diagram of the structure of the SHRED taken from the paper "Sensing with shallow recurrent decoder networks" by Jan P. Williams, Olivia Zahn, and J. Nathan Kutz:
<br> <img src="https://github.com/rileywe/Pyshred-Performances/blob/main/SHREDmodel.PNG" width="1000"/>

#### Autoencoder
An autoencoder consists of an encoder and a decoder component. The encoder part processes the input sequence and captures the temporal dependencies in the data, while the decoder part reconstructs the input sequence based on the encoded representation. In this application, the encoder in the SHRED is an LSTM, and the decoder is a shallow version of a simple feed-forward neural network, meaning that is has only one hidden layer. 

#### Recurrent Neural Network (RNN)
An RNN is a type of neural network designed to process sequential/time dependent data. Unlike feedforward neural networks that process data in a single forward pass, an RNN introduces the concept of "recurrence" by allowing information to persist and be passed from one step to the next. This enables the network to maintain an internal memory or state that captures the context and temporal dependencies of the sequential data. This allows the network to notice time-based patterns. 

#### Long Short-Term Memory Neural Network (LSTM)
An LSTM is a type of Recurrent Neural Network that is designed to process sequential data. Similar to an RNN, an LSTM creates feedback loops so that it can "remember" data and use previous data in order to process current data. In addition to this however, the LSTM implements a memory cell where it can selectively store and access data in these cells for later use when processing future information. It adds an extra layer of memory to the Recurrent Neural Network design to further increase its temporal processing abilities. 

### Algorithm Implementation and Development
First, default values for the number of sensors and lags are initialized:
```
num_sensors = 3 
lags = 52
```
Then, each method needed to generate the data, preprocess the data, train and test the model, and plot the results are defined:
```
def genData(lags)
def preprocess(lags=lags, num_sensors=num_sensors, noise_stdev=0.0)
def plot(test_indices, test_recons, title)
def run_model(train_dataset, valid_dataset, test_dataset, sc, sensors=num_sensors)
def plot(test_indices, test_recons, title)
```
In preprocess, the sensor locations are randomly initialized with:
```
sensor_locations = np.random.choice(m, size=num_sensors, replace=False)
```
and the added noise is randomly generated from the Gaussian distribution with the inputted variance:
```
noise = np.random.normal(0, noise_stdev, all_data_in.shape)
all_data_in += noise
```
In run_model, the SHRED model is initialized with 2 size 64 hidden layers in the LSTM:
```
shred = models.SHRED(sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=5)
```
The model is tested on the default parameters, and then swept through the following lists:
```
lag_list = [1, 26, 47, 52, 57, 78, 104]
noise_standard_deviations = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0]
sensor_numbers = [1, 2, 3, 4, 5, 10, 15]
```
and the results are plotted. The same procedure is used for each experiment. 

### Computational Results
Many more images were generated that are not shown here but can be found in the SHRED output images folder. 
Each of the parameter sweeps can be compared to the output of the model with the default values for each parameter shown here:
<br> <img src="https://github.com/rileywe/Pyshred-Performances/blob/main/SHREDOutputImages/baseParams.png" width="400"/>

#### Lags
Time lag does not seem to have very much of an effect on the model's accuracy. The plot of it's performance is shown here:
<br> <img src="https://github.com/rileywe/Pyshred-Performances/blob/main/SHREDOutputImages/lagPlot.png" width="400"/>

As long as the lag isnt extremely low, the accuracy will be about the same regardless of the value. The exception is when the lag is 52, corresponding to one year (and the default value), where the accuracy increases slightly. The output image at that point is shown here:
<br> <img src="https://github.com/rileywe/Pyshred-Performances/blob/main/SHREDOutputImages/lag52.png" width="400"/>

As compared to when the lag is 1 and when it is 78:
<br> <img src="https://github.com/rileywe/Pyshred-Performances/blob/main/SHREDOutputImages/lag1.png" width="400"/>
<br> <img src="https://github.com/rileywe/Pyshred-Performances/blob/main/SHREDOutputImages/lag78.png" width="400"/>

Except for in the extreme cases, the lag has very little effect on the performance of the model. When the lag is equal to one year of measurements, the data lines up a little better, but the effects are minor, and there is no bonus when the lag is equal to two years of measurements. 

#### Noise
The noise has a considerable and almost perfectly linear relationship on the performance of the model shown here:
<br> <img src="https://github.com/rileywe/Pyshred-Performances/blob/main/SHREDOutputImages/noisePlot.png" width="400"/>

The slope of this relationship is approximately -4.12% accuracy per 1.0 increase in noise standard deviation. Notice the sharper drop between 0.1 and 0.25 stdev noise. Somewhere between these two noise levels there is a threshold where the accuracy drops more rapidly and then settles into the linear relationship later on. Before this point between 0.0 and 0.1 stdev noise, the accuracy slope is only -3.3 accuracy/noise. Therefore, it is best to try to keep the noise below 0.1 standard deviations where the model is more efficient in dealing with the noise, but after that the performance will only get linearly worse with an increase in noise. 

The outputs for stdev = 0.1, 0.25, and 2.00 are shown here:
<br> <img src="https://github.com/rileywe/Pyshred-Performances/blob/main/SHREDOutputImages/stdev0.1.png" width="400"/>
<br> <img src="https://github.com/rileywe/Pyshred-Performances/blob/main/SHREDOutputImages/stdev0.25.png" width="400"/>
<br> <img src="https://github.com/rileywe/Pyshred-Performances/blob/main/SHREDOutputImages/stdev2.png" width="400"/>


#### Number of Sensors
Like the lags, the number of sensors also does not seem to have much of an effect on the model accuracy except in extreme cases. The results are shown here:
<br> <img src="https://github.com/rileywe/Pyshred-Performances/blob/main/SHREDOutputImages/sensorsPlot.png" width="400"/>

As long as the model has 3 sensors, the accuracy is not affected by the parameter. Here are some outputs for 1, 2, and 15 sensors:
<br> <img src="https://github.com/rileywe/Pyshred-Performances/blob/main/SHREDOutputImages/sensors1.png" width="400"/>
<br> <img src="https://github.com/rileywe/Pyshred-Performances/blob/main/SHREDOutputImages/sensors2.png" width="400"/>
<br> <img src="https://github.com/rileywe/Pyshred-Performances/blob/main/SHREDOutputImages/sensors15.png" width="400"/>


### Summary and Conclusions
The SHRED composed of an LSTM and shallow, single hidden layer decoder performed very well with 52 lag, 3 sensors, and no noise. The testing done here shows that 52 lag is the best, but anything not extremely low potentially below 26 performs only very slightly worse than 52 lag. The amount of noise added to the data has an almost linear effect with slope -4.12% change in accuracy per 1.0 noise standard deviation. There is a range between 0 and 0.1 that is slightly more efficient, and a range between 0.1 and 0.25 that is less efficient than -4.12, but overall the linear relationship generally holds true. Therefore, it's an issue to have noise, but wont break the model. The number of sensors, much like the lags, does not have an effect on the model accuracy as long as there are at least 3 sensors. Below that, the accuracy may drop by 5% at 1 sensor, and by about 2% at 2 sensors. With this data, it is recommended to use 52 lag, 3 sensors, and to reduce the amount of noise as much as possible, but especially try to keep it below 0.1 Gaussian standard deviations.
