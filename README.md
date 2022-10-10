# Deep Learning
Machine Learning vs Deep Learning
Deep Learning is a subset of Machine Learning.  
In deep learning, we models human brain using neural network model.  

[PyTorch Official site](https://pytorch.org/)

## Math behind Machine Learning (PyTorch)
### PCA 
In PCA, we measure how much each parameter contributes to the variation of data.  
We draw PC lines and minimize the sum of the squared distances from each data sample.
![pca image](https://github.com/Rick0317/ML/blob/master/images/pca_image.jpg)  
### Activation Functions
ReLU, Sigmoid, Tanh: [Plot and formula](https://drive.google.com/file/d/10xfankx86CWyhhsZhU4lHnslAARFZz_4/view?usp=sharing)  


## Data Arrangement

### First, we want to load the data from a csv file.
We use a build-in function "raw_data = open({file_name}, 'rt')" to open the file.('rt' = reading text, 'wt' = writing text)    
If the file contains a header, we can skip it by "next(raw_data)"  
Next, we use "csv.reader()" to read the raw data. This return a reader object so to use it, we change its type to list.  

### Next, we want to rearrange the shape of the table. (PCA)
In our case of nse_data.csv, we deleted the date column. I used excel to import the data and deleted the column and saved the file as a new file. In the process, we separated the input and target data.  
Now, we will prepare our data, which is the most important step.  
We use PCA to select the features of input data we will use.  
The PCA module from sklearn.decomposition is used. It calculates the variance of the given data. 

## Model 

### What kind of Model to use
The neural network model you should use depends on what you want to achieve. 


### Layer
Input Layer, Hidden Layer, Output Layer    
Fully connected layer: Layer for which all the inputs from one layer are connected to every activation unit of the next layer with weights  
Ex) nn.Linear() which changes the dimensionality of the input as it yields the output.  
  
There is no right number of layers you should use.

### CNN (Convolution Neural Network)
To identify the image that is difficult for a computer to do so, we use CNN, which is a certain set of layers.  
They have filters in their layer and this identifies the images.  
nn.Conv2d(in_channels, out_channels, kernel_size) kernel_size: specifying the height and width of the 2D convolution window
Convolution window: subset of the images you want to identify.

### How do we choose activation functions?
Usually the activation functions in the hidden layers are the same through the Neural Network model.  
Activation functions are chosen based on the architecture of Neural Network model.(MLP, CNN: ReLU, Recurrent Network: Tanh)  
Try different activation functions and compare the results is the best thing to do.

For the output layer, people usually use Linear, Sigmoid, Softmax
The activation functions for output layer depend on the problem we solve.  
For regression, we use Linear.  
For Binary Classification and Multilabel Classification, we use sigmoid.  
For Multiclass Classification, we use softmax.


[reference](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/)

## Using pytorch to achieve Distributed Data Parallel.
FSDP: Fully Shared Data Parallel

It basically splits the given data and calculate each set of them using different devices. Each device calculates forward path and backward path individually and update the weights. At the end of each iteration, the update information will be shared with the rank0 GPU. 

## Multi GPUs in Deep Learning
For a large number of inputs, single GPU may take enourmous time to process. 
This is where the concept of multi GPUs comes in.

We use the concept of data parallelism

## How to achieve multiple gpus
We use either Data Parallel, Distributed Data Parallel, or Fully Shared Data Parallel in Pytorch

NVIDIA allows one VM to have multiple GPUs

NCCL(NVIDIA Collective Communication Library) implements multi-GPU and multi-node communication primitives optimized for NVIDIA GPUs and Networking

Essentially, what happens is that the batch size is divided.