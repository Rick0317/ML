# Machine Learning
[PyTorch Official site](https://pytorch.org/)

## Math behind Machine Learning (PyTorch)
### PCA 
In PCA, we measure how much each parameter contributes to the variation of data.  
We draw PC lines and minimize the sum of the squared distances from each data sample.
![pca image](https://github.com/Rick0317/ML/blob/master/images/pca_image.jpg)
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