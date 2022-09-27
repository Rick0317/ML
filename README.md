# Machine Learning
## Using pytorch to achieve Distributed Data Parallel.

## Multi GPUs in Deep Learning
For a large number of inputs, single GPU may take enourmous time to process. 
This is where the concept of multi GPUs comes in.

We use the concept of data parallelism

## How to achieve multiple gpus
We use either Data Parallel, Distributed Data Parallel, or Fully Shared Data Parallel in Pytorch

NVIDIA allows one VM to have multiple GPUs

Essentially, what happens is that the batch size is divided.