# Machine Learning

## Math behind Machine Learning (PyTorch)


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