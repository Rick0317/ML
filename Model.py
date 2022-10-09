import torch
import torch.nn as nn

class NN(nn.Module):

  def __init__(self):
    super(NN, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(),
      nn.ReLU(), 
      
    )