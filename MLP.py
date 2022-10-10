import torch
import torch.nn as nn
import numpy as np
from data_loader import Data_loader
from data_modifier import Modifier
from Model import Model
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim

def train(model, input, target, optimizer, epoch_size):
  """
  train the given model with input and target data
  """
  model.train()
  ddp_loss = torch.zeros(2)
  criterion = nn.MSELoss()
  for epoch in range(epoch_size):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    ddp_loss[0] += loss.item()
    ddp_loss[1] += len(data)
    print(ddp_loss[0], ddp_loss[1])
  dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
  print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))

def evaluation(model):
  """
  evaluate the trained model.
  """

  return accuracy


if __name__ == "__main__":
  data = Data_loader("data/nse_input.csv", "data/nse_target.csv")
  input_array = data.give_array()[0]
  output_array = data.give_array()[1]
  modifier = Modifier(input_array, data.do_pca())
  input_array = modifier.modify()
  model = Model(input_array, output_array)
  optimizer = optim.Adam(model.parameters(), lr=0.01)
  train(model, input_array, output_array, optimizer,5)
  
  

