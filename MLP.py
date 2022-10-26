import torch
import torch.nn as nn
import numpy as np
from data_loader import Data_loader
from data_modifier import Modifier
from Model import Net
import torch.optim as optim

def train(model, data, target, optimizer, epoch_size):
  """
  train the given model with input and target data
  """
  criterion = nn.MSELoss()
  model.train()
  for epoch in range(epoch_size):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    #print(f'【EPOCH {epoch}】 loss : {loss.item():.5f}')

def evaluation(model):
  """
  evaluate the trained model.
  """
  pass


if __name__ == "__main__":
  data = Data_loader("data/nse_input.csv", "data/nse_target.csv")
  input_array = data.give_array()[0]
  output_array = data.give_array()[1]
  print("The type of input of the array:", type(input_array[0][0]))
  print("The type of output of the array:", type(output_array[0][0]))
  modifier = Modifier(input_array, data.do_pca())
  input_array = modifier.modify()
  input_array = torch.tensor(input_array)
  output_array = torch.tensor(output_array)
  model = Net()
  optimizer = optim.Adam(model.parameters(), lr=0.01)
  train(model, input_array, output_array, optimizer,1000)
  pred = model(torch.tensor(data.give_array()[2]))
  print(pred)
  print(data.give_array()[3])
  
  

