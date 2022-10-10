import torch.nn as nn

class Model(nn.Module):

  def __init__(self, input_data, target):
    super(Model, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Sigmoid(),
      nn.ReLU(), 
      nn.Sigmoid(),
      nn.ReLU()
    )
    self.layer2 = nn.Sequential(
      nn.Sigmoid(),
      nn.ReLU()
    )
    self.fc = nn.Sigmoid()


  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.fc(out)
    return out

  