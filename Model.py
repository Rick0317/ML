import torch.nn as nn

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 1, bias=True)

        nn.init.constant_(self.fc.weight, 0.0)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        return self.fc(x)


  