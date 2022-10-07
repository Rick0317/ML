import torch.nn as nn


class Net(nn.Module):
  def __init__(self, inputNum, outputNum):
    super(Net, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Linear(inputNum, outputNum),
      nn.ReLU()
    ) 
    self.layer2 = nn.Sequential(
      nn.Linear(inputNum, outputNum),
      nn.ReLU()
    )
    self.fc = nn.ReLU() # Fully Connected Layer 
  
  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.fc(out)
    return out
    
  

class Main:
  def start(self) -> int:

    inputList = []
    targetList = []
    for j in range(3500): # j should be incremented
      
      d = []
      for i in range(numInput):
        d.append(random.uniform(-1.0, 1.0))
      inputList.append(d)

      d = [0.0] * numOutput
      if inputList[-1][0] > 0:
        d[0] = 1.0
      else:
        d[0] = -1.0
      targetList.append(d)

      # print(inputList)
      # print(targetList)

      trainInputs = torch.tensor(inputList)
      # print(trainInputs)
      trainInputs = trainInputs.to(device)

      trainTargets = torch.tensor(targetList)
      # print(trainTargets)
      trainTargets = trainTargets.to(device)

      net = Net(numInput, numOutput)
      net = net.to(device)
      criterion = torch.nn.MSELoss()
      optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

      for loop in range(1000):
        optimizer.zero_grad()

        trainOutputs = net(trainInputs)
        loss = criterion(trainOutputs, trainTargets)
        print(loss.item())

        loss.backward()
        optimizer.step()

      return 0

if __name__ == '__main__':

  try:
    with open('input.csv', 'r', encoding='shift-jis') as f:
      reader = csv.reader(f) # reader object that reads lines in the give file f
      next(reader)           # Skipping the first line of the file
      inputList = []
      for row in reader:     # row is a list in python
        inputList.extend(row)

  except FileNotFoundError as e:
    print(e)
    sys.exit(0)



  try:
    with open('target.csv', 'r', encoding='shift-jis') as f:
      reader = csv.reader(f)
      next(reader)
      targetList = []
      for row in reader:
        targetList.extend(row)

  except FileNotFoundError as e:
    print(e)
    sys.exit(0)

  # device = 'cpu'
  device = 'cuda' # Choosing the device we will work on

  numInput = 270
  numOutput = 13430