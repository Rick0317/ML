import random
import pytorch
import csv
import sys

dev_gpu0 = torch.device("cuda:0")
dev_gpu1 = torch.device("cuda:1")
dev_gpu2 = torch.device("cuda:2")

array = torch.zeros(3)

array_0 = array.to(dev_gpu0)# 1個目のgpuを使う
array_1 = array.to(dev_gpu1)# 2個目のgpuを使う
array_2 = array.to(dev_gpu2)# 3個目のgpuを使う


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

class Net():
  def __init__(self, numin, numout):
    super(Net, self).__init__()
    fc = []
    fc.append(torch.nn.Linear(numin, numout))
    for i in range(10):
      fc.append(torch.nn.Linear(numin, numout))
    self.fc = torch.nn.ModelList(fc)
    self.relu = torch.nn.RELU()
  def forward(self, x):
    for fc in self.fc:
      x = fc(x)
      x = self.relu(x)
    return x

class Main():
  def start(self) -> int:
  # device = 'cpu'
    device = 'cuda' # Choosing the device we will work on

    numInput = 270
    numOutput = 13430

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

  if __name__ == "__main__":
    Main().start()
    