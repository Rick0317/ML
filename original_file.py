class Main:
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
    