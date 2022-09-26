import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
FullyShardedDataParallel as FSDP,
CPUOffload,
BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
default_auto_wrap_policy,
enable_wrap,
wrap,
)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class Model(nn.Module):

  def __init__(self): # Modelのインスタンスを生成する際の特徴を設定
    super(Model, self).__init__() # ニューラルネットワークの層を作成
    self.layer1 = nn.Sequential() # Sequentialの中に活性化関数を記述する
    self.layer2 = nn.Sequential()
    self.fc = nn.Linear() 
  
  def forward(self, x): # ModelをTrainさせる際に、インプット（x）を入れたときにアウトプットがいくらになるかを計算
    out = self.layer1(x)
    out = self.layer2(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc(out)
    return out



def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train() # Epochの繰り返し処理を行う
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))



def test(model, rank, world_size, test_loader):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            ddp_loss[0] += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
            100. * ddp_loss[1] / ddp_loss[2]))



def train(gpu, model):

  torch.manual_seed(0)
  model = nn.DataParallel(model) #
  torch.cuda.set_device(gpu) #デフォルトのデバイスを定める
  model.cuda(gpu) #モデルを現在用いているデバイスに渡す
  batch_size = 100
  
  criterion = nn.MSELoss().cuda(gpu) # 損失関数を定める
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # 
  
  #model = nn.parallel.DistributedDataParallel(model,device_ids=[gpu])

  train_loader = torch.utils.data.DataLoader(dataset=RandomDataset(100, 200),
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0,
                                              pin_memory=True,
                                              )
  
  

  start = datetime.now()
  total_step = len(train_loader)
  for epoch in range(3500):
      for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        trainOutputs = model(images)
        loss = criterion(trainOutputs, labels)
        print(loss.item())
        loss.backward()
        optimizer.step()

  
        if (i + 1) % 100 == 0 and gpu == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch + 1, 
                args.epochs, 
                i + 1, 
                total_step,
                loss.item())
                )

  if gpu == 0:
      print("Training complete in: " + str(datetime.now() - start))

def fsdp_main(rank, world_size, args):
    setup(rank, world_size) #環境の設定

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ]) 

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform)

    sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    my_auto_wrap_policy = functools.partial(
            default_auto_wrap_policy, min_num_params=100
        )
    torch.cuda.set_device(rank)


    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = Model().to(rank)

    model = FSDP(model)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    init_start_event.record()
    for epoch in range(1, args.epochs + 1):
        train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        test(model, rank, world_size, test_loader)
        scheduler.step()

    init_end_event.record()

    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")

    if args.save_model:
        # use a barrier to make sure training is done on all ranks
        dist_barrier()
        # state_dict for FSDP model is only available on Nightlies for now
        States = model.state_dict()
    if rank == 0:
        torch.save(states, "mnist_cnn.pt")

    cleanup()

# Evaluate the accuracy of the training
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc
    
"""
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
      
"""
if __name__ == '__main__':

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
  
  device = 'cuda' # Choosing the device we will work on
  # Training settings
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                      help='input batch size for training (default: 64)')
  parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                      help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs', type=int, default=10, metavar='N',
                      help='number of epochs to train (default: 14)')
  parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                      help='learning rate (default: 1.0)')
  parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                      help='Learning rate step gamma (default: 0.7)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--save-model', action='store_true', default=False,
                      help='For Saving the current Model')
  args = parser.parse_args()
  torch.manual_seed(args.seed)
  WORLD_SIZE = torch.cuda.device_count()
  mp.spawn(fsdp_main,
      args=(WORLD_SIZE, args),
      nprocs=WORLD_SIZE,
      join=True)
  model = Model()
  train(model)
  torch.save(model.state_dict(), 'model.pth')
