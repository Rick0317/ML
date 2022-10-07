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
    os.environ['MASTER_ADDR'] = 'localhost' #Master Address (IP address) を設定
    os.environ['MASTER_PORT'] = '12355' # Master Portを設定

    dist.init_process_group("nccl", rank=rank, world_size=world_size) # 処理グループを開始する。ncclでNVIDIAのマルチGPUを可能に

def cleanup():
    dist.destroy_process_group() # 処理グループを終了させる

class Model(nn.Module):
  """
  ニューラルネットワーククラス。

  解決すべき課題：
  モデルのレイヤー内にてLinearのArgumentをどうすれば良いか
  fcの特徴
  reshapeのargument
  """

  def __init__(self, inputNum, outputNum): # Modelのインスタンスを生成する際の特徴を設定
    super(Model, self).__init__() # ニューラルネットワークの層を作成
    self.layer1 = nn.Sequential(
      nn.Linear(inputNum, outputNum), # 行列の考え方でテンソルの次元を変える。
      nn.ReLU()
    ) # Sequentialの中に活性化関数を記述する
    self.layer2 = nn.Sequential(
      nn.Linear(),
      nn.ReLU()
    )
    self.fc = nn.Linear(inputNum, outputNum) # fc = Fully Connected Layer 
  
  def forward(self, x): # ModelをTrainさせる際に、インプット（x）を入れたときにアウトプットがいくらになるかを計算
    out = self.layer1(x) # 初めのレイヤーのアウトプットを計算
    out = self.layer2(out) #前レイヤーのアウトプットを入れ、アウトプットを計算
    out = out.reshape(out.size(0), (-1, )) # output.size(0)に対して形を変える。 -1 : ベクトルにする。
    out = self.fc(out) # 
    return out



def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
  """train function
  """
  model.train() #モデルに対して、trainを行うことを伝える。（trainとevaluationで設定を変える必要がある時が存在する）
  ddp_loss = torch.zeros(2).to(rank) 
  criterion = nn.MSELoss().to(rank) # 損失関数を定める
  if sampler:
      sampler.set_epoch(epoch) # samplerはDataSetに対してインデックスを振る。
  for batch_idx, (data, target) in enumerate(train_loader):
      data, target = data.to(rank), target.to(rank) # rankのGPUにインプットと正解値を与える。

      optimizer.zero_grad() # 勾配を０にリセット
      output = model(data) # モデルにインプットを与え、そのアウトプットを格納
      loss = F.nll_loss(output, target, reduction='sum') # 損失関数、アウトプット、正解値をもとに、損失を計算
      loss.backward() #損失から重みを調整
      optimizer.step() # 勾配降下法において坂を降りていく操作。
      ddp_loss[0] += loss.item()
      ddp_loss[1] += len(data)
 
  dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM) # ddp_loss(テンソル)を、全てのマシーンが最後の結果を持つように分配

  if rank == 0: # マルチGPUでの処理が完了し、rank0のGPUに結果が渡された時、以下をprintする。
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

  torch.manual_seed(0) # 乱数の生成に
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

def fsdp_main(rank, world_size, args): #モデルの生成からモデルのtraining、そして保存まで行う。
    setup(rank, world_size) #機械学習の環境の設定、処理グループによる処理を開始させる。 

    transform=transforms.Compose([
        transforms.ToTensor(),  # PILやImageやndarrayをテンソルに変換 計算する際に同じタイプ（クラス）として扱いたいのでこの操作を行う。
        transforms.Normalize((0.1307,), (0.3081,)) #テンソルを平均と標準偏差を用いて正規化
        ]) # transforms.Composeにより複数のtransforms(変換)をまとめる。

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform) # MNISTとは、公開されているサンプルデータ
    # データの取得は独自に行う

    sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True) #datasetをsubset化(分配)する。
    sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1} # 後々に入力されるデータをプロパティとしてデータにする
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2} # 同じく
    cuda_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': False} # 同じく

    train_kwargs.update(cuda_kwargs) # cuda_kwargs dictionaryをtrain_kwargsに追加する
    test_kwargs.update(cuda_kwargs) # cuda_kwargs dictionaryをtest_kwargsに追加する 

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs) # 上記で定めたプロパティをdataset1に対して与える。
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs) # 同じく

    my_auto_wrap_policy = functools.partial(
            default_auto_wrap_policy, min_num_params=100
        )

    torch.cuda.set_device(rank) # 用いるGPUを定める。
    init_start_event = torch.cuda.Event(enable_timing=True) # イベントが時間を測る可動化をenable_timingで定める。
    init_end_event = torch.cuda.Event(enable_timing=True) #同じく

    model = Model().to(rank) #モデルをrankが参照するGPUに渡す。
    model = FSDP(model) # モデルをデータ並行処理ワーカーに分配

    optimizer = optim.Adam(model.parameters(), lr=0.01) #勾配降下法における計算方法を定める。
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma) #学習率を徐々に下げる。（Gammaにがそのレートを定める）(細かくする。)
    init_start_event.record() # イベントを記録する
    for epoch in range(1, args.epochs + 1): #与えられたEpochをもとに、何度このデータをモデルに通すかを決め、その回数分だけ処理する。
        train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1) #上記で定めたtrain関数モデルをtrainする。
        test(model, rank, world_size, test_loader) #上記で定めたtest関数でモデルをtestする。
        scheduler.step() # ステップを増やす。

    init_end_event.record()

    if rank == 0: #マルチGPUでの処理が完了し、rank0のGPUに結果が渡されたとき
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")

    if args.save_model:
        dist_barrier() # モデルのtrainが全てのランクにおいて完了したことを保証。
        # state_dict for FSDP model is only available on Nightlies for now
        states = model.state_dict() #
    if rank == 0:
        torch.save(states, "mnist_cnn.pt") #trainが終了したモデルをファイルとして保存。.ptという拡張子を用いる。

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
  
  # Training の設定を入力するためのparser
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
  args = parser.parse_args() # 入力された情報をargsに格納する。

  torch.manual_seed(args.seed)
  WORLD_SIZE = torch.cuda.device_count() # GPUの数を格納
  mp.spawn(fsdp_main,
      args=(WORLD_SIZE, args),
      nprocs=WORLD_SIZE,
      join=True)