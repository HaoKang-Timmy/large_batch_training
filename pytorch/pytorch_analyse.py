import numpy as np
np.random.seed(1337)
import torch
torch.manual_seed(1337)
from copy import deepcopy
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
model = torchvision.models.vgg11_bn()
model.classifier[-1]= nn.Linear(4096,10)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
X_train = torch.utils.data.DataLoader(
    trainset, batch_size=16, shuffle=True, num_workers=12)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
X_test = torch.utils.data.DataLoader(
    testset, batch_size=1000, shuffle=False, num_workers=12)
criterion = nn.CrossEntropyLoss()

x0 = deepcopy(model.state_dict())
for batch_size in [256,5000]:
    
    optimizer = torch.optim.Adam(model.parameters())
    model.load_state_dict(x0)
    X_train = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=12)
    X_test = torch.utils.data.DataLoader(
            testset, batch_size=1000, shuffle=False, num_workers=12)
    average_loss_over_epoch = '-'
    average_acc = '-'
    np.random.seed(1337)
    for e in range(100):
        model.eval()
        print ('Epoch:', e, ' of ', 100, 'Average loss:', average_loss_over_epoch,"average_acc:",average_acc)
        average_loss_over_epoch = 0.

        torch.save(model.state_dict(), ('SB' if batch_size==256 else 'LB')+'vgg.pth')
        train_loss = 0
        correct = 0
        total = 0
        model.train()
        for batch_idx, (inputs, targets) in enumerate(X_train):
        
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            average_loss_over_epoch = train_loss / total
            average_acc = correct / total


trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
X_train = torch.utils.data.DataLoader(
    trainset, batch_size=1000, shuffle=True, num_workers=12)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
X_test = torch.utils.data.DataLoader(
    testset, batch_size=1000, shuffle=False, num_workers=12)

mbatch = torch.load('LBvgg.pth')
mstoch = torch.load('SBvgg.pth')
i = 0

# Fill in the train and test, loss and accuracy values
# for `grid_size' points in the interpolation
grid_size = 25 #How many points of interpolation between [-1, 2]
data_for_plotting = np.zeros((grid_size, 4))
alpha_range = np.linspace(-1, 2, grid_size)
criterion = nn.CrossEntropyLoss()
for alpha in alpha_range:
    mydict = {}
    for key, value in mbatch.items():
        mydict[key] = value * alpha + (1 - alpha) * mstoch[key]
    model = torchvision.models.vgg11_bn()
    model.classifier[-1]= nn.Linear(4096,10)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
    model.load_state_dict(mydict)
    print("finish")
    testloss = trainloss = testacc = trainacc = 0.
    j = 0
    # for datatype in [(X_train, y_train), (X_test, y_test)]:
    #     dataX = datatype[0]
    #     datay = datatype[1]
    #     for smpl in np.split(np.random.permutation(range(dataX.shape[0])), 10):
    #         ops = opfun(dataX[smpl])
    #         tgts = Variable(torch.from_numpy(datay[smpl]).long().squeeze())
    #         data_for_plotting[i, j] += F.nll_loss(ops, tgts).data.numpy()[0] / 10.
    #         data_for_plotting[i, j+2] += accfun(ops, datay[smpl]) / 10.
    #     j += 1
    total = 0.
    correct = 0.
    total_loss = 0.
    for batch_idx, (inputs, targets) in enumerate(X_train):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        check = int((outputs != outputs).sum())
        # if(check>0):
        #     print("your data contains Nan")
        # else:
        #     print("Your data does not contain Nan, it might be other problem")
        #print(inputs)
        loss = criterion(outputs, targets)
        # print(inputs.size())
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        total_loss +=loss.item()
        #print(total,total_loss,correct)
    data_for_plotting[i, j] = total_loss / total
    data_for_plotting[i,j+2] = correct / total
    print(total_loss / total, correct / total)
    j+=1
    total = 0
    correct = 0
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(X_test):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        total_loss +=loss.item()
    data_for_plotting[i, j] = total_loss / total
    data_for_plotting[i,j+2] = correct / total
    print (total_loss / total,correct / total)
    i += 1
np.save('intermediate-values', data_for_plotting)

# Actual plotting;
# if matplotlib is not available, use any tool of your choice by
# loading intermediate-values.npy
import matplotlib.pyplot as plt
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.semilogy(alpha_range, data_for_plotting[:, 0], 'b-')
ax1.semilogy(alpha_range, data_for_plotting[:, 1], 'b--')

ax2.plot(alpha_range, data_for_plotting[:, 2], 'r-')
ax2.plot(alpha_range, data_for_plotting[:, 3], 'r--')

ax1.set_xlabel('alpha')
ax1.set_ylabel('Cross Entropy', color='b')
ax2.set_ylabel('Accuracy', color='r')
ax1.legend(('Train', 'Test'), loc=0)

ax1.grid(b=True, which='both')
plt.savefig('C3ish.pdf')
print ('Saved figure; Task complete')