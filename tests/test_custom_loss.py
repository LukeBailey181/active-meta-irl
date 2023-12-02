import numpy as np
import torch.nn as nn
import torch
from torch import nn, optim
import torch.nn.functional as F


# Define the network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


def Q(s):
    denom = (
        torch.exp(net(torch.tensor([1.0, 0.0, 0.0, 0.0])))
        + torch.exp(net(torch.tensor([0.0, 1.0, 0.0, 0.0])))
        + torch.exp(net(torch.tensor([0.0, 0.0, 1.0, 0.0])))
        + torch.exp(net(torch.tensor([0.0, 0.0, 0.0, 1.0])))
    )
    return torch.exp(net(s)) / denom


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, states):
        loss = 0
        for state in states:
            loss += torch.log(Q(state))
        return -loss


loss = CustomLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


dataset = []
print("Building dataset")
for i in range(1000):
    r = np.random.random()
    if r < 0.1:
        dataset.append([1.0, 0.0, 0.0, 0.0])
    elif r < 0.2:
        dataset.append([0.0, 1.0, 0.0, 0.0])
    elif r < 0.3:
        dataset.append([0.0, 0.0, 1.0, 0.0])
    else:
        dataset.append([0.0, 0.0, 0.0, 1.0])

print("Starting training")
for i in range(100):
    if i % 10 == 0:
        print("Iteration", i)
    optimizer.zero_grad()
    loss_Now = loss(torch.tensor(dataset))
    loss_Now.backward()
    optimizer.step()

print("Expected probabilities:")
print(net(torch.tensor([1.0, 0.0, 0.0, 0.0])))
print(net(torch.tensor([0.0, 1.0, 0.0, 0.0])))
print(net(torch.tensor([0.0, 0.0, 1.0, 0.0])))
print(net(torch.tensor([0.0, 0.0, 0.0, 1.0])))

print("Q values:")
print(Q(torch.tensor([1.0, 0.0, 0.0, 0.0])).item())
print(Q(torch.tensor([0.0, 1.0, 0.0, 0.0])).item())
print(Q(torch.tensor([0.0, 0.0, 1.0, 0.0])).item())
print(Q(torch.tensor([0.0, 0.0, 0.0, 1.0])).item())
