import numpy as np
import torch.nn as nn
import torch
from torch import nn, optim
import torch.nn.functional as F

# The Game:
#   Input is in {1,2,3,4}.
#   Network is run on a one-hot encoding, WITHOUT using torch.functional.one_hot
#   It needs to predict the input's square


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


def f(s):
    s_cp = s.item()
    if s_cp == 1:
        input = torch.tensor([1.0, 0, 0, 0]).float()
    elif s_cp == 2:
        input = torch.tensor([0, 1, 0, 0]).float()
    elif s_cp == 3:
        input = torch.tensor([0, 0, 1, 0]).float()
    elif s_cp == 4:
        input = torch.tensor([0, 0, 0, 1]).float()
    else:
        print("ERRRORRR!!!!")

    return net(input)


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, states, answers):
        loss = 0
        for i, state in enumerate(states):
            output = f(state)
            loss += (output - answers[i]) ** 2
        return loss


loss = CustomLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


dataset = []
dy = []
print("Building dataset")
for i in range(1000):
    r = np.random.random()
    if r < 0.1:
        dataset.append(1)
        dy.append(1)
    elif r < 0.2:
        dataset.append(2)
        dy.append(4)
    elif r < 0.3:
        dataset.append(3)
        dy.append(9)
    else:
        dataset.append(4)
        dy.append(16)

import matplotlib.pyplot as plt

plt.scatter(dataset, dy)
plt.show()

loss_plot = []

print("Starting training")
for i in range(500):
    if i % 10 == 0:
        print("Iteration", i)
    optimizer.zero_grad()
    loss_Now = loss(torch.tensor(dataset), dy)
    loss_Now.backward()
    loss_plot.append(loss_Now.item())
    optimizer.step()


print("Expected values:")
print(net(torch.tensor([1.0, 0.0, 0.0, 0.0])))
print(net(torch.tensor([0.0, 1.0, 0.0, 0.0])))
print(net(torch.tensor([0.0, 0.0, 1.0, 0.0])))
print(net(torch.tensor([0.0, 0.0, 0.0, 1.0])))

plt.scatter(range(1000), loss_plot)
plt.show()
