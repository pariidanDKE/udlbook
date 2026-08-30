import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

# input size, number of layers, output size
D_i, D_k, D_o = 10, 40, 5

# dense model with two hidden layers
model = nn.Sequential(
    nn.Linear(D_i,D_k),
    nn.ReLU(),
    nn.Linear(D_k,D_k),
    nn.ReLU(),
    nn.Linear(D_k,D_o)
)

# He initialisation of parameters
def init_weights(layer_in):
    if isinstance(layer_in,nn.Linear):
        nn.init.kaiming_uniform_(layer_in.weight)
        layer_in.bias.data.fill_(0.0)

model.apply(init_weights)

# least squares loss, where we average losses by default and not sum as Prince has it
criterion = nn.MSELoss()

# construct SGD optimzer, tied to model parameters
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum= 0.9)

# every 10 steps half the LR
lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# create 100 random data points
x = torch.randn(100,D_i)
y = torch.randn(100,D_o)

data_loader = DataLoader(TensorDataset(x,y), batch_size=10, shuffle=True)


for epoch in range(100):
    epoch_loss = 0.0

    # loop over batches
    for i, data in enumerate(data_loader):

        # retrieve bathc inputs and labels
        x_batch, y_batch = data

        # make sure parameter gradients are wiped from last iteration
        optimizer.zero_grad()

        # forward pass
        prediction = model(x_batch)
        loss = criterion(prediction, y_batch)

        # backward pass -> full backprop
        loss.backward()

        # SGD update : update params with gradient computed in backward
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch} loss: {epoch_loss:3f}")

    # log another step in the LR scheduelr
    lr_scheduler.step()













