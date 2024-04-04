import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)  # First hidden layer
        self.fc2 = nn.Linear(512, 512)  # Second hidden layer
        self.fc3 = nn.Linear(512, 10)  # Output layer

    def forward(self, x):
        # Ensure the input tensor is of the expected shape [batch_size, 1, 28, 28]
        if x.size()[1:] != (1, 28, 28):
            raise ValueError(f'Expected input tensor shape [batch_size, 1, 28, 28], got {x.size()}')

        x = x.view(-1, 28 * 28)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

# Set up training
# def train(model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 100 == 0:
#             print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
#
# # Load data
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])
#
# train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#
# # Model, optimizer, and device setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SimpleNN().to(device)
# optimizer = optim.Adam(model.parameters())
#
# # Training the model
# for epoch in range(1, 6):
#     train(model, device, train_loader, optimizer, epoch)
