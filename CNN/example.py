import torch
from torch import nn
from torchvision import datasets, transforms

# from torch.utils.data import DataLoader, Dataset

torch.manual_seed(0)
torch.cuda.manual_seed(0)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(  # 16*14*14
            nn.Conv2d(16, 32, 5, 1, 2),  # 32*14*14
            nn.ReLU(),
            nn.MaxPool2d(2)  # 32*7*7
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # x (b*1*32*32)
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(batch_size, -1)
        out = self.out(x)
        return out
