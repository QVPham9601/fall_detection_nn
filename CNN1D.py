from torch import nn
from torch.nn.modules import linear
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Dropout(p = 0.25),
            nn.Conv1d(9, 32, kernel_size = 8, stride = 3, padding = 3),
            nn.ReLU(),
            nn.MaxPool1d(3, 3)
        )#output size = 111

        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 128, kernel_size = 8, stride = 3, padding = 3),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )#output size = 18

        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 512, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )#output size = 5


        self.layer4 = nn.Sequential(
            nn.Linear(4*512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 15)
        )

    def forward(self, x):        
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)

        return out