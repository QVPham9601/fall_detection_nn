from torch import nn
import torch
from torch.nn import functional


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        #this is for mobile use adaptation
        #self.quant = torch.quantization.QuantStub()
        #self.dequant = torch.quantization.DeQuantStub()

        self.layer1 = nn.Sequential(
            nn.Dropout(p = 0.25),
            nn.Conv1d(12, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )#output size = 125

        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )#output size = 62

        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )#output size = 31

        self.layer4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )#output size = 15

        self.layer5 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )#output size = 7


        self.layer6 = nn.Sequential(
            nn.Linear(7*512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 15),
            #nn.LogSoftmax(0)
        )

    def forward(self, x):     
        #x = self.quant(x)
         
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = self.layer6(out)

        #out = self.dequant(out)

        return out