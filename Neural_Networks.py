from torch import nn
import torch
from torch.nn import functional


class Neural_Network(nn.Module):
    def __init__(self):
        super(Neural_Network, self).__init__()

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
            #nn.ReLU(),
            #nn.Linear(128, 15),
            #nn.LogSoftmax(0)
        )

        self.numeric_features_ = nn.Sequential(
            nn.Linear(12,64),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.25),
            nn.Linear(64,128),
            nn.ReLU(inplace=True)
        )

        self.combined_features_ = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(256,15)
        )


    def forward(self, x, y):     
        #x = self.quant(x)
         
        out_ts = self.layer1(x)
        out_ts = self.layer2(out_ts)
        out_ts = self.layer3(out_ts)
        out_ts = self.layer4(out_ts)
        out_ts = self.layer5(out_ts)
        out_ts = out_ts.view(out_ts.size(0), -1)
        out_ts = self.layer6(out_ts)

        out_num = self.numeric_features_(y)

        out = torch.cat((out_ts,out_num) , dim =1)

        out = self.combined_features_(out)

        #out = self.dequant(out)


        return out
    






    '''
    class NeuralNetwork(nn.Module):
    def __init__(self):
        super (NeuralNetwork, self).__init__()
        self.image_features_ = nn.Sequential(
            nn.Conv2d(3,16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv2d(16,64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),
        )
        self.numeric_features_ = nn.Sequential(
            nn.Linear(9,64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64,64*64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.combined_features_ = nn.Sequential(
            nn.Linear(64*64*2, 64*64*2*2),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64*64*2*2, 64*64*2),
            nn.ReLU(inplace=True),
            nn.Linear(64*3*3*2, 64),
            nn.Linear(64,5),
        )

    def forward(self, x,y):
        x = self.image_features_(x)
        x = x.view(-1,64*64)
        y = self.numeric_features_(y)
        z = torch.cat((x,y),1)
        z = self.combined_features_(z)
        return z

    for epoch in range(1,n_epochs+1):
    loss_train = 0.0
    for imgs, numeric_features, price in train_loader:
        imgs = imgs.to(device)
        numeric_features = numeric_features.to(device)
        price = price.to(device)
        output = model(imgs, numeric_features)

        loss = loss_fn(output,price)

        #L2 regularization
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2).sum() for p in model.parameters())
        loss = loss + l2_lambda*l2_norm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_train += loss.item()
    '''