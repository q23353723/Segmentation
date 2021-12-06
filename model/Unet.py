import torch
from model.part import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.ETTup1 = Up(1024, 512 // factor, bilinear)
        self.ETTup2 = Up(512, 256 // factor, bilinear)
        self.ETTup3 = Up(256, 128 // factor, bilinear)
        self.ETTup4 = Up(128, 64, bilinear)
        self.ETToutc = OutConv(64, n_classes)

        self.CAup1 = Up(1024, 512 // factor, bilinear)
        self.CAup2 = Up(512, 256 // factor, bilinear)
        self.CAup3 = Up(256, 128 // factor, bilinear)
        self.CAup4 = Up(128, 64, bilinear)
        self.CAoutc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.ETTup1(x5, x4)
        x = self.ETTup2(x, x3)
        x = self.ETTup3(x, x2)
        x = self.ETTup4(x, x1)
        x_logits = self.ETToutc(x)

        y = self.CAup1(x5, x4)
        y = self.CAup2(y, x3)
        y = self.CAup3(y, x2)
        y = self.CAup4(y, x1)
        y_logits = self.CAoutc(y)
        return torch.sigmoid(x_logits), torch.sigmoid(y_logits)