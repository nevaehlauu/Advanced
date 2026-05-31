import torch
import torch.nn as nn

# ResNet18
class ResidualBlock_1d(nn.Module):
    def __init__(self, In_channels, Out_channels, downsample=False):
        super(ResidualBlock_1d, self).__init__()
        self.stride = 1
        # if downsample == True:
        #     self.stride = 2
        self.layer = nn.Sequential(
            nn.Conv1d(In_channels, Out_channels, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm1d(Out_channels),
            nn.ReLU(),

            nn.Conv1d(Out_channels, Out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(Out_channels),
            nn.ReLU()
        )

        if In_channels != Out_channels:
            self.res_layer  = nn.Sequential(
                nn.Conv1d(In_channels, Out_channels, kernel_size=1, stride=self.stride),
                nn.BatchNorm1d(Out_channels)
            )
        else:
            self.res_layer = None
    
    def forward(self, x):
        # x = self.layer(x)
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x) + residual

class Net_1d(nn.Module):
    # def __init__(self, features_name, classes):
    def __init__(self, in_channels, classes):
        super(Net_1d, self).__init__()
        # self.features_name = features_name
        # in_channel = len(self.features_name)
        self.Layers = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            ResidualBlock_1d(32, 32, False),
            ResidualBlock_1d(32, 32, False),

            ResidualBlock_1d(32, 64, False),
            ResidualBlock_1d(64, 64, False),

            ResidualBlock_1d(64, 128, False),
            ResidualBlock_1d(128, 128, False),

            ResidualBlock_1d(128, 256, False),
            ResidualBlock_1d(256, 256, False),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(256, classes)
    
    def forward(self, x):
        x0 = self.Layers(x)
        x = x0.view(x0.size(0), -1)
        x = self.fc(x)
        return x0, x


class ResidualBlock(nn.Module):
    def __init__(self, In_channels, Out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        self.stride = 1
        # if downsample == True:
        #     self.stride = 2
        self.layer = nn.Sequential(
            nn.Conv2d(In_channels, Out_channels, kernel_size=(3, 1), stride=self.stride, padding='same'),
            nn.BatchNorm2d(Out_channels),
            nn.ReLU(),

            nn.Conv2d(Out_channels, Out_channels, kernel_size=(3, 1), stride=1, padding='same'),
            nn.BatchNorm2d(Out_channels),
            nn.ReLU()
        )

        if In_channels != Out_channels:
            self.res_layer  = nn.Sequential(
                nn.Conv2d(In_channels, Out_channels, kernel_size=(1, 1), stride=self.stride),
                nn.BatchNorm2d(Out_channels)
            )
        else:
            self.res_layer = None
    
    def forward(self, x):
        # x = self.layer(x)
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x) + residual


class Net(nn.Module):
    def __init__(self, in_channels, classes):
        super(Net, self).__init__()
        self.Layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(7, 1), stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            ResidualBlock(32, 32, False),
            ResidualBlock(32, 32, False),

            ResidualBlock(32, 64, False),
            ResidualBlock(64, 64, False),

            ResidualBlock(64, 128, False),
            ResidualBlock(128, 128, False),

            ResidualBlock(128, 256, False),
            ResidualBlock(256, 256, False),

            nn.AdaptiveAvgPool2d(1)
        )

        self.cls = nn.Linear(256, classes)
    
    def forward(self, x):
        x = self.Layers(x)
        x = x.view(x.size(0), -1)
        x = self.cls(x)
        return x