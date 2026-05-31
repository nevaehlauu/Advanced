import torch.nn as nn
import torch

class Bottlrneck(nn.Module):
    def __init__(self,In_channel,Med_channel,Out_channel,downsample=False):
        super(Bottlrneck, self).__init__()
        self.stride = 1
        # if downsample == True:
        #     self.stride = 2

        self.layer = torch.nn.Sequential(
            nn.Conv1d(In_channel, Med_channel, 1, self.stride),
            nn.BatchNorm1d(Med_channel),
            nn.ReLU(),
            nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            nn.BatchNorm1d(Med_channel),
            nn.ReLU(),
            nn.Conv1d(Med_channel, Out_channel, 1),
            nn.BatchNorm1d(Out_channel),
            nn.ReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = nn.Conv1d(In_channel, Out_channel, 1, self.stride)
        else:
            self.res_layer = None

    def forward(self, x):
        if self.res_layer is not None: 
            residual = self.res_layer(x)
        else: 
            residual = x
        return self.layer(x) + residual

##############################
# ResNet50

# class Net(nn.Module):
#     def __init__(self, features_name, classes):
#         super(Net, self).__init__()
#         self.features_name = features_name
#         in_channel = len(self.features_name)
#         self.Layers = nn.Sequential(
#             nn.Conv1d(in_channel, 64, kernel_size=7, stride=1, padding=3),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),

#             Bottlrneck(64, 64, 256, False),
#             Bottlrneck(256,64,256,False),
#             Bottlrneck(256,64,256,False),
#             #
#             Bottlrneck(256,128,512, False),
#             Bottlrneck(512,128,512, False),
#             Bottlrneck(512,128,512, False),
#             Bottlrneck(512,128,512, False),
#             #
#             Bottlrneck(512,256,1024, False),
#             Bottlrneck(1024,256,1024, False),
#             Bottlrneck(1024,256,1024, False),
#             Bottlrneck(1024,256,1024, False),
#             Bottlrneck(1024,256,1024, False),
#             Bottlrneck(1024,256,1024, False),
#             #
#             Bottlrneck(1024,512,2048, False),
#             Bottlrneck(2048,512,2048, False),
#             Bottlrneck(2048,512,2048, False),

#             nn.AdaptiveAvgPool1d(1) 
#         )
#         self.cls = nn.Sequential(
#             # nn.Linear(2048,2048),
#             # nn.ReLU(inplace=True),
#             # nn.Dropout(),

#             nn.Linear(2048, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),

#             nn.Linear(512, classes)
#             )   

#     # def change_logging(self, x):
#     #     # x维度：batchsize, in_channel, sequence_len
#     #     logging_token = []
#     #     for key in self.features_name:
#     #         logging_token.append(x[key].transpose(1, 2))
#     #     logging_token = torch.cat(tuple(logging_token), dim=1)
#     #     return logging_token

#     def forward(self, x):
#         # x = self.change_logging(x)
#         x = self.Layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.cls(x)
#         return x

############
# ResNet18
class ResidualBlock(nn.Module):
    def __init__(self, In_channels, Out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
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

class Net(nn.Module):
    def __init__(self, features_name, classes):
        super(Net, self).__init__()
        self.features_name = features_name
        in_channel = len(self.features_name)
        self.Layers = nn.Sequential(
            nn.Conv1d(in_channel, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            ResidualBlock(32, 32, False),
            ResidualBlock(32, 32, False),

            ResidualBlock(32, 64, False),
            ResidualBlock(64, 64, False),

            ResidualBlock(64, 128, False),
            ResidualBlock(128, 128, False),

            ResidualBlock(128, 256, False),
            ResidualBlock(256, 256, False),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(256, classes)

    # def change_logging(self, x):
    #     # x维度：batchsize, in_channel, sequence_len
    #     logging_token = []
    #     for key in self.features_name:
    #         logging_token.append(x[key].transpose(1, 2))
    #     logging_token = torch.cat(tuple(logging_token), dim=1)
    #     return logging_token
    
    def forward(self, x):
        # x = self.change_logging(x)
        x = self.Layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x