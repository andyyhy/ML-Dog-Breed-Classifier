"""
EECS 445 - Introduction to Machine Learning
Winter 2022 - Project 2
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config

class Challenge(nn.Module):
    def __init__(self):
        super().__init__()

        ## TODO: define each layer of your network
        self.conv1 = torch.nn.Conv2d(3, 16, 5, stride=2, padding=2)
        self.pool = torch.nn.MaxPool2d(2, 2, padding=0)
        self.conv2_drop = nn.Dropout2d()
        self.conv2 = torch.nn.Conv2d(16, 64, 5, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(64, 8, 5, stride=2, padding=2)
        self.fc_1 = torch.nn.Linear(32, 128)
        self.fc_2 = torch.nn.Linear(128, 256)
        self.fc_3 = torch.nn.Linear(256, 2)
        ##

        self.init_weights()

    def init_weights(self):
        torch.manual_seed(42)
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)
        ## TODO: initialize the parameters for your network
        C_in = self.fc_1.weight.size(1)
        nn.init.normal_(self.fc_1.weight, 0.0, 1 / sqrt(C_in))
        nn.init.constant_(self.fc_1.bias, 0.0)
        C_in = self.fc_2.weight.size(1)
        nn.init.normal_(self.fc_2.weight, 0.0, 1 / sqrt(C_in))
        nn.init.constant_(self.fc_2.bias, 0.0)
        C_in = self.fc_3.weight.size(1)
        nn.init.normal_(self.fc_3.weight, 0.0, 1 / sqrt(C_in))
        nn.init.constant_(self.fc_3.bias, 0.0)
        ##

    def forward(self, x):
        """ You may optionally use the x.shape variables below to resize/view the size of 
            the input matrix at different points of the forward pass
        """
        N, C, H, W = x.shape

        ## TODO: forward pass
        z = self.conv1(x)
        z = F.relu(z)
        z = self.pool(z)
        z = self.conv2(z)
        z = F.relu(z)
        z = self.pool(z)
        z = self.conv3(z)
        z = F.relu(z)
        z = z.reshape(-1, 32)
        z = self.fc_1(z)
        z = F.relu(z)
        z = self.fc_2(z)
        z = F.relu(z)
        z = self.conv2_drop(z)
        z = self.fc_3(z)
        ##

        return z
