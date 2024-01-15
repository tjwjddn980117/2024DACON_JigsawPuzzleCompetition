import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks.blocks import ConvBlock, DeconvBlock

class JIGSAW_NET(nn.Module):
    def __init__(self, in_channels):
        '''
        This is the basic jigsaw_net. We had batch-norm. This Block don't change the size.

        Arguments:
            in_channels (int): the number of channel.
        
        Inputs:
            x (nparray): [batch_size, channel, H, W].
        '''
        super(JIGSAW_NET, self).__init__()
        self.in_channels = in_channels

        # Contraction path
        self.conv1 = ConvBlock(in_channels, 16, 'he_normal')
        self.conv2 = ConvBlock(16, 32, 'he_normal')
        self.conv3 = ConvBlock(32, 64, 'he_normal')
        self.conv4 = ConvBlock(64, 128, 'he_normal')
        self.conv5 = ConvBlock(128, 256, 'he_normal')

        self.pool = nn.MaxPool2d(2)

        # Expansion path
        self.convUp1 = DeconvBlock(256, 128, 'he_normal')
        self.convUp2 = DeconvBlock(128, 64, 'he_normal')
        self.convUp3 = DeconvBlock(64, 32, 'he_normal')
        self.convUp4 = DeconvBlock(32, 16, 'he_normal')


    def forward(self, x):
        conv1 = self.conv1(x)
        residual1 = self.pool(conv1)

        conv2 = self.conv2(conv1)
        residual2 = self.pool(conv2)

        conv3 = self.conv3(conv2)
        residual3 = self.pool(conv3)

        conv4 = self.conv4(conv3)
        residual4 = self.pool(conv4)