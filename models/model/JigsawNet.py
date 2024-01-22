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

        self.final_conv = nn.Conv2d(in_channels, 16, kernel_size=(28, 28), stride=(28, 28))
        
    def forward(self, x):

        # Contraction path
        conv1 = self.conv1(x)
        residual1 = self.pool(conv1)
        conv2 = self.conv2(residual1)
        residual2 = self.pool(conv2)
        conv3 = self.conv3(residual2)
        residual3 = self.pool(conv3)
        conv4 = self.conv4(residual3)
        residual4 = self.pool(conv4)
        conv5 = self.conv5(residual4)

        # Expansion path
        convUp1 = self.convUp1(conv5, conv4)
        convUp2 = self.convUp2(convUp1, conv3)
        convUp3 = self.convUp3(convUp2, conv2)
        convUp4 = self.convUp4(convUp3, conv1)

        lastPool = self.pool(convUp4)
        # [B, 4, 4, 16]
        out = self.final_conv(lastPool)
        out = nn.BatchNorm2d(16)(out)
        out = out.view(out.size(0), out.size(1), -1)
        out = F.softmax(out, dim=1)

        return out
