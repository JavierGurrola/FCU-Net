import torch
import torch.nn as nn


class UpsampleBlock(nn.Module):
    r"""
    Upsample module of the FCUnet.
    Args:
        in_channels (int): Number of channels that the module expects to receive from the image to upsample.
        cat_channels (int): Number of channels that the module expects to receive from the image to concatenate.
        out_channels (int): Number of channels that the module generates..
    """
    def __init__(self, in_channels, cat_channels, out_channels):
        super(UpsampleBlock, self).__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + cat_channels, out_channels, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.01),
            nn.LeakyReLU()
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, (2, 2), stride=(2, 2)),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        upsample, concat = x
        upsample = self.upsample(upsample)
        return self.bottleneck(torch.cat([concat, upsample], 1))


class FactorizedBlock(nn.Module):
    r"""
    Factorized convolutional block
    Args:
        in_channels (int): Number of channels that the module expects to receive.
        inner_channels (int): Number of channels generated in the factorized convolution.
    """
    def __init__(self, in_channels, inner_channels):
        super(FactorizedBlock, self).__init__()

        self.conv_0 = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(inner_channels, momentum=0.01),
            nn.LeakyReLU()
        )

        self.conv_1_3 = nn.Conv2d(inner_channels, inner_channels, (1, 3), padding=(0, 1), bias=False)
        self.conv_3_1 = nn.Conv2d(inner_channels, inner_channels, (3, 1), padding=(1, 0), bias=False)

        self.norm = nn.Sequential(
            nn.BatchNorm2d(2 * inner_channels, momentum=0.01),
            nn.LeakyReLU()
        )

    def forward(self, x):
        out = self.conv_0(x)
        return self.norm(torch.cat([self.conv_1_3(out), self.conv_3_1(out)], 1))


class FCUnet(nn.Module):
    r"""
    Factorized U-net for Retinal Vessel Segmentation.
    """
    def __init__(self, **kwargs):
        super().__init__()
        dropout_rate = kwargs['dropout rate']

        filters_0 = kwargs['base filters']
        filters_1 = 2 * filters_0
        filters_2 = 4 * filters_0
        filters_3 = 8 * filters_0

        # Encoder:
        # Level 0
        self.block_0_0 = nn.Sequential(
            nn.Conv2d(3, filters_0, (3, 3), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_rate, inplace=True),
            FactorizedBlock(filters_0, filters_0 // 2)
        )    

        # Level 1:
        self.block_1_0 = nn.Sequential(
            nn.Conv2d(filters_0, filters_1, kernel_size=(2, 2), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_rate, inplace=True),
            FactorizedBlock(filters_1, filters_1 // 2)
        )

        # Level 2:
        self.block_2_0 = nn.Sequential(
            nn.Conv2d(filters_1, filters_2, kernel_size=(2, 2), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_rate, inplace=True),
            FactorizedBlock(filters_2, filters_2 // 2)
        )

        # Level 3 (Bottleneck)
        self.block_3_0 = nn.Sequential(
            nn.Conv2d(filters_2, filters_3, kernel_size=(2, 2), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_rate, inplace=True),
            FactorizedBlock(filters_3, filters_3 // 2)
        )

        # Decoder
        # Level 2:
        self.block_2_1 = nn.Sequential(
            UpsampleBlock(filters_3, filters_2, filters_2),
            nn.Dropout2d(dropout_rate, inplace=True),
            FactorizedBlock(filters_2, filters_2 // 2)
        )

        # Level 1:
        self.block_1_1 = nn.Sequential(
            UpsampleBlock(filters_2, filters_1, filters_1),
            nn.Dropout2d(dropout_rate, inplace=True),
            FactorizedBlock(filters_1, filters_1 // 2)
        )

        # Level 0:
        self.block_0_1 = nn.Sequential(
            UpsampleBlock(filters_1, filters_0, filters_0),
            nn.Dropout2d(dropout_rate, inplace=True),
            FactorizedBlock(filters_0, filters_0 // 2),
            nn.Conv2d(filters_0, 1, (3, 3), padding=(1, 1))
        )

    def forward(self, inputs):
        out_0 = self.block_0_0(inputs)          # Level 0
        out_1 = self.block_1_0(out_0)           # Level 1
        out_2 = self.block_2_0(out_1)           # Level 2
        out_3 = self.block_3_0(out_2)           # Level 3 (Bottleneck)

        out_2 = self.block_2_1([out_3, out_2])  # Level 2
        out_1 = self.block_1_1([out_2, out_1])  # Level 1
        return self.block_0_1([out_1, out_0])   # Level 0
