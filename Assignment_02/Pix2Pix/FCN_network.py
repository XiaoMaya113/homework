import torch
import torch.nn as nn


def conv_block(in_channels, out_channels, normalize=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not normalize)]
    if normalize:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def up_block(in_channels, out_channels, dropout=0.0):
    layers = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        b = base_channels
        self.down1 = conv_block(in_channels, b, normalize=False)
        self.down2 = conv_block(b, b * 2)
        self.down3 = conv_block(b * 2, b * 4)
        self.down4 = conv_block(b * 4, b * 8)
        self.down5 = conv_block(b * 8, b * 8)
        self.down6 = conv_block(b * 8, b * 8)

        self.up1 = up_block(b * 8, b * 8, dropout=0.5)
        self.up2 = up_block(b * 16, b * 8, dropout=0.5)
        self.up3 = up_block(b * 16, b * 4)
        self.up4 = up_block(b * 8, b * 2)
        self.up5 = up_block(b * 4, b)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(b * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6)
        u2 = self.up2(torch.cat([u1, d5], dim=1))
        u3 = self.up3(torch.cat([u2, d4], dim=1))
        u4 = self.up4(torch.cat([u3, d3], dim=1))
        u5 = self.up5(torch.cat([u4, d2], dim=1))
        return self.final(torch.cat([u5, d1], dim=1))


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6, base_channels=64):
        super().__init__()
        b = base_channels
        self.net = nn.Sequential(
            conv_block(in_channels, b, normalize=False),
            conv_block(b, b * 2),
            conv_block(b * 2, b * 4),
            nn.Conv2d(b * 4, b * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(b * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(b * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, source, target):
        return self.net(torch.cat([source, target], dim=1))


class FullyConvNetwork(UNetGenerator):
    """Compatibility name for the assignment template."""

    def __init__(self):
        super().__init__(in_channels=3, out_channels=3, base_channels=64)
