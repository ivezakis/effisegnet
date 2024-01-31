import math

import torch
import torch.nn as nn
from monai.networks.nets import EfficientNetBNFeatures
from monai.networks.nets.efficientnet import get_efficientnet_image_size


class GhostModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        ratio=2,
        dw_size=3,
        stride=1,
        relu=True,
    ):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                init_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                bias=False,
            ),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
            nn.BatchNorm2d(init_channels),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=False,
            ),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
            nn.BatchNorm2d(new_channels),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.out_channels, :, :]


class EffiSegNetBN(nn.Module):
    def __init__(
        self,
        ch=64,
        pretrained=True,
        freeze_encoder=False,
        deep_supervision=False,
        model_name="efficientnet-b0",
    ):
        super(EffiSegNetBN, self).__init__()
        self.model_name = model_name
        self.encoder = EfficientNetBNFeatures(
            model_name=model_name,
            pretrained=pretrained,
        )

        # remove unused layers
        del self.encoder._avg_pooling
        del self.encoder._dropout
        del self.encoder._fc

        # extract the last number from the model name, example: efficientnet-b0 -> 0
        b = int(model_name[-1])

        num_channels_per_output = [
            (16, 24, 40, 112, 320),
            (16, 24, 40, 112, 320),
            (16, 24, 48, 120, 352),
            (24, 32, 48, 136, 384),
            (24, 32, 56, 160, 448),
            (24, 40, 64, 176, 512),
            (32, 40, 72, 200, 576),
            (32, 48, 80, 224, 640),
            (32, 56, 88, 248, 704),
            (72, 104, 176, 480, 1376),
        ]

        channels_per_output = num_channels_per_output[b]

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.deep_supervision = deep_supervision

        upsampled_size = get_efficientnet_image_size(model_name)
        self.up1 = nn.Upsample(size=upsampled_size, mode="nearest")
        self.up2 = nn.Upsample(size=upsampled_size, mode="nearest")
        self.up3 = nn.Upsample(size=upsampled_size, mode="nearest")
        self.up4 = nn.Upsample(size=upsampled_size, mode="nearest")
        self.up5 = nn.Upsample(size=upsampled_size, mode="nearest")

        self.conv1 = nn.Conv2d(
            channels_per_output[0], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(ch)

        self.conv2 = nn.Conv2d(
            channels_per_output[1], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(ch)

        self.conv3 = nn.Conv2d(
            channels_per_output[2], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(ch)

        self.conv4 = nn.Conv2d(
            channels_per_output[3], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(ch)

        self.conv5 = nn.Conv2d(
            channels_per_output[4], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn5 = nn.BatchNorm2d(ch)

        self.relu = nn.ReLU(inplace=True)

        if self.deep_supervision:
            self.conv7 = nn.Conv2d(
                ch, 1, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn7 = nn.BatchNorm2d(ch)
            self.conv8 = nn.Conv2d(
                ch, 1, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn8 = nn.BatchNorm2d(ch)
            self.conv9 = nn.Conv2d(
                ch, 1, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn9 = nn.BatchNorm2d(ch)
            self.conv10 = nn.Conv2d(
                ch, 1, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn10 = nn.BatchNorm2d(ch)
            self.conv11 = nn.Conv2d(
                ch, 1, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn11 = nn.BatchNorm2d(ch)

        self.bn6 = nn.BatchNorm2d(ch)
        self.ghost1 = GhostModule(ch, ch)
        self.ghost2 = GhostModule(ch, ch)

        self.conv6 = nn.Conv2d(ch, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)

        x0 = self.conv1(x0)
        x0 = self.relu(x0)
        x0 = self.bn1(x0)

        x1 = self.conv2(x1)
        x1 = self.relu(x1)
        x1 = self.bn2(x1)

        x2 = self.conv3(x2)
        x2 = self.relu(x2)
        x2 = self.bn3(x2)

        x3 = self.conv4(x3)
        x3 = self.relu(x3)
        x3 = self.bn4(x3)

        x4 = self.conv5(x4)
        x4 = self.relu(x4)
        x4 = self.bn5(x4)

        x0 = self.up1(x0)
        x1 = self.up2(x1)
        x2 = self.up3(x2)
        x3 = self.up4(x3)
        x4 = self.up5(x4)

        x = x0 + x1 + x2 + x3 + x4
        x = self.bn6(x)
        x = self.ghost1(x)
        x = self.ghost2(x)
        x = self.conv6(x)

        if self.deep_supervision:
            x0 = self.bn7(x0)
            x0 = self.conv7(x0)

            x1 = self.bn8(x1)
            x1 = self.conv8(x1)

            x2 = self.bn9(x2)
            x2 = self.conv9(x2)

            x3 = self.bn10(x3)
            x3 = self.conv10(x3)

            x4 = self.bn11(x4)
            x4 = self.conv11(x4)

            return x, [x0, x1, x2, x3, x4]

        return x
