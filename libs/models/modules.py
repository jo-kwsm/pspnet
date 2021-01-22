import torch
import torch.nn as nn
import torch.nn.functional as F


class conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        bias,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=bias
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu(x)

        return outputs


class conv2DBatchNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        bias,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=bias
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        outputs = self.batchnorm(x)

        return outputs


class FeatureMap_convolution(nn.Module):
    def __init__(self):
        super().__init__()
        self.cbnr_1 = conv2DBatchNormRelu(3, 64, 3, 2, 1, 1, False)
        self.cbnr_2 = conv2DBatchNormRelu(64, 64, 3, 1, 1, 1, False)
        self.cbnr_3 = conv2DBatchNormRelu(64, 128, 3, 1, 1, 1, False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.cbnr_1(x)
        x = self.cbnr_2(x)
        x = self.cbnr_3(x)
        outputs = self.maxpool(x)

        return outputs


class bottleNeckPSP(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        stride,
        dilation,
    ):
        super().__init__()
        self.cbr_1 = conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.cbr_2 = conv2DBatchNormRelu(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.cb_3 = conv2DBatchNorm(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.cb_residual = conv2DBatchNorm(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cbr_1(x)
        conv = self.cbr_2(conv)
        conv = self.cb_3(conv)
        residual = self.cb_residual(x)
        outputs = self.relu(conv + residual)

        return outputs


class bottleNeckIdentifyPSP(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        dilation,
    ):
        super().__init__()
        self.cbr_1 = conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.cbr_2 = conv2DBatchNormRelu(mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.cb_3 = conv2DBatchNorm(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cbr_1(x)
        conv = self.cbr_2(conv)
        conv = self.cb_3(conv)
        residual = x
        outputs = self.relu(conv + residual)

        return outputs


class ResidualBlockPSP(nn.Sequential):
    def __init__(
        self,
        n_blocks,
        in_channels,
        mid_channels,
        out_channels,
        stride,
        dilation,
    ):
        super().__init__()

        self.add_module(
            "block1",
            bottleNeckPSP(
                in_channels,
                mid_channels,
                out_channels,
                stride,
                dilation,
            )
        )

        for i in range(n_blocks-1):
            self.add_module(
                "block" + str(i+2),
                bottleNeckIdentifyPSP(
                    out_channels,
                    mid_channels,
                    dilation
                )
            )


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes, height, width):
        super().__init__()

        self.height = height
        self.width = width
        
        out_channels = int(in_channels / len(pool_sizes))

        self.avpool_1 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[0])
        self.cbr_1 = conv2DBatchNormRelu(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False
        )

        self.avpool_2 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[1])
        self.cbr_2 = conv2DBatchNormRelu(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False
        )

        self.avpool_3 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[2])
        self.cbr_3 = conv2DBatchNormRelu(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False
        )

        self.avpool_4 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[3])
        self.cbr_4 = conv2DBatchNormRelu(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False
        )

    def forward(self, x):
        outputs = [x]

        for i in range(1, 5):
            out = getattr(self, "avpool_" + str(i))(x)
            out = getattr(self, "cbr_" + str(i))(out)
            out = F.interpolate(
                out,
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=True
            )

            outputs.append(out)

        outputs = torch.cat(outputs, dim=1)

        return outputs


class DecodePSPFeature(nn.Module):
    def __init__(self, height, width, n_classes):
        super().__init__()

        self.height = height
        self.width = width

        self.cbr = conv2DBatchNormRelu(
            in_channels=4096,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False,
        )
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(
            in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        outputs = F.interpolate(
            x,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=True,
        )

        return outputs


class AuxiliaryPSPlayers(nn.Module):
    def __init__(self, in_channels, height, width, n_classes):
        super().__init__()

        self.height = height
        self.width = width

        self.cbr = conv2DBatchNormRelu(
            in_channels=in_channels,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False,
        )
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(
            in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        outputs = F.interpolate(
            x,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=True,
        )

        return outputs
