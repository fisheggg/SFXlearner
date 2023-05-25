"""
From https://github.com/minzwon/sota-music-tagging-models/blob/master/training/modules.py
"""
import torch
import torch.nn as nn


class Conv_1d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_1d, self).__init__()
        self.conv = nn.Conv1d(
            input_channels, output_channels, shape, stride=stride, padding=shape // 2
        )
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(pooling)

    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, output_channels, shape, stride=stride, padding=shape // 2
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class ResSE_1d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=3):
        super(ResSE_1d, self).__init__()
        # convolution
        self.conv_1 = nn.Conv1d(
            input_channels, output_channels, shape, stride=stride, padding=shape // 2
        )
        self.bn_1 = nn.BatchNorm1d(output_channels)
        self.conv_2 = nn.Conv1d(
            output_channels, output_channels, shape, padding=shape // 2
        )
        self.bn_2 = nn.BatchNorm1d(output_channels)

        # squeeze & excitation
        self.dense1 = nn.Linear(output_channels, output_channels)
        self.dense2 = nn.Linear(output_channels, output_channels)

        # residual
        self.diff = False
        if (stride != 1) or (input_channels != output_channels):
            self.conv_3 = nn.Conv1d(
                input_channels,
                output_channels,
                shape,
                stride=stride,
                padding=shape // 2,
            )
            self.bn_3 = nn.BatchNorm1d(output_channels)
            self.diff = True
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.mp = nn.MaxPool1d(pooling)

    def forward(self, x):
        # convolution
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))

        # squeeze & excitation
        se_out = nn.AvgPool1d(out.size(-1))(out)
        se_out = se_out.squeeze(-1)
        se_out = self.relu(self.dense1(se_out))
        se_out = self.sigmoid(self.dense2(se_out))
        se_out = se_out.unsqueeze(-1)
        out = torch.mul(out, se_out)

        # residual
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.mp(self.relu(out))
        return out


class SampleCNN(nn.Module):
    """
    Lee et al. 2017
    Sample-level deep convolutional neural networks for music auto-tagging using raw waveforms.
    Sample-level CNN.
    """

    def __init__(self, in_channels, n_class=13):
        super(SampleCNN, self).__init__()
        self.layer1 = Conv_1d(in_channels, 128, shape=3, stride=3, pooling=1)
        self.layer2 = Conv_1d(128, 128, shape=3, stride=1, pooling=3)
        self.layer3 = Conv_1d(128, 128, shape=3, stride=1, pooling=3)
        self.layer4 = Conv_1d(128, 256, shape=3, stride=1, pooling=3)
        self.layer5 = Conv_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer6 = Conv_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer7 = Conv_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer8 = Conv_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer9 = Conv_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer10 = Conv_1d(256, 512, shape=3, stride=1, pooling=3)
        self.layer11 = Conv_1d(512, 512, shape=1, stride=1, pooling=1)
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(512, n_class)

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.dense(x)
        x = nn.Sigmoid()(x)
        return x


class SampleCNNSE(nn.Module):
    """
    Kim et al. 2018
    Sample-level CNN architectures for music auto-tagging using raw waveforms.
    Sample-level CNN + residual connections + squeeze & excitation.
    """

    def __init__(self, in_channels, n_class=13):
        super(SampleCNNSE, self).__init__()
        self.layer1 = ResSE_1d(in_channels, 128, shape=3, stride=3, pooling=1)
        self.layer2 = ResSE_1d(128, 128, shape=3, stride=1, pooling=3)
        self.layer3 = ResSE_1d(128, 128, shape=3, stride=1, pooling=3)
        self.layer4 = ResSE_1d(128, 256, shape=3, stride=1, pooling=3)
        self.layer5 = ResSE_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer6 = ResSE_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer7 = ResSE_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer8 = ResSE_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer9 = ResSE_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer10 = ResSE_1d(256, 512, shape=3, stride=1, pooling=3)
        self.layer11 = ResSE_1d(512, 512, shape=1, stride=1, pooling=1)
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(512, 512)
        self.bn = nn.BatchNorm1d(512)
        self.dense2 = nn.Linear(512, n_class)

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = x.squeeze(-1)
        x = nn.ReLU()(self.bn(self.dense1(x)))
        x = self.dropout(x)
        x = self.dense2(x)
        x = nn.Sigmoid()(x)
        return
