import torch
import torch.nn as nn


class Conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, size):
        super(Conv_block, self).__init__()
        self.padding = [(size[0] - 1) // 2, (size[1] - 1) // 2]
        self.conv = nn.Conv2d(in_ch, out_ch, size, padding=self.padding, stride=1)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SP(nn.Module):
    def __init__(self, in_ch):
        super(SP, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_ch // 2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch // 2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        xx = self.conv1(x)
        xx = self.conv2(xx)
        return xx


class HDD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HDD, self).__init__()

        self.in_ch = in_ch
        self.mid_mid = out_ch // 7
        self.out_ch = out_ch
        self.conv1x1_mid = Conv_block(self.in_ch, self.out_ch, [1, 1])
        self.conv1x1_2 = nn.Conv2d(self.out_ch, self.out_ch, 1)
        self.conv3x3_3_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x3_2_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x3_1_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x3_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x1_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv1x3_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])

        self.conv3x3_3_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x3_1_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x3_2_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x3_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x1_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv1x3_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        # self.conv1x1_2 = Conv_block(self.mid_mid, self.mid_mid, [1, 1])
        self.conv1x1_1 = nn.Conv2d(self.out_ch, self.out_ch, 1)
        self.rel = nn.ReLU(inplace=True)
        if self.in_ch > self.out_ch:
            self.short_connect = nn.Conv2d(in_ch, out_ch, 1, padding=0)

    def forward(self, x):
        xxx = self.conv1x1_mid(x)
        x0 = xxx[:, 0:self.mid_mid, ...]
        x1 = xxx[:, self.mid_mid:self.mid_mid * 2, ...]
        x2 = xxx[:, self.mid_mid * 2:self.mid_mid * 3, ...]
        x3 = xxx[:, self.mid_mid * 3:self.mid_mid * 4, ...]
        x4 = xxx[:, self.mid_mid * 4:self.mid_mid * 5, ...]
        x5 = xxx[:, self.mid_mid * 5:self.mid_mid * 6, ...]
        x6 = xxx[:, self.mid_mid * 6:self.mid_mid * 7, ...]
        x1 = self.conv1x3_1(x1)
        x2 = self.conv3x1_1(x2 + x1)
        x3 = self.conv3x3_1(x3 + x2)
        x4 = self.conv3x3_1_1(x4 + x3)
        x5 = self.conv3x3_2_1(x5 + x4)
        x6 = self.conv3x3_3_1(x5 + x6)
        xxx = self.conv1x1_1(torch.cat((x0, x1, x2, x3, x4, x5, x6), dim=1))
        x0 = xxx[:, 0:self.mid_mid, ...]
        x1_2 = xxx[:, self.mid_mid:self.mid_mid * 2, ...]
        x2_2 = xxx[:, self.mid_mid * 2:self.mid_mid * 3, ...]
        x3_2 = xxx[:, self.mid_mid * 3:self.mid_mid * 4, ...]
        x4_2 = xxx[:, self.mid_mid * 4:self.mid_mid * 5, ...]
        x5_2 = xxx[:, self.mid_mid * 5:self.mid_mid * 6, ...]
        x6_2 = xxx[:, self.mid_mid * 6:self.mid_mid * 7, ...]
        x1 = self.conv1x3_2(x1_2)
        x2 = self.conv3x1_2(x1 + x2_2)
        x3 = self.conv3x3_2(x2 + x3_2)
        x4 = self.conv3x3_1_2(x3 + x4_2)
        x5 = self.conv3x3_2_2(x4 + x5_2)
        x6 = self.conv3x3_3_2(x5 + x6_2)
        xx = torch.cat((x0, x1, x2, x3, x4, x5, x6), dim=1)
        xx = self.conv1x1_2(xx)
        if self.in_ch > self.out_ch:
            x = self.short_connect(x)
        return self.rel(xx + x + xxx)


class HDD_first(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HDD_first, self).__init__()

        self.in_ch = in_ch
        self.mid_mid = out_ch // 7
        self.out_ch = out_ch
        self.conv1x1_mid = Conv_block(self.in_ch, self.out_ch, [1, 1])
        self.conv1x1_2 = nn.Conv2d(self.out_ch, self.out_ch, 1)
        self.conv3x3_3_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x3_2_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x3_1_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x3_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x1_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv1x3_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])

        self.conv3x3_3_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x3_1_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x3_2_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x3_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x1_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv1x3_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        # self.conv1x1_2 = Conv_block(self.mid_mid, self.mid_mid, [1, 1])
        self.conv1x1_1 = nn.Conv2d(self.out_ch, self.out_ch, 1)
        self.rel = nn.ReLU(inplace=True)
        if self.in_ch > self.out_ch:
            self.short_connect = nn.Conv2d(in_ch, out_ch, 1, padding=0)

    def forward(self, x):
        xxx = self.conv1x1_mid(x)
        x0 = xxx[:, 0:self.mid_mid, ...]
        x1 = xxx[:, self.mid_mid:self.mid_mid * 2, ...]
        x2 = xxx[:, self.mid_mid * 2:self.mid_mid * 3, ...]
        x3 = xxx[:, self.mid_mid * 3:self.mid_mid * 4, ...]
        x4 = xxx[:, self.mid_mid * 4:self.mid_mid * 5, ...]
        x5 = xxx[:, self.mid_mid * 5:self.mid_mid * 6, ...]
        x6 = xxx[:, self.mid_mid * 6:self.mid_mid * 7, ...]
        x1 = self.conv1x3_1(x1)
        x2 = self.conv3x1_1(x2 + x1)
        x3 = self.conv3x3_1(x3 + x2)
        x4 = self.conv3x3_1_1(x4 + x3)
        x5 = self.conv3x3_2_1(x5 + x4)
        x6 = self.conv3x3_3_1(x5 + x6)
        xxx = self.conv1x1_1(torch.cat((x0, x1, x2, x3, x4, x5, x6), dim=1))
        x0 = xxx[:, 0:self.mid_mid, ...]
        x1_2 = xxx[:, self.mid_mid:self.mid_mid * 2, ...]
        x2_2 = xxx[:, self.mid_mid * 2:self.mid_mid * 3, ...]
        x3_2 = xxx[:, self.mid_mid * 3:self.mid_mid * 4, ...]
        x4_2 = xxx[:, self.mid_mid * 4:self.mid_mid * 5, ...]
        x5_2 = xxx[:, self.mid_mid * 5:self.mid_mid * 6, ...]
        x6_2 = xxx[:, self.mid_mid * 6:self.mid_mid * 7, ...]
        x1 = self.conv1x3_2(x1_2)
        x2 = self.conv3x1_2(x1 + x2_2)
        x3 = self.conv3x3_2(x2 + x3_2)
        x4 = self.conv3x3_1_2(x3 + x4_2)
        x5 = self.conv3x3_2_2(x4 + x5_2)
        x6 = self.conv3x3_3_2(x5 + x6_2)
        xx = torch.cat((x0, x1, x2, x3, x4, x5, x6), dim=1)
        xx = self.conv1x1_2(xx)
        if self.in_ch > self.out_ch:
            x = self.short_connect(x)
        return self.rel(xx + xxx), self.rel(xx + x + xxx)


class Conv_down_2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_down_2, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=True)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Conv_down(nn.Module):
    '''(conv => ReLU) * 2 => MaxPool2d'''

    def __init__(self, in_ch, out_ch, flage):

        super(Conv_down, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.flage = flage
        self.conv = HDD(self.in_ch, self.out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_down = Conv_down_2(self.out_ch, self.out_ch)

    def forward(self, x):

        x = self.conv(x)
        if self.flage == True:
            pool_x = torch.cat((self.pool(x), self.conv_down(x)), dim=1)
        else:
            pool_x = None
        return pool_x, x


class Conv_down_first(nn.Module):
    '''(conv => ReLU) * 2 => MaxPool2d'''

    def __init__(self, in_ch, out_ch, flage):
        super(Conv_down_first, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.flage = flage
        self.conv = HDD_first(self.in_ch, self.out_ch)

    def forward(self, x):
        x, y = self.conv(x)
        return x, y


class Conv_down2(nn.Module):
    '''(conv => ReLU) * 2 => MaxPool2d'''

    def __init__(self, in_ch, out_ch, flage):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Conv_down2, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.flage = flage

        self.conv1 = nn.Sequential(
            Conv_block(1, self.out_ch, [3, 3]),
            # Conv_block(self.out_ch, self.out_ch, [3, 3]),
        )
        self.conv2 = nn.Sequential(
            Conv_block(1, self.out_ch, [3, 3]),
            # Conv_block(self.out_ch, self.out_ch, [3, 3]),
        )
        self.conv3 = nn.Sequential(
            Conv_block(1, self.out_ch, [3, 3]),
            # Conv_block(self.out_ch, self.out_ch, [3, 3]),
        )
        self.conv4 = nn.Sequential(
            Conv_block(1, self.out_ch, [3, 3]),
            # Conv_block(self.out_ch, self.out_ch, [3, 3]),
        )
        self.conv5 = nn.Sequential(
            Conv_block(1, self.out_ch, [3, 3]),
            # Conv_block(self.out_ch, self.out_ch, [3, 3]),
        )

        # self.sp4 = SP(self.out_ch)
        # self.sp2 = SP(self.out_ch)
        # self.sp3 = SP(self.out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_down1 = Conv_down_2(self.out_ch, self.out_ch)
        self.conv_down2 = Conv_down_2(self.out_ch, self.out_ch)
        self.conv_down3 = Conv_down_2(self.out_ch, self.out_ch)
        self.conv_down4 = Conv_down_2(self.out_ch, self.out_ch)
        self.conv_down5 = Conv_down_2(self.out_ch, self.out_ch)

    def forward(self, x):
        x1 = x[:, 0, :, :].unsqueeze(1)
        x2 = x[:, 1, :, :].unsqueeze(1)
        x3 = x[:, 2, :, :].unsqueeze(1)
        x4 = x[:, 3, :, :].unsqueeze(1)
        x5 = x[:, 4, :, :].unsqueeze(1)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        x5 = self.conv5(x5)

        x_1 = self.conv_down1(x1)
        x_2 = self.conv_down2(x2)
        x_3 = self.conv_down3(x3)
        x_4 = self.conv_down4(x4)
        x_5 = self.conv_down5(x5)

        return torch.cat(((x_1), (x_2), (x_3)), dim=1), \
               torch.cat(((x_2), (x_3), (x_4)), dim=1), \
               torch.cat(((x_3), (x_4), (x_5)), dim=1)


class Conv_up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_up, self).__init__()
        self.up = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, stride=1)
        self.conv = HDD(in_ch, out_ch)
        self.interp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x1, x2):
        x1 = self.interp(x1)
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv(x1)
        return x1


class SE(nn.Module):
    def __init__(self, in_ch, r=2):
        super(SE, self).__init__()
        self.pool_1 = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // r, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_ch // r, in_ch, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.pool_1(x)
        x1 = self.fc1(x1)
        x1 = x * x1
        return x1


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CsNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CsNet, self).__init__()
        num_filters = 70
        self.filter = num_filters
        if not isinstance(in_channels, list):
            in_channels = [in_channels]
        if not isinstance(out_channels, list):
            out_channels = [out_channels]
        self.in_channels = in_channels
        self.out_channels = out_channels

        for i, (n_chan, n_class) in enumerate(zip(in_channels, out_channels)):
            setattr(self, 'in{i}'.format(i=i), OutConv(n_chan, self.filter))
            setattr(self, 'out{i}'.format(i=i), OutConv(self.filter, n_class))
        self.down = Conv_down_2(self.filter, self.filter)
        self.Conv_down2 = Conv_down(self.filter, self.filter, True)
        self.Conv_down3 = Conv_down(self.filter * 2, self.filter * 2, True)
        self.Conv_down4 = Conv_down(self.filter * 4, self.filter * 4, True)
        self.Conv_down5 = Conv_down(self.filter * 8, self.filter * 8, False)

        self.Conv_up1_2 = Conv_up(self.filter * 8, self.filter * 4)
        self.Conv_up1_3 = Conv_up(self.filter * 4, self.filter * 2)
        self.Conv_up1_4 = Conv_up(self.filter * 2, self.filter)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, task_idx=0):
        x = getattr(self, 'in{}'.format(task_idx))(x)
        x = self.down(x)
        x, conv2 = self.Conv_down2(x)
        x, conv3 = self.Conv_down3(x)
        # _, x = self.Conv_down4(x)
        x, conv4 = self.Conv_down4(x)
        _, x = self.Conv_down5(x)

        x = self.Conv_up1_2(x, conv4)
        x = self.Conv_up1_3(x, conv3)
        x = self.Conv_up1_4(x, conv2)

        x = self.up1(x)

        logits = getattr(self, 'out{}'.format(task_idx))(x)
        return {'output': torch.sigmoid(logits)}
