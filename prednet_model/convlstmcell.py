import torch
import math
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair

# https://gist.github.com/Kaixhin/57901e91e5c5a8bac3eb0cbbdd3aba81

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(ConvLSTMCell, self).__init__()
        # print(in_channels) 6
        # print(out_channels) 3
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        kernel_size = _pair(kernel_size)  # (3, 3)
        stride = _pair(stride)  # (1, 1)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels  # 6
        self.out_channels = out_channels    # 3
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_h = tuple(
            k // 2 for k, s, p, d in zip(kernel_size, stride, padding, dilation))
        # print(self.padding_h)   # (1, 1)
        self.dilation = dilation
        self.groups = groups
        self.weight_ih = Parameter(torch.Tensor(
            4 * out_channels, in_channels // groups, *kernel_size))  # [12, 6, 3, 3]
        self.weight_hh = Parameter(torch.Tensor(
            4 * out_channels, out_channels // groups, *kernel_size))  # [12, 3, 3, 3]
        self.weight_ch = Parameter(torch.Tensor(
            3 * out_channels, out_channels // groups, *kernel_size))    # [9, 3, 3, 3]
        if bias:
            # print(1)
            self.bias_ih = Parameter(torch.Tensor(4 * out_channels))  # 12
            self.bias_hh = Parameter(torch.Tensor(4 * out_channels))  # 12
            self.bias_ch = Parameter(torch.Tensor(3 * out_channels))  # 9
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            self.register_parameter('bias_ch', None)

        self.register_buffer('wc_blank', torch.zeros(1, 1, 1, 1)) # dont change
        self.reset_parameters()

    def reset_parameters(self):
        n = 4 * self.in_channels  # 12
        for k in self.kernel_size:
            n *= k
        # n = 12x3x3

        stdv = 1. / math.sqrt(n)
        self.weight_ih.data.uniform_(-stdv, stdv)
        self.weight_hh.data.uniform_(-stdv, stdv)
        self.weight_ch.data.uniform_(-stdv, stdv)
        if self.bias_ih is not None:
            self.bias_ih.data.uniform_(-stdv, stdv)
            self.bias_hh.data.uniform_(-stdv, stdv)
            self.bias_ch.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        # input.shape [b, 6, 128, 160]
        h_0, c_0 = hx  # b, c, h, w (b, 3, 128, 160)
        wx = F.conv2d(input, self.weight_ih, self.bias_ih,
                      self.stride, self.padding, self.dilation, self.groups)

        wh = F.conv2d(h_0, self.weight_hh, self.bias_hh, self.stride,
                      self.padding_h, self.dilation, self.groups)

        # Cell uses a Hadamard product instead of a convolution?
        wc = F.conv2d(c_0, self.weight_ch, self.bias_ch, self.stride,
                      self.padding_h, self.dilation, self.groups)

        # print(wx.shape) [b, 12, 128, 160]
        # print(wh.shape) [b, 12, 128, 160]
        # print(wc.shape) [b, 9, 128, 160]

        a =torch.cat((wc[:, :2 * self.out_channels], Variable(self.wc_blank).expand(
            wc.size(0), wc.size(1) // 3, wc.size(2), wc.size(3)), wc[:, 2 * self.out_channels:]), 1)

        # print(wc.shape)
        # print(Variable(self.wc_blank).expand(
        #     wc.size(0), wc.size(1) // 3, wc.size(2), wc.size(3)).shape)

        wxhc = wx + wh + torch.cat((wc[:, :2 * self.out_channels], Variable(self.wc_blank).expand(
            wc.size(0), wc.size(1) // 3, wc.size(2), wc.size(3)), wc[:, 2 * self.out_channels:]), 1)

        # print(wxhc.shape) b, 12, 128, 160

        i = F.sigmoid(wxhc[:, :self.out_channels])
        f = F.sigmoid(wxhc[:, self.out_channels:2 * self.out_channels])
        g = F.tanh(wxhc[:, 2 * self.out_channels:3 * self.out_channels])
        o = F.sigmoid(wxhc[:, 3 * self.out_channels:])

        # print(i.shape) b, 3, 128, 160
        # print(f.shape) b, 3, 128, 160
        # print(g.shape) b, 3, 128, 160
        # print(o.shape) b, 3, 128, 160

        c_1 = f * c_0 + i * g  # b, 3, 128, 160
        h_1 = o * F.tanh(c_1)  # b, 3, 128, 160


        # print(c_1.shape)
        # print(h_1.shape)

        return h_1, (h_1, c_1)
