import torch
import torch.nn as nn
from torch.nn import functional as F
from prednet_model.convlstmcell import ConvLSTMCell
from torch.autograd import Variable


class PredNet(nn.Module):
    def __init__(self, R_channels, A_channels, output_mode='error'):
        super(PredNet, self).__init__()
        self.r_channels = R_channels + (0, )  # for convenience (3, 48, 96, 192, 0)
        self.a_channels = A_channels  # (3, 48, 96, 192)
        self.n_layers = len(R_channels)  # 3
        self.output_mode = output_mode

        default_output_modes = ['prediction', 'error']
        assert output_mode in default_output_modes, 'Invalid output_mode: ' + str(output_mode)

        for i in range(self.n_layers):
            cell = ConvLSTMCell(2 * self.a_channels[i] + self.r_channels[i+1],                                                                             self.r_channels[i],
                                (3, 3))  # 6
            setattr(self, 'cell{}'.format(i), cell)

        for i in range(self.n_layers):
            conv = nn.Sequential(nn.Conv2d(self.r_channels[i], self.a_channels[i], 3, padding=1), nn.ReLU())
            if i == 0:
                conv.add_module('satlu', SatLU())
            setattr(self, 'conv{}'.format(i), conv)


        self.upsample = nn.Upsample(scale_factor=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for l in range(self.n_layers - 1):
            update_A = nn.Sequential(nn.Conv2d(2* self.a_channels[l], self.a_channels[l+1], (3, 3), padding=1), self.maxpool)
            setattr(self, 'update_A{}'.format(l), update_A)

        self.reset_parameters()

    def reset_parameters(self):
        for l in range(self.n_layers):
            cell = getattr(self, 'cell{}'.format(l))
            cell.reset_parameters()

    def forward(self, input):

        # print(self.a_channels)

        R_seq = [None] * self.n_layers  # 1
        H_seq = [None] * self.n_layers  # 1
        E_seq = [None] * self.n_layers  # 1
        w, h = input.size(-2), input.size(-1)  # 128, 160
        batch_size = input.size(0)

        for l in range(self.n_layers):
            E_seq[l] = Variable(torch.zeros(batch_size, 2*self.a_channels[l], w, h)).cuda()  # b, 6, 128, 160
            R_seq[l] = Variable(torch.zeros(batch_size, self.r_channels[l], w, h)).cuda()  # b, 3, 128, 160
            w = w//2  # 64
            h = h//2  # 80
        time_steps = input.size(1)  # 10
        total_error = []
        
        for t in range(time_steps):
            A = input[:,t]  # b, c, w, h (1, 3, 128, 160)
            # print(A.shape)
            A = A.type(torch.cuda.FloatTensor)
            
            for l in reversed(range(self.n_layers)):
                cell = getattr(self, 'cell{}'.format(l))  # ConvLSTM
                if t == 0:
                    E = E_seq[l]  # b, 6, 128, 160
                    R = R_seq[l]  # b, 3, 128, 160
                    hx = (R, R)
                else:
                    E = E_seq[l]
                    R = R_seq[l]
                    hx = H_seq[l]
                if l == self.n_layers - 1:
                    R, hx = cell(E, hx)  # b, 3, 128, 160
                else:
                    tmp = torch.cat((E, self.upsample(R_seq[l+1])), 1)
                    # print(tmp.shape)
                    R, hx = cell(tmp, hx)

                R_seq[l] = R  # b, 3, 128, 160
                H_seq[l] = hx  # (b, 3, 128, 160)x2


            for l in range(self.n_layers):
                conv = getattr(self, 'conv{}'.format(l))
                A_hat = conv(R_seq[l])  # b, 3, 128, 160

                # print(A_hat.shape)

                if l == 0:
                    frame_prediction = A_hat
                pos = F.relu(A_hat - A)
                neg = F.relu(A - A_hat)
                E = torch.cat([pos, neg],1)  # b, 6, 128, 160
                # print(E.shape)
                E_seq[l] = E
                if l < self.n_layers - 1:
                    update_A = getattr(self, 'update_A{}'.format(l))
                    # print(update_A.shape)
                    A = update_A(E)

            b, c, h, w = frame_prediction.shape
            if t == 0:
                frame_predictions = frame_prediction.reshape(b, 1, h, w, c)
            else:
                frame_predictions = torch.cat([frame_predictions, frame_prediction.reshape(b, 1, h, w, c)], dim=1)

            if self.output_mode == 'error':
                mean_error = torch.cat([torch.mean(e.view(e.size(0), -1), 1, keepdim=True) for e in E_seq], 1)
                # batch x n_layers
                total_error.append(mean_error)

        if self.output_mode == 'error':
            return torch.stack(total_error, 2), frame_predictions # batch x n_layers x nt
        elif self.output_mode == 'prediction':
            return frame_prediction

class SatLU(nn.Module):

    def __init__(self, lower=0, upper=255, inplace=False):
        super(SatLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input):
        return F.hardtanh(input, self.lower, self.upper, self.inplace)


    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' ('\
            + 'min_val=' + str(self.lower) \
	        + ', max_val=' + str(self.upper) \
	        + inplace_str + ')'