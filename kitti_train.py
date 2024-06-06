import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_loader.kitti_data import KITTI
from prednet_model.prednet import PredNet

from evaluation.evaluate import eval

num_epochs = 150
batch_size = 4
A_channels = tuple([3])
R_channels = tuple([3])

A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)

lr = 0.001 # if epoch < 75 else 0.0001
nt = 5 # num of time steps

layer_loss_weights = Variable(torch.FloatTensor([[1.], [0.], [0.], [0.]]).cuda())
time_loss_weights = 1./(nt - 1) * torch.ones(nt, 1)
time_loss_weights[0] = 0
time_loss_weights = Variable(time_loss_weights.cuda())

DATA_DIR = '/home/ana/Study/Occupancy_flow/Short_PredNet/data_evidential_grid_splits/double_prong'

train_file = os.path.join(DATA_DIR, 'X_val.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_val.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')


kitti_train = KITTI(train_file, train_sources, nt)
kitti_val = KITTI(val_file, val_sources, nt)

train_loader = DataLoader(kitti_train, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(kitti_val, batch_size=batch_size, shuffle=True, num_workers=2)

model = PredNet(R_channels, A_channels, output_mode='error')
if torch.cuda.is_available():
    print('Using GPU.')
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def lr_scheduler(optimizer, epoch):
    if epoch < num_epochs //2:
        return optimizer
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
        return optimizer



# for epoch in range(num_epochs):
#     optimizer = lr_scheduler(optimizer, epoch)
#     for i, inputs in enumerate(train_loader):
#         inputs = inputs.permute(0, 1, 4, 2, 3) # batch x time_steps x channel x width x height
#         inputs = Variable(inputs.cuda())
#         errors = model(inputs) # batch x n_layers x nt
#         # print(errors.data[0])
#         loc_batch = errors.size(0)
#         errors = torch.mm(errors.view(-1, nt), time_loss_weights) # batch*n_layers x 1
#         errors = torch.mm(errors.view(loc_batch, -1), layer_loss_weights)
#         errors = torch.mean(errors)
#
#         optimizer.zero_grad()
#
#         errors.backward()
#
#         optimizer.step()
#         if i%10 == 0:
#             print('Epoch: {}/{}, step: {}/{}, errors: '.format(epoch, num_epochs, i, len(kitti_train)//batch_size))
#
# torch.save(model.state_dict(), 'training.pt')
import numpy as np
if __name__=='__main__':
    for epoch in range(num_epochs):
        optimizer = lr_scheduler(optimizer, epoch)
        with torch.inference_mode():
            model.eval()
            is_l, mse_l, msed_l = [], [], []
            X_test, X_hats = [], []
            for i, inputs in enumerate(val_loader):
                inputs = inputs.permute(0, 1, 4, 2, 3)  # batch x time_steps x channel x width x height
                inputs = Variable(inputs.cuda())
                _, X_hat = model(inputs)  # batch x n_layers x nt
                b, ts, c, w, h = inputs.shape
                is_, mse, mesd = eval(inputs.reshape(b, ts, w, h, c), X_hat)

                is_l.append(is_)
                mse_l.append(mse)
                msed_l.append(mesd)

            is_ = sum(is_l) / len(is_l)
            mse = sum(mse_l) / len(mse_l)
            mesd = sum(msed_l) / len(msed_l)

            print((f'Epoch eval: {epoch-1}/{num_epochs}, IS: {is_}, MSE: {mse}, MESD: {mesd}'))

        for i, inputs in enumerate(train_loader):
            inputs = inputs.permute(0, 1, 4, 2, 3) # batch x time_steps x channel x width x height
            inputs = Variable(inputs.cuda())
            errors, _ = model(inputs) # batch x n_layers x nt
            loc_batch = errors.size(0)
            errors = torch.mm(errors.view(-1, nt), time_loss_weights) # batch*n_layers x 1
            errors = torch.mm(errors.view(loc_batch, -1), layer_loss_weights)
            errors = torch.mean(errors)

            optimizer.zero_grad()

            errors.backward()

            optimizer.step()
            if i%10 == 0:
                print(f'Epoch: {epoch}/{num_epochs}, step: {i}/{len(kitti_train)//batch_size}, errors: {errors}')






