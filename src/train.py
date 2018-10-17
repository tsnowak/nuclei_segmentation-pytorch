import os
from time import time
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from Datasets import *
from Models import *
from metrics import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

weights_root    = './../weights'
config_file     = os.path.join(weights_root, 'config1.txt')
weights_file    = os.path.join(weights_root, 'weights1.pth')
try:
    os.makedirs(weights_root)
except OSError:
    pass

epochs      = 25
batch_size  = 1
lr          = 1e-3
lr_decay    = .8
weight_decay= 1e-9

config = '{: <25} {: <25}\n'.format('Epochs: ', epochs) + \
         '{: <25} {: <25}\n'.format('Batch Size: ', batch_size) + \
         '{: <25} {: <25}\n'.format('Learning Rate: ', lr) + \
         '{: <25} {: <25}\n'.format('Learning Rate Decay: ', lr_decay) + \
         '{: <25} {: <25}\n'.format('Weight Decay: ', weight_decay)


model = UNet(3, 1).to(device)

Train = CellHistology('train', '/home/tsnowak/Software/Interview-Prep/EchonousProblem/data')
Val = CellHistology('val', '/home/tsnowak/Software/Interview-Prep/EchonousProblem/data')
TrainDL = DataLoader(Train, batch_size=batch_size, shuffle=True, num_workers=8)
ValDL = DataLoader(Val, batch_size=batch_size, shuffle=False, num_workers=8)

loss_fn = nn.BCEWithLogitsLoss()
opt = optim.Adam(model.parameters(), lr=lr, betas=(.9, .999), weight_decay=weight_decay)
lr_sched = optim.lr_scheduler.ExponentialLR(opt, gamma=lr_decay)

def train():

    iter = 1
    for epoch in range(1, epochs+1):

        print("Epoch: " + str(epoch))
        lr_sched.step() # learning rate scheduler
        ts = time()
        model.train()

        for batch_t in TrainDL:
            opt.zero_grad()
            image = Variable(batch_t['X']).to(device)
            label = Variable(batch_t['Y']).to(device)
            prediction = model(image)
            loss = loss_fn(prediction, label)
            print("iter: {}, loss: {}".format(iter, loss.data))
            loss.backward()
            opt.step()
            iter+=1

        print("Validation run %d: " % (epoch,))
        miou_list = []
        for vbatch_t in ValDL:
            image = vbatch_t['X'].to(device)
            label = vbatch_t['Y'].to(device)
            prediction = model(image)
            vloss = loss_fn(prediction, label)
            prediction = torch.sigmoid(prediction)
            miou = iou_pytorch(prediction, label, thresh=.5)
            miou_list.append(miou)
            print("loss: {}, miou: {}".format(vloss.data, miou))

        print("Avg. Mean IOU: {}".format(np.mean(miou_list)))

    with open(config_file, 'w') as f:
        f.write(config)
    torch.save(model.state_dict(), weights_file)

if __name__ == '__main__':
    train()
