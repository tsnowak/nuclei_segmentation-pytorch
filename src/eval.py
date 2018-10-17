import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from Models import *
from Datasets import *
from metrics import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

weights_root    = './../weights'
config_file     = os.path.join(weights_root, 'config.txt')
weights_file    = os.path.join(weights_root, 'weights.pth')

model = UNet(3, 1).to(device)
model.load_state_dict(torch.load(weights_file))

Train = CellHistology('train', '/home/tsnowak/Software/Interview-Prep/EchonousProblem/data')
Val = CellHistology('val', '/home/tsnowak/Software/Interview-Prep/EchonousProblem/data')
TrainDL = DataLoader(Train, batch_size=1, shuffle=True, num_workers=8)
ValDL = DataLoader(Val, batch_size=1, shuffle=False, num_workers=8)

iter = 1
model.eval()
miou_list = []
for batch_t in TrainDL:
    image = batch_t['X'].to(device)
    label = batch_t['Y'].to(device)
    prediction = model(image)
    prediction = torch.sigmoid(prediction)
    miou = iou_pytorch(prediction, label, thresh=.5)
    miou_list.append(miou)

    if iter % 15 == 3:
        image = image[0,...].cpu()
        label = label[0,...].cpu()
        prediction = prediction[0,...].cpu()
        Train.visualize( image, label, prediction )
    iter+=1

print("Train Mean IoU: {}".format(np.mean(miou_list)))

iter = 1
miou_list = []
for vbatch_t in ValDL:
    image = vbatch_t['X'].to(device)
    label = vbatch_t['Y'].to(device)
    prediction = model(image)
    prediction = torch.sigmoid(prediction)
    miou = iou_pytorch(prediction, label, thresh=.5)
    miou_list.append(miou)

    if iter % 3 == 0:
        image = image[0,...].cpu()
        label = label[0,...].cpu()
        prediction = prediction[0,...].cpu()
        Val.visualize( image, label, prediction )
    iter+=1

print("Validation Mean IoU: {}".format(np.mean(miou_list)))
