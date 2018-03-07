import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch as t
from torch.utils import data
from torchvision import transforms as tsf
from preprocess import process
import utils as utils
import PIL
import nuclei
from lossUtil import *
import unet
from testData import *
import scipy.misc

TRAIN_PATH = './data/stage1_train.pth'
TEST_PATH = './data/stage1_test.pth'

test = process('./data/stage1_test/',False)
t.save(test, TEST_PATH)
train_data = process('./data/stage1_train/')
t.save(train_data, TRAIN_PATH)

s_trans = tsf.Compose([
    tsf.ToPILImage(),
    tsf.Resize((128,128)),
    tsf.ToTensor(),
    tsf.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])
]
)
t_trans = tsf.Compose([
    tsf.ToPILImage(),
    tsf.Resize((128,128),interpolation=PIL.Image.NEAREST),
    tsf.ToTensor(),]
)
dataset = nuclei.NUCLEI(train_data,s_trans,t_trans)
dataloader = t.utils.data.DataLoader(dataset,num_workers=2,batch_size=4)

model = unet.UNet(3,1)#.cuda()
optimizer = t.optim.Adam(model.parameters(),lr = 1e-3)

for epoch in range(2):
    for x_train, y_train  in tqdm(dataloader):
        x_train = t.autograd.Variable(x_train)#.cuda())
        y_train = t.autograd.Variable(y_train)#.cuda())
        optimizer.zero_grad()
        o = model(x_train)
        loss = soft_dice_loss(o, y_train)
        loss.backward()
        optimizer.step()

testset = TestDataset(TEST_PATH, s_trans)
testdataloader = t.utils.data.DataLoader(testset,num_workers=2,batch_size=2)
model = model.eval()
for name, data in testdataloader:
    im = PIL.Image.fromarray(data[1].data.cpu().permute(1,2,0).numpy()*0.5+0.5)
    im.save("./data/output/"+ name + "-i.png")
    data = t.autograd.Variable(data, volatile=True)#.cuda())
    o = model(data)
    im = PIL.Image.fromarray(o[1][0].data.cpu().numpy())
    im.save("./data/output/" + name + "-o.png")
    #break

tm=o[1][0].data.cpu().numpy()
plt.subplot(121)
plt.imshow(data[1].data.cpu().permute(1,2,0).numpy()*0.5+0.5)
plt.subplot(122)
plt.imshow(tm)