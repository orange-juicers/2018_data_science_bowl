import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch as t
from torch.utils import data
from torchvision import transforms as tsf
import torchvision.utils as v_utils
from preprocess import process
import utils as utils
import PIL
import nuclei
from lossUtil import *
import unet
from average import *
from testData import *
import scipy.misc
import uuid

THIS_DIR = os.path.dirname(__file__)
TRAIN_PATH = THIS_DIR + '/data/stage1_train.pth'
TEST_PATH = THIS_DIR + '/data/stage1_test.pth'

test = process(THIS_DIR + '/data/stage1_test/',False)
t.save(test, TEST_PATH)

train_data = process(THIS_DIR + '/data/stage1_train/')
t.save(train_data, TRAIN_PATH)
# ============================
# Initialize Hyper-parameters
# ============================
num_epochs = 20
train_batch_size = 10
learning_rate = 1e-3
test_batch_size = 1
num_workers = 2
# ================
# Train
# ================
# source image transform
src_trans = tsf.Compose([
    tsf.ToPILImage(),
    tsf.Resize((128,128)),
    tsf.ToTensor(),
    tsf.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])
])
# target image transform
tgt_trans = tsf.Compose([
    tsf.ToPILImage(),
    tsf.Resize((128,128),interpolation=PIL.Image.NEAREST),
    tsf.ToTensor(),]
)

# load the train dataset
train_dataset = nuclei.NUCLEI(train_data,src_trans,tgt_trans)
train_loader = t.utils.data.DataLoader(dataset = train_dataset,num_workers=num_workers,batch_size=train_batch_size, shuffle=True)

# load the model
model = unet.UNet(3,1)#.cuda()
optimizer = t.optim.Adam(model.parameters(),lr = learning_rate)

# =============
# checkpoint
# ===========
checkpoint_file = THIS_DIR + '/data/checkpoint-'+str(uuid.uuid4())+'.pth.tar'

# iterate over the train data set for training
best_loss = 99999
for epoch in range(num_epochs):
    train_loss = Average()
    model.train()
    for i, (x_train, y_train)  in enumerate(train_loader):
        x_train = t.autograd.Variable(x_train)#.cuda())
        y_train = t.autograd.Variable(y_train)#.cuda())
        optimizer.zero_grad()
        o = model(x_train)
        loss = soft_dice_loss(o, y_train)
        loss.backward()
        optimizer.step()
        train_loss.update(loss.data[0], x_train.size(0))
    
    print("Epoch {}, Loss: {}".format(epoch+1, train_loss.avg))
    is_best = bool(train_loss.avg < best_loss)
    if is_best:
        best_loss = train_loss.avg
    #https://blog.floydhub.com/checkpointing-tutorial-for-tensorflow-keras-and-pytorch/
    utils.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'best_loss': best_loss
    }, is_best, checkpoint_file)
# Q: Can we save the weights at this time to a file?
# torch.save(model.state_dict(), 'MODEL_EPOCH{}_LOSS{}.pth'.format(epoch+1, l))

# ======================
# Test
# ======================
# Q: Can we load the saved weights at this time?

# Dropout and BatchNorm (and maybe some custom modules) behave differently during training and evaluation. 
# You must let the model know when to switch to eval mode by calling .eval() on the model.
# https://stackoverflow.com/questions/48146926/whats-the-meaning-of-function-eval-in-torch-nn-module
model = model.eval()

# load test data
test_dataset = TestDataset(TEST_PATH, src_trans)
test_loader = t.utils.data.DataLoader(dataset = test_dataset,num_workers=num_workers,batch_size=test_batch_size, shuffle=False)

# create output dir
OUTPUT_DIR= THIS_DIR+"/data/output/"
if not os.path.exists(OUTPUT_DIR):
    os.removedirs(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

# iterate over test data and generate output images
for name, data in test_loader:
    inputVar = t.autograd.Variable(data, volatile=True)#.cuda())
    v_utils.save_image(inputVar.cpu().data, THIS_DIR+"/data/output/"+ name[0] + "-i.png")
    outputVar = model(inputVar)
    v_utils.save_image(outputVar.cpu().data, THIS_DIR+"/data/output/"+ name[0] + "-o.png")

# in the end plot 1 image
#tm=o[1][0].data.cpu().numpy()
#plt.subplot(121)
#plt.imshow(data[1].data.cpu().permute(1,2,0).numpy()*0.5+0.5)
#plt.subplot(122)
#plt.imshow(tm)