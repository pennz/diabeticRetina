#!/usr/bin/env python
# coding: utf-8

#
#     high-resolution images
#     better data sampling
#     ensuring there is no leaking between training and validation sets, sample(replace = True) is real dangerous
#     better target variable (age) normalization
#     pretrained models
#     attention/related techniques to focus on areas
#

# In[ ]:


from fastai import vision
import fastai
import pdb
import pretrainedmodels
import os
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from tqdm import tqdm
import torch.nn as nn
import torchvision
import pandas as pd
import numpy as np
import sys
package_dir = "../input/pretrained-models/pretrained-models/pretrained-models.pytorch-master/"
sys.path.insert(0, package_dir)

device = torch.device("cuda:0")
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[ ]:


class RetinopathyDatasetTest(Dataset):
    def __init__(self, csv_file, transform):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(
            '../input/aptos2019-blindness-detection/test_images', self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = self.transform(image)
        return {'image': image}


# In[ ]:


model = pretrainedmodels.__dict__['resnet101'](pretrained=None)

model.avg_pool = nn.AdaptiveAvgPool2d(1)

model.last_linear = nn.Sequential(
    nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1,
                   affine=True, track_running_stats=True),
    nn.Dropout(p=0.25),
    nn.Linear(in_features=2048, out_features=2048, bias=True),
    nn.ReLU(),
    nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1,
                   affine=True, track_running_stats=True),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=2048, out_features=1, bias=True),
)
model.load_state_dict(torch.load("../input/mmmodel/model.bin"))
## change model loss, to see if it can help

model = model.to(device)


# In[ ]:


for param in model.parameters():
    param.requires_grad = False

model.eval()


# In[ ]:


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_dataset = RetinopathyDatasetTest(csv_file='../input/aptos2019-blindness-detection/sample_submission.csv',
                                      transform=test_transform)

# #### TTA for the lazy, like me -> changed to only test once
test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4)
test_preds1 = np.zeros((len(test_dataset), 1))
tk0 = tqdm(test_data_loader)
for i, x_batch in enumerate(tk0):
    x_batch = x_batch["image"]
    pred = model(x_batch.to(device))
    test_preds1[i * 32:(i + 1) * 32] = pred.detach(
    ).cpu().squeeze().numpy().ravel().reshape(-1, 1)


# In[ ]:

test_preds = test_preds1


# In[ ]:


coef = [0.5, 1.5, 2.5, 3.5]

for i, pred in enumerate(test_preds):
    if pred < coef[0]:
        test_preds[i] = 0
    elif pred >= coef[0] and pred < coef[1]:
        test_preds[i] = 1
    elif pred >= coef[1] and pred < coef[2]:
        test_preds[i] = 2
    elif pred >= coef[2] and pred < coef[3]:
        test_preds[i] = 3
    else:
        test_preds[i] = 4


sample = pd.read_csv(
    "../input/aptos2019-blindness-detection/sample_submission.csv")
sample.diagnosis = test_preds.astype(int)
sample.to_csv("submission.csv", index=False)


# In[ ]:


sample


# In[ ]:


test_preds4
