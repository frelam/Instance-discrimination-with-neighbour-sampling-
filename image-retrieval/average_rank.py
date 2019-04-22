from extract_cnn_vgg16_keras import VGGNet

import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import os
import glob
from query import Query
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File('featureCNN.h5', 'r')
# feats = h5f['dataset_1'][:]
feats = h5f['dataset_1'][:]
print(feats)
imgNames = h5f['dataset_2'][:]
print(imgNames)
h5f.close()
queryDir = '/home/omnisky/image-retrieval/query/7-35.jpg'
query_path = '/media/omnisky/59784c4c-dc3f-498a-92ba-e9df5f0d313f/shangbiao_32x32'
f1 = '/home/omnisky/shu_wujiandu/queryset/'
label = int(os.path.split(queryDir)[-1].split('-')[1].split('.jpg')[0])
path1 = os.path.join(f1, str(label))
file1 = glob.glob(path1 + '/*.jpg')
file_number = len(file1)
print('label:', label)

N = 923343
Nrel = file_number - 1
Nrel_all = Nrel * (Nrel + 1) / 2.0

q = Query()
rank2 = q.query(feats,imgNames,queryDir,label,N,Nrel,Nrel_all,query_path)

print('rank2:',rank2)

