# -*- coding: utf-8 -*-
# Author: yongyuan.name
from extract_cnn_vgg16_keras import VGGNet

import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import os
import glob
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument("-query", default="/home/omnisky/shu_wujiandu/ShangBiaoDataSet _32X32/",
help = "Path to query which contains image to be queried")
parser.add_argument("-index", default="featureCNN1.h5",
help = "Name of index file")
# parser.add_argument("-result",default="/home/omnisky/flask-keras-cnn-image-retrieval-master/result/",
# help = "Path for output retrieved images")
args = parser.parse_args()
# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File(args.index,'r')
# feats = h5f['dataset_1'][:]
feats = h5f['dataset_1'][:]
print(feats)
imgNames = h5f['dataset_2'][:]
print(imgNames)
h5f.close()

print("--------------------------------------------------")
print("               searching starts")
print("--------------------------------------------------")

# read and show query image
queryDir ='/home/omnisky/shu_wujiandu/ShangBiaoDataSet _32X32/11-21159696.jpg'
show_image_number = 30
#f1 = '/home/omnisky/shu_wujiandu/ShangBiaoDataSet _32X32/'
# label = int(os.path.split(queryDir)[-1].split('-')[1].split('.jpg')[0])
# path1 = os.path.join(f1,str(label))
# file1 = glob.glob(path1+'/*.jpg')
# file_number = len(file1)
# print('label:',label)

# init VGGNet16 model
model = VGGNet()
# extract query image's feature, compute simlarity score and sort
queryVec = model.extract_feat(queryDir)
scores = np.dot(queryVec, feats.T)
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]
print (scores)
print (rank_ID)
print (rank_score)

# number of top retrieved images to show
# N = 923343
# Nrel = file_number - 1
# Nrel_all = Nrel*(Nrel+1)/2.0
#
# print('Nerl_all:', Nrel_all)

maxres = show_image_number
imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
print("top %d images in order are: " %maxres, imlist)

# rank = rank1(N,Nrel,Nrel_all,label,imlist)

# print('rank =',rank)

# show top #maxres retrieved result one by one
#plt.figure(figsize=(10,5))

for i,im in enumerate(imlist):
    if i > show_image_number:
        break
    image = mpimg.imread('/home/omnisky/shu_wujiandu/test'+"/"+str(im, 'utf-8'))
    plt.subplot(1,show_image_number+1 , i + 1)
    plt.imshow(image)

    plt.xticks([])
    plt.yticks([])
    #plt.title("search output %d" %(i+1))
    plt.imshow(image)
plt.show()
#cv2.imwrite()

