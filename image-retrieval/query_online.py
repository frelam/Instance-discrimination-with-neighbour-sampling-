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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ap = argparse.ArgumentParser()
# ap.add_argument("-query", default="/home/omnisky/flask-keras-cnn-image-retrieval-master/query/",
# 	help = "Path to query which contains image to be queried")
# ap.add_argument("-index", default='featureCNN_k=10000.h5',
# 	help = "Path to index")
# ap.add_argument("-result",default="/home/omnisky/flask-keras-cnn-image-retrieval-master/result/",
# 	help = "Path for output retrieved images")
# args = vars(ap.parse_args())
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument("-query", default="/media/omnisky/59784c4c-dc3f-498a-92ba-e9df5f0d313f/shangbiao_32x32/",
help = "Path to query which contains image to be queried")
parser.add_argument("-index", default="featureCNN1.h5",
help = "Name of index file")
parser.add_argument("-result",default="/home/omnisky/flask-keras-cnn-image-retrieval-master/result/",
help = "Path for output retrieved images")
args = parser.parse_args()

def rank1(N,Nrel,Nrel_all,label,imlist):
    count = 0
    j = 0
    M = N * Nrel
    for i, name in enumerate(imlist):
        name = str(name, 'utf-8')
        #print(name)
        if name == 'deneme.jpg':
            continue
        elif name.split('-')[1] == str(label)+'.jpg':
            name_label = name.split('-')[1].split('.jpg')[0]
            if name_label == str(label):
                print(name)
                print(i)
                j += 1
                count += i + 1
    print('j =',j)
    print('N =',N)
    print('Nrel =', Nrel)
    print('Nrel_all=',Nrel_all)
    print('count =', count)
    rank = (1 / M) * (count - Nrel_all)

    return rank
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
queryDir = '/home/omnisky/shu_wujiandu/queryset_32X32/3-14.jpg'
#queryDir = '/home/omnisky/image-retrieval/query/1-19.jpg'  # 1-19 xiaoguo zuihao
#queryImg = mpimg.imread(queryDir)
#plt.title("Query Image")
#plt.imshow(queryImg)
#plt.show()

f1 = '/home/omnisky/shu_wujiandu/queryset/'
label = int(os.path.split(queryDir)[-1].split('-')[1].split('.jpg')[0])
path1 = os.path.join(f1,str(label))
file1 = glob.glob(path1+'/*.jpg')
file_number = len(file1)
print('label:',label)

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
N = 923343
Nrel = file_number - 1
Nrel_all = Nrel*(Nrel+1)/2.0

print('Nerl_all:', Nrel_all)

maxres = N-1
imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
#print("top %d images in order are: " %maxres, imlist)

rank = rank1(N,Nrel,Nrel_all,label,imlist)

print('rank =',rank)

# show top #maxres retrieved result one by one
#plt.figure(figsize=(10,5))
show_image_number = 15
for i,im in enumerate(imlist):
    if i > show_image_number:
        break
    image = mpimg.imread('/media/omnisky/59784c4c-dc3f-498a-92ba-e9df5f0d313f/shangbiao_32x32'+"/"+str(im, 'utf-8'))
    plt.subplot(1,show_image_number+1 , i + 1)
    plt.imshow(image)

    plt.xticks([])
    plt.yticks([])
    #plt.title("search output %d" %(i+1))
    plt.imshow(image)
plt.show()
#cv2.imwrite()

