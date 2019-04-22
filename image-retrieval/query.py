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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"



class Query:
    def __init__(self):
        ap = argparse.ArgumentParser()
        #ap.add_argument("-query", default="/home/omnisky/flask-keras-cnn-image-retrieval-master/query/",
        #    help = "Path to query which contains image to be queried")
        #ap.add_argument("-index", default='featureCNN.h5',
        #    help = "Path to index")
        ap.add_argument("-result",default="/home/omnisky/flask-keras-cnn-image-retrieval-master/result/",
            help = "Path for output retrieved images")
        args = vars(ap.parse_args())
        #args = ap.parse_args()

    def rank1(self,N,Nrel,Nrel_all,label,imlist):
        count = 0
        j = 0
        M = N * Nrel
        Ap_sum = 0
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
                    print(j/(i+1))
                    j += 1
                    count += i
                    Ap_sum += j/(i+1)

        print('j =',j)
        print('count =', count)
        Ap_sum /= j
        print('AP =', Ap_sum)
        rank = (1 / M) * (count - Nrel_all)

        return rank,Ap_sum
    def query(self,feats,imgNames,queryDir,label,N,Nrel,Nrel_all,query_path):
        print("--------------------------------------------------")
        print("               searching starts")
        print("--------------------------------------------------")
        # init VGGNet16 model
        model = VGGNet()
        # extract query image's feature, compute simlarity score and sort
        queryVec = model.extract_feat(queryDir)
        scores = np.dot(queryVec, feats.T)
        rank_ID = np.argsort(scores)[::-1]
        rank_score = scores[rank_ID]
        # print (scores)
        # print (rank_ID)
        # print (rank_score)

        maxres = N-1
        imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
        #print("top %d images in order are: " %maxres, imlist)

        rank, AP = self.rank1(N,Nrel,Nrel_all,label,imlist)

        print('rank =',rank)

        # show top #maxres retrieved result one by one
        # for i,im in enumerate(imlist):
        #     if i > 9:
        #         break
        #     image = mpimg.imread(query_path + "/"+str(im, 'utf-8'))
        #     plt.subplot(1,10 , i + 1)
        #     plt.imshow(image)
        #
        #     plt.xticks([])
        #     plt.yticks([])
        #     #plt.title("search output %d" %(i+1))
        #     plt.imshow(image)
        # plt.show()
        return rank, AP
