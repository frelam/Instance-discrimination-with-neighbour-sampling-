# -*- coding: utf-8 -*-
# Author: yongyuan.name
import os
import h5py
import numpy as np
import argparse
import torchvision.transforms as transforms

from extract_cnn_vgg16_keras import VGGNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# ap = argparse.ArgumentParser()
# ap.add_argument("-database", default="/media/omnisky/59784c4c-dc3f-498a-92ba-e9df5f0d313f/shangbiao_32x32",
# 	help = "Path to database which contains images to be indexed")
# ap.add_argument("-index", default="featureCNN_k=10000.t7.h5",
# 	help = "Name of index file")
# args = vars(ap.parse_args())

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument("-database", default="/media/omnisky/59784c4c-dc3f-498a-92ba-e9df5f0d313f/shangbiao_32x32",
help = "Path to database which contains images to be indexed")
parser.add_argument("-index", default="featureCNN_Resnet50_us_GAN_m=4096_2.1.1.h5",
help = "Name of index file")
args = parser.parse_args()

'''
ckpt_resnet34_Mymodel.t7
 Returns a list of filenames for all jpg images in a directory. 
'''
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

'''
 Extract features and index the images
'''
if __name__ == "__main__":

    #db = args["database"]
    img_list = get_imlist(args.database)

    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")
    
    feats = []
    names = []

    model = VGGNet()
    for i, img_path in enumerate(img_list):
        #print (img_path)
        norm_feat = model.extract_feat(img_path)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        print("extracting feature from image No. %d , %d images in total" %((i+1), len(img_list)))

    feats = np.array(feats)
    # print(feats)
    # directory for storing extracted features
    output = args.index
    
    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")


    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data = feats)
    # h5f.create_dataset('dataset_2', data = names)
    h5f.create_dataset('dataset_2', data = np.string_(names))
    h5f.close()
