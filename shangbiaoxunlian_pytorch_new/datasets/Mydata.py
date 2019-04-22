import torch
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def default_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class CustomData(torch.utils.data.Dataset):
    def __init__(self, img_path, txt_path, dataset = '', data_transforms= None, loader = default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            if dataset == 'train' :
                self.train_data = [os.path.join(img_path, line.strip().split()[0]) for line in lines]
                self.train_labels = [int(line.strip().split()[-1]) for line in lines]
            else:
                self.test_data = [os.path.join(img_path, line.strip().split()[0]) for line in lines]
                self.test_labels = [int(line.strip().split()[-1]) for line in lines]


        self.dataset = dataset
        self.transform = data_transforms
        self.loader = loader

    def __len__(self):
        if self.dataset == 'train':
            return len(self.train_data)
        else:
            return len(self.test_data)
    def __getitem__ (self, index):
        if self.dataset == 'train':
            img,target = self.train_data[index],self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = self.loader(img)

        if self.transform is not None:
            try:
                img = self.transform(img)
            except:
                print("Cannot transform image: {}".format(img))

        return img,target,index