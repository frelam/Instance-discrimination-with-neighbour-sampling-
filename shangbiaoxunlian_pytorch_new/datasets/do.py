import numpy as np
import cv2
import os
import argparse
import shutil


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to path')
    parser.add_argument('--path_1', default="/home/omnisky/shu_wujiandu/new_queryset_allinone")
    #parser.add_argument('--path_2', default="/home/omnisky/shangbiao/222")
    args = parser.parse_args()
    return args

id = []
label = []
label1 = []
args = get_parser()
image_name = os.listdir(args.path_1)  # du wenjian mingzi   luanxu
#path_2_idx.sort(key=lambda x:int(x[:]))
print (image_name) #  os.path.join(args.path_1,path_2_idx[i])
for i in image_name:
    a = i.split('.')[0].split('-')[1]
    a = int(a)
    label1.append(a)

print(label1)
id = np.argsort(label1)
# print (id)
image_name = np.array(image_name)[id]
print (image_name)
for i in image_name:
    a = i.split('.')[0].split('-')[1]
    a = int(a)
    label.append(a)
print (label)
z = list(zip(image_name,label))
print(z)
print (z[0][0])
print (z[0][1])

f = open("/home/omnisky/shu_wujiandu/test_colour.txt", 'w')
for i in range(len(z)):
    f.write(str(z[i][0]))
    f.write(' ')
    f.write(str(z[i][1]))
    f.write('\r\n')




