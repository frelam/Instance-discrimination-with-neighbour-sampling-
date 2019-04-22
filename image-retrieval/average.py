import h5py
import os
import glob
from query import Query

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File('featureCNN_Resnet50_us_GAN_m=4096_2.1.1.h5', 'r')
# feats = h5f['dataset_1'][:]
feats = h5f['dataset_1'][:]
#print(feats)
imgNames = h5f['dataset_2'][:]
#print(imgNames)
h5f.close()
rank = []
AP_total = []
minNumber = []
query_path = '/media/omnisky/683cd494-d120-4dbd-81a4-eb3a90330106/shangbiao_32x32'
l = os.listdir(query_path)
N = len(l)
f1 = '/home/omnisky/shu_wujiandu/queryset/'
min = 1
AP_max = 0
queryNumeber = []
for i in range(1, 36):
    queryset = '/home/omnisky/shu_wujiandu/queryset/' + str(i)

    file = glob.glob(queryset + '/*.jpg')
    queryNumeber.append(len(file))
print(queryNumeber)
q = Query()
for j in range(1,36):
    for i in range(1,(queryNumeber[j-1]+1)):
        #queryDir = '/home/omnisky/image-retrieval/query/2-' + str(i) + '.jpg'
        #print(queryDir)
        queryDir = '/home/omnisky/image-retrieval/query/'+str(i)+'-'+ str(j) + '.jpg'
        #label = int(os.path.split(queryDir)[-1].split('-')[1].split('.jpg')[0])
        #path1 = os.path.join(f1, str(label))
        #file1 = glob.glob(path1 + '/*.jpg')
        label = j
        file_number = queryNumeber[j-1]
        print('label:', label)
        Nrel = file_number - 1
        Nrel_all = Nrel * (Nrel + 1) / 2.0
        rank2, AP = q.query(feats, imgNames, queryDir, label, N, Nrel, Nrel_all, query_path)
        if min > rank2:
            min = rank2
            min_num = i
            AP_max = AP
    print('j_min=', min)
    rank.append(min)
    AP_total.append(AP_max)
    minNumber.append(min_num)
    min = 1

print('minNumber=',minNumber)
print('best_min=',min)
print('rank=',rank)

average_rank = 0
for i in rank:
    average_rank += i
average_rank = average_rank / 35.0
print('average_rank =',average_rank)
MAP = 0
for i in AP_total:
    MAP += i
MAP = MAP / 35.0
print('MAP =',MAP)

MSE = 0
for i in rank:
    MSE += (i - average_rank) ** 2
MSE = MSE / 35.0
print('MSE =', MSE)
print('Resnet34_us_m=4096')
