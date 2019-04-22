import os
txt_path = '/home/omnisky/shu_wujiandu/shangbiaoxunlian_pytorch_new/TxtFile/' + 'train'+ '.txt'
with open(txt_path) as input_file:
    lines = input_file.readlines()

    path = [os.path.join('/ImagePath', line.strip().split()[0]) for line in lines ]
    path1 = [int(line.strip().split()[-1]) for line in lines]
    print (path)
    print (path1)





