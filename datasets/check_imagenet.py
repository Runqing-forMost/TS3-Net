import os
base = '/media/data/jrq_data/open_lth_datasets/ILSVRC2012_img_val/'
imgs = os.listdir('/media/data/jrq_data/open_lth_datasets/ILSVRC2012_img_val')
imgs = sorted(imgs)[:50]


c = 0
img_total = []
for i in imgs:
    temp = base + i
    sub_img = os.listdir(temp)
    for j in sub_img:
        img_total.append(os.path.join(temp, j))
        print(img_total)
        break
    c += 1
