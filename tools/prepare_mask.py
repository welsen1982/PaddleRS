
import os.path as osp
from glob import glob

import cv2
from tqdm import tqdm


# 数据集路径
DATA_DIR = '/home/zju/data_szw/dataset/levir-cd/'

train_prefix = osp.join('train', 'label')
test_prefix = osp.join('test','label')
val_prefix = osp.join('val', 'label')

train_paths = glob(osp.join(DATA_DIR, train_prefix, '*.png'))
test_paths = glob(osp.join(DATA_DIR, test_prefix, '*.png'))
val_paths = glob(osp.join(DATA_DIR, val_prefix, '*.png'))

# print(train_paths)
for path in tqdm(train_paths+val_paths+test_paths):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    im[im>0] = 1
    # 原地改写
    cv2.imwrite(path, im)