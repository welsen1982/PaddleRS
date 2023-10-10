# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os,random,shutil
import os.path as osp
import argparse
from math import ceil
from glob import glob

import paddlers
from tqdm import tqdm

from utils import Raster, save_geotiff, time_it

SUBSETS = ('train', 'val', 'test')
SUBDIRS = ('A', 'B', 'label')
FILE_LIST_PATTERN = "{subset}.txt"

def get_path_tuples(*dirs, glob_pattern='*', data_dir=None):
    """
    Get tuples of image paths. Each tuple corresponds to a sample in the dataset.
    
    Args:
        *dirs (str): Directories that contains the images.
        glob_pattern (str, optional): Glob pattern used to match image files. 
            Defaults to '*', which matches arbitrary file. 
        data_dir (str|None, optional): Root directory of the dataset that 
            contains the images. If not None, `data_dir` will be used to 
            determine relative paths of images. Defaults to None.
    
    Returns:
        list[tuple]: For directories with the following structure:
            ├── img  
            │   ├── im1.png
            │   ├── im2.png
            │   └── im3.png
            │
            ├── mask
            │   ├── im1.png
            │   ├── im2.png
            │   └── im3.png
            └── ...

        `get_path_tuples('img', 'mask', '*.png')` will return list of tuples:
            [('img/im1.png', 'mask/im1.png'), ('img/im2.png', 'mask/im2.png'), ('img/im3.png', 'mask/im3.png')]
    """

    all_paths = []
    for dir_ in dirs:
        paths = glob(osp.join(dir_, glob_pattern), recursive=True)
        paths = sorted(paths)
        if data_dir is not None:
            paths = [osp.relpath(p, data_dir) for p in paths]
        all_paths.append(paths)
    all_paths = list(zip(*all_paths))
    return all_paths


def make_dir(target):
    '''
    创建和源文件相似的文件路径函数
    :param source: 源文件位置
    :param target: 目标文件位置
    '''
    for i in ['train', 'val','test']:
        for j in ['A','B','label']:
            path = target + '/' + i + '/' + j
            if not osp.exists(path):
                os.makedirs(path)

def divideTrainValiTest(image_dir,target):
    '''
        创建和源文件相似的文件路径
        :param source: 源文件位置
        :param target: 目标文件位置
    '''
    # 得到源文件下的种类
    # pic_name = os.listdir(source)
    
    # 对于每一类里的数据进行操作
    # for classes in pic_name:
        # 得到这一种类的图片的名字
    image_list = [e for e in os.listdir(osp.join(image_dir,'A')) if e.endswith('.tif')]
    total_number = len(image_list)*3
    random.shuffle(image_list)
    
    # 按照8：1：1比例划分
    train_list = image_list[0:int(0.85 * len(image_list))]
    valid_list = image_list[int(0.85 * len(image_list)):int(0.95 * len(image_list))]
    test_list = image_list[int(0.95 * len(image_list)):]
    # print(valid_list)
    
    with tqdm(total=total_number) as pbar:
        # 对于每个图片，移入到对应的文件夹里面
        for train_pic in train_list:
            shutil.copyfile(osp.join(image_dir,'A',train_pic), osp.join(target, 'train','A',train_pic))
            shutil.copyfile(osp.join(image_dir,'B',train_pic), osp.join(target, 'train','B',train_pic))
            shutil.copyfile(osp.join(image_dir,'label',train_pic), osp.join(target, 'train','label',train_pic))
            pbar.update(1)
        for validation_pic in valid_list:
            shutil.copyfile(osp.join(image_dir,'A',validation_pic), osp.join(target, 'val','A',validation_pic))
            shutil.copyfile(osp.join(image_dir,'B',validation_pic), osp.join(target, 'val','B',validation_pic))
            shutil.copyfile(osp.join(image_dir,'label',validation_pic), osp.join(target, 'val','label',validation_pic))
            pbar.update(1)
        for test_pic in test_list:
            shutil.copyfile(osp.join(image_dir,'A',test_pic), osp.join(target, 'test','A',test_pic))
            shutil.copyfile(osp.join(image_dir,'B',test_pic), osp.join(target, 'test','B',test_pic))
            shutil.copyfile(osp.join(image_dir,'label',test_pic), osp.join(target, 'test','label',test_pic))
            pbar.update(1)

    for subset in SUBSETS:
        path_tuples = get_path_tuples(
            *(osp.join(target, subset, subdir) for subdir in SUBDIRS),
            glob_pattern='**/*.tif',
            data_dir=target)
        file_list = osp.join(
            target, FILE_LIST_PATTERN.format(subset=subset))
        create_file_list(file_list, path_tuples)
        print(f"Write file list to {file_list}.")

def create_file_list(file_list, path_tuples, sep=' '):
    """
    Create file list.
    
    Args:
        file_list (str): Path of file list to create.
        path_tuples (list[tuple]): See get_path_tuples().
        sep (str, optional): Delimiter to use when writing lines to file list. 
            Defaults to ' '.
    """

    with open(file_list, 'w') as f:
        for tup in path_tuples:
            line = sep.join(tup)
            f.write(line + '\n')


def _calc_window_tf(geot, loc):
    x, hr, r1, y, r2, vr = geot
    nx, ny = loc
    return (x + nx * hr, hr, r1, y + ny * vr, r2, vr)


@time_it
def split_data(image_dir, gt_dir, block_size, save_dir):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    os.makedirs(osp.join(save_dir, "A"),exist_ok=True)
    os.makedirs(osp.join(save_dir, "B"),exist_ok=True)
    if gt_dir is not None:
        os.makedirs(osp.join(save_dir, "label"),exist_ok=True)

    image_list = [e for e in os.listdir(osp.join(image_dir,'A')) if e.endswith('.tif')]
    
    for image_file in image_list:
        image_name, image_ext = image_file.split(".")
        image_a = Raster(osp.join(image_dir,'A',image_file))
        image_b = Raster(osp.join(image_dir,'B',image_file))
        mask = Raster(osp.join(gt_dir,image_file)) if gt_dir is not None else None
        if mask is not None and (image_a.width != mask.width or
                                image_a.height != mask.height):
            raise ValueError("image's shape must equal mask's shape.")
        rows = ceil(image_a.height / block_size)
        cols = ceil(image_a.width / block_size)
        total_number = int(rows * cols)
        with tqdm(total=total_number) as pbar:
            for r in range(rows):
                for c in range(cols):
                    loc_start = (c * block_size, r * block_size)
                    image_a_title = image_a.getArray(loc_start,
                                                (block_size, block_size))
                    image_a_save_path = osp.join(save_dir, "A", (
                        image_name + "_" + str(r) + "_" + str(c) + "." + image_ext))
                    window_a_geotf = _calc_window_tf(image_a.geot, loc_start)
                    save_geotiff(image_a_title, image_a_save_path, image_a.proj,
                                window_a_geotf)
                    
                    image_b_title = image_b.getArray(loc_start,
                                                (block_size, block_size))
                    image_b_save_path = osp.join(save_dir, "B", (
                        image_name + "_" + str(r) + "_" + str(c) + "." + image_ext))
                    window_b_geotf = _calc_window_tf(image_b.geot, loc_start)
                    save_geotiff(image_b_title, image_b_save_path, image_b.proj,
                                window_b_geotf)
                    
                    if mask is not None:
                        mask_title = mask.getArray(loc_start,
                                                (block_size, block_size))
                        mask_save_path = osp.join(save_dir, "label",
                                                (image_name + "_" + str(r) + "_" +
                                                str(c) + "." + image_ext))
                        save_geotiff(mask_title, mask_save_path, image_a.proj,
                                    window_a_geotf)
                    pbar.update(1)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, \
                        help="Path of input image.")
    parser.add_argument("--mask_path", type=str, default=None, \
                        help="Path of input labels.")
    parser.add_argument("--block_size", type=int, default=512, \
                        help="Size of image block. Default value is 512.")
    parser.add_argument("--save_dir", type=str, default="dataset", \
                        help="Directory to save the results. Default value is 'dataset'.")
    args = parser.parse_args()
    split_data(args.image_path, args.mask_path, args.block_size, args.save_dir)
    make_dir(args.save_dir)
    divideTrainValiTest(args.save_dir,args.save_dir)

