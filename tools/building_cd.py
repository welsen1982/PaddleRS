import os
from paddlers import transforms as T
import os.path as osp
import paddlers as pdrs
# from paddlers.utils.postprocs import building_regularization, prepro_mask
import argparse
from .utils import time_it
from . import mask2shape as m2s


# 影像波段数量
NUM_BANDS = 3

# model_path = "/home/zju/data_szw/paddlers/output/deeplabv3p_inria/best_model/"
# model_path = "/home/zju/lizhu_docker/building_seg/other/best_model/"


@time_it
def extract_data(image1_path,image2_path,save_dir, block_size,overlap,model_path):
    """
    extract_data.

    Args:
        image1_path (str): Path of input image1.
        image2_path (str): Path of input image2.
        save_dir (str): Directory to save the generated file lists.
        block_size(int):Size of image block. Default value is 512.
        overlap(int):Overlap between two blocks. Defaults to 36.
    """

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    test_transforms = T.Compose([
    T.DecodeImg(),
    T.SelectBand([1,2,3]),
    # T.Resize(target_size=256),
    # 验证阶段与训练阶段的数据归一化方式必须相同
    T.Normalize(
        mean=[0.5] * NUM_BANDS, std=[0.5] * NUM_BANDS),
    T.ArrangeChangeDetector('test')
    ])

    image_file = osp.split(image1_path)[-1]
    result_path = osp.join(save_dir,image_file)
    shp_file = osp.join(save_dir,image_file.split('.')[0]+'.shp')

    model = pdrs.tasks.load_model(model_path)
    model.slider_predict((image1_path,image2_path),save_dir,block_size,overlap,test_transforms)
    m2s.mask2shape(image1_path, result_path,shp_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image1_path", type=str, required=True, help="Path of input image1."
    )

    parser.add_argument(
        "--image2_path", type=str, required=True, help="Path of input image2."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="output",
        help="Directory to save the results. Default value is 'output'.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=512,
        help="Size of image block. Default value is 512.",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=36,
        help="Overlap between two blocks. Defaults to 36.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/other/best_model/",
        help="Path of model",
    )   

    args = parser.parse_args()
    extract_data(args.image1_path,args.image2_path, args.save_dir, args.block_size,args.overlap,args.model_path)
