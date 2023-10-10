#!/usr/bin/env python

# 变化检测模型BIT训练示例脚本
# 执行此脚本前，请确认已正确安装PaddleRS库

import paddlers as pdrs
from paddlers import transforms as T

# # 数据集存放目录
# DATA_DIR = '/home/zju/data_szw/paddlers/data/levir-cd'
# # 训练集`file_list`文件路径
# TRAIN_FILE_LIST_PATH = '/home/zju/data_szw/paddlers/data/levir-cd/train.txt'
# # 验证集`file_list`文件路径
# EVAL_FILE_LIST_PATH = '/home/zju/data_szw/paddlers/data/levir-cd/val.txt'
# # 实验目录，保存输出的模型权重和结果
# EXP_DIR =  '/home/zju/data_szw/paddlers/output/cd/bit_levircd/'

# 数据集存放目录
DATA_DIR = '/home/zju/data_szw/paddlers/data/WHU-BCD'
# 训练集`file_list`文件路径
TRAIN_FILE_LIST_PATH = '/home/zju/data_szw/paddlers/data/WHU-BCD/train.txt'
# 验证集`file_list`文件路径
EVAL_FILE_LIST_PATH = '/home/zju/data_szw/paddlers/data/WHU-BCD/val.txt'
# 实验目录，保存输出的模型权重和结果
EXP_DIR =  '/home/zju/data_szw/paddlers/output/cd/bit_whubcd/'

# 影像波段数量
NUM_BANDS = 3

# # 下载和解压AirChange数据集
# pdrs.utils.download_and_decompress(
#     'https://paddlers.bj.bcebos.com/datasets/airchange.zip', path='./data/')

# 定义训练和验证时使用的数据变换（数据增强、预处理等）
# 使用Compose组合多种变换方式。Compose中包含的变换将按顺序串行执行
# API说明：https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/apis/data.md
train_transforms = T.Compose([
    # 读取影像
    T.DecodeImg(),
    # 随机裁剪
    T.RandomCrop(
        # 裁剪区域将被缩放到256x256
        crop_size=512,
        # 裁剪区域的横纵比在0.5-2之间变动
        aspect_ratio=[0.5, 2.0],
        # 裁剪区域相对原始影像长宽比例在一定范围内变动，最小不低于原始长宽的1/5
        scaling=[0.2, 1.0]),
    #对输入进行随机色彩变换
    T.RandomDistort(),
    #对输入进行随机模糊
    T.RandomBlur(),
    #随机交换两个输入图像
    T.RandomSwap(),
    # 以50%的概率实施随机水平翻转
    T.RandomHorizontalFlip(prob=0.5),
    # 以50%的概率实施随机垂直翻转
    T.RandomVerticalFlip(prob=0.5),
    # 将数据归一化到[-1,1]
    T.Normalize(
        mean=[0.5] * NUM_BANDS, std=[0.5] * NUM_BANDS),
    T.ArrangeChangeDetector('train')
])

eval_transforms = T.Compose([
    T.DecodeImg(),
    # 验证阶段与训练阶段的数据归一化方式必须相同
    T.Normalize(
        mean=[0.5] * NUM_BANDS, std=[0.5] * NUM_BANDS),
    T.ReloadMask(),
    T.ArrangeChangeDetector('eval')
])

# 分别构建训练和验证所用的数据集
train_dataset = pdrs.datasets.CDDataset(
    data_dir=DATA_DIR,
    file_list=TRAIN_FILE_LIST_PATH,
    label_list=None,
    transforms=train_transforms,
    num_workers=12,
    shuffle=True,
    with_seg_labels=False
    # binarize_labels=True
    )

eval_dataset = pdrs.datasets.CDDataset(
    data_dir=DATA_DIR,
    file_list=EVAL_FILE_LIST_PATH,
    label_list=None,
    transforms=eval_transforms,
    num_workers=12,
    shuffle=False,
    with_seg_labels=False
    # binarize_labels=True
    )

# 使用默认参数构建BIT模型
# 目前已支持的模型请参考：https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/intro/model_zoo.md
# 模型输入参数请参考：https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/change_detector.py
model = pdrs.tasks.cd.BIT(
    in_channels=NUM_BANDS,
    num_classes=2,
    backbone='resnet34',
    dec_depth=16,
    dec_head_dim=16
)

# 执行模型训练
model.train(
    num_epochs=10,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    save_interval_epochs=5,
    # 每多少次迭代记录一次日志
    log_interval_steps=5,
    save_dir=EXP_DIR,
    # 是否使用early stopping策略，当精度不再改善时提前终止训练
    early_stop=False,
    # 是否启用VisualDL日志功能
    use_vdl=True,
    # 指定从某个检查点继续训练
    resume_checkpoint=None)
