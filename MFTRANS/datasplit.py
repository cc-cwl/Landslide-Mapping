# import os
# import shutil
# import random
#
# # 文件夹路径
# segmentation_class_dir = '/data1/sgy_mask2former/data_lushan/luding_label'
# jpeg_images_dir = '/data1/sgy_mask2former/data_lushan/luding_image'
#
# # 创建新的文件夹结构
# os.makedirs('/data1/sgy_mask2former/data_lushan/train/SegmentationClass', exist_ok=True)
# os.makedirs('/data1/sgy_mask2former/data_lushan/train/JPEGImages', exist_ok=True)
# os.makedirs('/data1/sgy_mask2former/data_lushan/val/SegmentationClass', exist_ok=True)
# os.makedirs('/data1/sgy_mask2former/data_lushan/val/JPEGImages', exist_ok=True)
#
# # 获取文件名列表
# segmentation_files = os.listdir(segmentation_class_dir)
# jpeg_files = os.listdir(jpeg_images_dir)
#
# # 确保文件名对应
# assert sorted(segmentation_files) == sorted(jpeg_files), "SegmentationClass 文件夹和 JPEGImages 文件夹中的文件名不一致"
#
# # 打乱文件列表并按照8:2划分
# random.shuffle(segmentation_files)
# total_files = len(segmentation_files)
# train_files = segmentation_files[:int(0.8 * total_files)]
# test_files = segmentation_files[int(0.8 * total_files):]
#
# # 移动文件到训练集和测试集文件夹
# for file in train_files:
#     shutil.copy(os.path.join(segmentation_class_dir, file), '/data1/sgy_mask2former/data_lushan/train/SegmentationClass/')
#     shutil.copy(os.path.join(jpeg_images_dir, file), '/data1/sgy_mask2former/data_lushan/train/JPEGImages/')
#
# for file in test_files:
#     shutil.copy(os.path.join(segmentation_class_dir, file), '/data1/sgy_mask2former/data_lushan/val/SegmentationClass/')
#     shutil.copy(os.path.join(jpeg_images_dir, file), '/data1/sgy_mask2former/data_lushan/val/JPEGImages/')
#     # shutil.copy(os.path.join(jpeg_images_dir, file), 'test/JPEGImages/')
#
# print("文件划分完成！")
#
#
# # 目前，泸定的数据有问题
# #

import os
import shutil
import random

# 定义全局变量表示 data_lushan 的路径
data_path = '/data1/sgy_mask2former/data_lushan_cwl'

# 文件夹路径
segmentation_class_dir = os.path.join(data_path, 'labels')
jpeg_images_dir = os.path.join(data_path, 'images')

# 创建新的文件夹结构
os.makedirs(os.path.join(data_path, 'train', 'SegmentationClass'), exist_ok=True)
os.makedirs(os.path.join(data_path, 'train', 'JPEGImages'), exist_ok=True)
os.makedirs(os.path.join(data_path, 'val', 'SegmentationClass'), exist_ok=True)
os.makedirs(os.path.join(data_path, 'val', 'JPEGImages'), exist_ok=True)

# 获取文件名列表
segmentation_files = os.listdir(segmentation_class_dir)
jpeg_files = os.listdir(jpeg_images_dir)

# 确保文件名对应
assert sorted(segmentation_files) == sorted(jpeg_files), "SegmentationClass 文件夹和 JPEGImages 文件夹中的文件名不一致"

# 打乱文件列表并按照 8:2 划分
random.shuffle(segmentation_files)
total_files = len(segmentation_files)
train_files = segmentation_files[:int(0.7 * total_files)]
test_files = segmentation_files[int(0.7 * total_files):]

# 移动文件到训练集和测试集文件夹
for file in train_files:
    shutil.copy(os.path.join(segmentation_class_dir, file), os.path.join(data_path, 'train', 'SegmentationClass'))
    shutil.copy(os.path.join(jpeg_images_dir, file), os.path.join(data_path, 'train', 'JPEGImages'))

for file in test_files:
    shutil.copy(os.path.join(segmentation_class_dir, file), os.path.join(data_path, 'val', 'SegmentationClass'))
    shutil.copy(os.path.join(jpeg_images_dir, file), os.path.join(data_path, 'val', 'JPEGImages'))

print("文件划分完成！")