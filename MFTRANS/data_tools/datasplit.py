import os
import shutil
import random

import cv2

# 文件夹路径
segmentation_class_dir = '/data1/sgy_mask2former/data_lushan_cwl/label'
jpeg_images_dir = '/data1/sgy_mask2former/data_lushan_cwl/image'

# 创建新的文件夹结构
os.makedirs('train/SegmentationClass', exist_ok=True)
os.makedirs('train/JPEGImages', exist_ok=True)
os.makedirs('test/SegmentationClass', exist_ok=True)
os.makedirs('test/JPEGImages', exist_ok=True)

#数据增强
# def data_au(dir1,dir2):
#     idx = 0
#     dir1 ,dir2 = sorted(dir1) ,sorted(dir2)
#     for di1,di2 in os.listdir(zip(dir1,dir2)):
#         filename = os.path.basename(di1)
#         if random.rand(0,1) > 0.5:
#             img1,img2 =cv2.imwrite(os.path.join(dir1,filename),cv2.imread(os.path.join(dir2,filename)))
#             img = xx(),img2 =xx()
#             cv2.imwrite(os.path.join(dir1,filename,f'{id}'),img)
# data_au(segmentation_class_dir,jpeg_images_dir)
# 获取文件名列表
segmentation_files = os.listdir(segmentation_class_dir)
print(segmentation_files)
jpeg_files = os.listdir(jpeg_images_dir)

# 确保文件名对应
assert sorted(segmentation_files) == sorted(jpeg_files), "SegmentationClass 文件夹和 JPEGImages 文件夹中的文件名不一致"

# 打乱文件列表并按照8:2划分
random.shuffle(segmentation_files)
total_files = len(segmentation_files)
train_files = segmentation_files[:int(0.8 * total_files)]
test_files = segmentation_files[int(0.8 * total_files):]

# 移动文件到训练集和测试集文件夹
for file in train_files:
    shutil.copy(os.path.join(segmentation_class_dir, file), 'train/SegmentationClass/')
    shutil.copy(os.path.join(jpeg_images_dir, file), 'train/JPEGImages/')

for file in test_files:
    shutil.copy(os.path.join(segmentation_class_dir, file), 'test/SegmentationClass/')
    shutil.copy(os.path.join(jpeg_images_dir, file), 'test/JPEGImages/')

print("文件划分完成！")