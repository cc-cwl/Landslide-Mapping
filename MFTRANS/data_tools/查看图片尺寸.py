from PIL import Image
import os

# 设置文件夹路径


from PIL import Image
import os

# 设置你的图片文件夹路径
folder_path = '/data1/sgy_mask2former/data_2024717/val/SegmentationClass'

# 存储图片尺寸的集合
sizes = set()

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件是否是图片
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
        # 打开图片
        with Image.open(os.path.join(folder_path, filename)) as img:
            # 获取图片尺寸
            size = img.size
            # 将尺寸添加到集合中
            sizes.add(size)

# 打印出所有不同的尺寸
print("不同的图片尺寸有：")
for size in sizes:
    print(size)