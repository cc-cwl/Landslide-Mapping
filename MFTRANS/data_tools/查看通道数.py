from PIL import Image
import os

# 指定图片文件夹的路径
folder_path = r'/data1/sgy_mask2former/data_2024717/val/SegmentationClass'

# 支持的图片格式
image_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

# 用于存储不同通道模式的集合
unique_modes = set()
x=0
# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件是否是图片
    if filename.endswith(image_formats):
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, filename)

        # 打开图片
        with Image.open(file_path) as img:
            # 获取图片的模式
            mode = img.mode

            # 添加当前图片的模式到集合中
            unique_modes.add(mode)
            x+=1
            print(mode)

# 打印所有不同的通道模式
print("Unique image modes (channel types) across all images:")
for mode in unique_modes:
    print(mode)
print(x)