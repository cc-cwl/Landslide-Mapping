from PIL import Image
import os

# 指定图片文件夹的路径
folder_path = r'/data1/sgy_mask2former/data_2024717/train/SegmentationClass'

# 用于存储所有不同像素值的集合
unique_pixels = set()

# 支持的图片格式
image_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件是否是图片
    if filename.endswith(image_formats):
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, filename)

        # 打开图片
        with Image.open(file_path) as img:
            # 将图片转换为'RGB'模式
            img = img.convert('RGB')

            # 获取图片的尺寸
            width, height = img.size

            # 遍历图片中的每个像素
            for x in range(width):
                for y in range(height):
                    # 添加当前像素的RGB值到集合中
                    pixel = img.getpixel((x, y))
                    unique_pixels.add(tuple(pixel))

# 打印所有不同的像素值
print("Unique pixel values across all images:")
for pixel in unique_pixels:
    print(pixel)
