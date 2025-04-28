import os
from PIL import Image

# 设置要遍历的文件夹路径
folder_path = '/data1/sgy_mask2former/data_luding1/images_png'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件扩展名是否为TIF格式
    if filename.lower().endswith('.tif') or filename.lower().endswith('.png'):
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, filename)

        try:
            # 打开图片文件
            with Image.open(file_path) as img:
                # 获取图片的尺寸
                width, height = img.size

                # 检查尺寸是否为256x256
                if width != 256 or height != 256:
                    # 如果不是，删除文件
                    os.remove(file_path)
                    print(f'Deleted {file_path} as it is not 256x256.')
        except IOError:
            # 如果文件损坏或无法打开，可以选择跳过或记录错误
            print(f'Error opening {file_path}. Skipping file.')

print('Cleanup complete.')