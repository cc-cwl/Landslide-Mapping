import os

# 定义文件夹路径
folder_path = '/data1/sgy_mask2former/data_2024717/val/SegmentationClass'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件是否为PNG图片且名称中包含“jzg”
    if filename.endswith('.png') and 'jzg' in filename:
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, filename)
        # 删除文件
        os.remove(file_path)
        print(f"Deleted: {file_path}")