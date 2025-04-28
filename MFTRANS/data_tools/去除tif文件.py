import os

# 指定要搜索的目录
directory = r'/data1/sgy_mask2former/data/train/SegmentationClass'

# 遍历目录中的文件
for filename in os.listdir(directory):
    # 检查文件后缀是否为.tif
    if filename.endswith('.tif'):
        # 构建完整的文件路径
        file_path = os.path.join(directory, filename)
        # 删除文件
        os.remove(file_path)
        print(f'Deleted: {file_path}')