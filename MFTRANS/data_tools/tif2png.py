from PIL import Image
import os

# 设置输入文件夹路径
input_folder_path = '/data1/sgy_mask2former/data_luding1/labels'
# 设置输出文件夹路径
output_folder_path = '/data1/sgy_mask2former/data_luding1/labels_png'

# 确保输出文件夹存在
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder_path):
    # 检查文件扩展名是否为tif
    if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
        # 构建完整的文件路径
        input_file_path = os.path.join(input_folder_path, filename)

        # 打开tif图片
        with Image.open(input_file_path) as img:
            # 构建输出文件的完整路径
            output_file_path = os.path.join(output_folder_path,
                                            filename.replace('.tif', '.png').replace('.tiff', '.png'))
            # 将图片转换为PNG格式并保存到输出文件夹
            img.save(output_file_path, 'PNG')

print("转换完成！PNG图片已保存到输出文件夹。")