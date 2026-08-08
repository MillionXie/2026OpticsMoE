import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 读取tif格式图像
input_file = 'images/Image_31.tif'  # 替换为你的tif文件路径
image = Image.open(input_file)

# 转换为numpy数组
image_array = np.array(image)
print(image_array.shape)

# # 保存为npy格式
# output_npy = 'output_image.npy'
# np.save(output_npy, image_array)

# 展示图像
plt.imshow(image_array, cmap='gray')  # 如果是彩色图像，去掉cmap参数
plt.title('TIF Image Display')
plt.axis('off')
plt.show()


