import numpy as np
import cv2
import rasterio

# 输入和输出路径
tif_path = r"/DAY1/2019_1101_nofire_B2348_B12_10m_roi.tif"
png_path = r"/DAY1/2019_1101_nofire_B2348_B12_10m_roi.png"

# 读取TIFF文件
with rasterio.open(tif_path) as src:
    # 读取所有波段数据
    data = src.read()

    # 亮度调整参数
    GAMMA = 1.5  # 伽马值 >1 提亮图像，<1 变暗图像
    LOWER_PERCENTILE = 2  # 低分位数 (排除2%的暗部像素)
    UPPER_PERCENTILE = 98  # 高分位数 (排除2%的亮部像素)

    # 多波段图像处理 (RGB)
    if data.shape[0] >= 3:
        rgb_data = data[:3, :, :]  # 取前三个波段

        # 1. 分位数拉伸 (排除极端值)
        stretched = np.zeros_like(rgb_data, dtype=np.float32)
        for i in range(3):
            band = rgb_data[i]
            # 计算2%和98%分位数
            valid_pixels = band[~np.isnan(band)]
            lower = np.percentile(valid_pixels, LOWER_PERCENTILE)
            upper = np.percentile(valid_pixels, UPPER_PERCENTILE)

            # 拉伸到0-1范围
            band_stretched = (band - lower) / (upper - lower)
            band_stretched = np.clip(band_stretched, 0, 1)

            # 2. 伽马校正 (提升亮度)
            band_stretched = np.power(band_stretched, 1 / GAMMA)

            stretched[i] = band_stretched

        # 转换为8位图像 (0-255)
        image_8bit = (stretched * 255).astype(np.uint8)

        # 转换维度: [通道, 高, 宽] -> [高, 宽, 通道]
        image_8bit = np.transpose(image_8bit, (1, 2, 0))

        # 3. 直方图均衡化 (增强对比度)
        # 分别对每个通道进行均衡化
        for i in range(3):
            image_8bit[:, :, i] = cv2.equalizeHist(image_8bit[:, :, i])

        # 转换通道顺序 BGR (OpenCV默认要求)
        image_8bit = cv2.cvtColor(image_8bit, cv2.COLOR_RGB2BGR)

    # 单波段图像处理
    else:
        band = data[0, :, :]  # 取第一个波段

        # 分位数拉伸
        valid_pixels = band[~np.isnan(band)]
        lower = np.percentile(valid_pixels, LOWER_PERCENTILE)
        upper = np.percentile(valid_pixels, UPPER_PERCENTILE)

        # 拉伸到0-1范围
        stretched = (band - lower) / (upper - lower)
        stretched = np.clip(stretched, 0, 1)

        # 伽马校正
        stretched = np.power(stretched, 1 / GAMMA)

        # 转换为8位图像
        image_8bit = (stretched * 255).astype(np.uint8)

        # 直方图均衡化
        image_8bit = cv2.equalizeHist(image_8bit)

# 保存为PNG文件
cv2.imwrite(png_path, image_8bit)

print(f"转换成功！已保存至: {png_path}")
print(f"使用的亮度调整参数: 伽马值={GAMMA}, 分位数范围={LOWER_PERCENTILE}%-{UPPER_PERCENTILE}%")

from matplotlib import pyplot as plt
plt.imshow(cv2.cvtColor(image_8bit, cv2.COLOR_BGR2RGB))  # 彩色图像
# 或 plt.imshow(image_8bit, cmap='gray')  # 灰度图像
plt.title("Converted Image")
plt.show()