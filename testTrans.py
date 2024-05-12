import cv2
import numpy as np

# 读取源图像
src_img = cv2.imread('bus.jpg')

# 源图像的四个点
src_points = np.float32([[0, 0], [src_img.shape[1], 0], [src_img.shape[1], src_img.shape[0]], [0, src_img.shape[0]]])

# 目标图像的四个点，这里简单地将其设为与源图像一致
dst_points = np.float32([[0, 0], [src_img.shape[1], 0], [src_img.shape[1], src_img.shape[0]], [0, src_img.shape[0]]])

# 计算透视变换矩阵
perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# 进行透视变换
transformed_img = cv2.warpPerspective(src_img, perspective_matrix, (src_img.shape[1], src_img.shape[0]))

# 保存结果
cv2.imwrite('transformed_bus.jpg', transformed_img)
