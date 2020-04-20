import os
import numpy as np
from PIL import Image



def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

'''rgb转为灰度
    gray = 0.299*r + 0.587*g + 0.114*b
'''
# def rgb2gray(rgb):
#     return np.dot(rgb, [0.299, 0.587, 0.114])

'''将mask数据转换为灰度的二分类label'''

def lane_mask_to_label(file_dir, filename, save_dir):
    # 将图像灰度化
    file_path = os.path.join(file_dir, file_name)
    image = Image.open(file_path)
    # 将图像二值化为0、1，作为二分类的标签
#注释
    gray = np.array(image)
    c = np.bincount(gray.flat)
    return c
    # # print(np.argmax(c))
    # # index = np.where(c >= 1000)
    # # print(index)
    # # exit()
    # # print(len(np.where(gray == np.argmax(c))[0]))
    # # assert len(index[0]) == 2
    #
    # # 将灰度图中大于0的像素值赋值为1，等于0（黑色）的像素值不变，实现二值化为0、1
    # gray[gray < 105] = 0
    # gray[gray >= 105] = 255

    # 图片二值化

    # 自定义灰度界限，大于这个值为黑色，小于这个值为白色
    # threshold = 105
    #
    # table = []
    # for i in range(256):
    #     if i < threshold:
    #         table.append(0)
    #     else:
    #         table.append(1)
    #
    # # 图片二值化
    # photo = image.point(table, '1')
    #
    #
    #
    # # image = Image.fromarray(gray)
    # # 图片保存路径
    # mkdirs(save_dir)
    # save_path = os.path.join(save_dir, filename)
    # # image.save(save_path)
    # photo.save(save_path)

if __name__ == '__main__':
    save_dir = r'.\data_road\training\gt_image_two'
    file_dir = r'.\data_road\training\gt_image_2'
    file_name = r'um_000000.png'
    
    file_list = os.listdir(file_dir)
    a = 0
    b = 0
    # 制作len(file_list)张label
    for file_name in file_list:
        c = lane_mask_to_label(file_dir, file_name, save_dir)
        a += c[0]
        b += c[1]
    # print(a, b)
    # print(a//b)

















    
