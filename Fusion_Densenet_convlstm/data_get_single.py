import random
file_name = 'E:/dataset/kitti/data_road/training/image_2/'
gt_name = 'E:/dataset/kitti/data_road/training/gt_image_2/'
file_name_cloud = 'E:/dataset/ADI/'
png_name = "um_00000"
png_name2 = "um_0000"
png_gt_name = "um_road_00000"
png_gt_name2 = "um_road_0000"
# 获取单帧数据


# 获取不同数据index 图片数据
number = []
for i in range(10):
    # 3.生成随机数
    num = random.randint(0, 94)
    # 4.添加到列表中
    number.append(num)
# print(number)
# exit()

um_index = [i for i in range(95)]
umm_index = [i for i in range(96)]
uu_index = [i for i in range(98)]
# 去除测试集

for i in range(len(um_index)):
    for j in um_index:
        if j in number:
            um_index.remove(j)
            umm_index.remove(j)
            uu_index.remove(j)

# print(um_index)
# # print(umm_index)
# # print(uu_index)
# # print(number)
# # exit()


listText = open(file_name+'single_image_train_index0414.txt', 'w+')
for i in um_index:
    if i < 10:
        file = file_name + png_name
        gt = gt_name + png_gt_name
    else:
        file = file_name + png_name2
        gt = gt_name + png_gt_name2
    listText.write(file+str(i) + ".png" + " ")
    listText.write(gt+str(i) + ".png" + "\n")
png_name = "umm_00000"
png_name2 = "umm_0000"
png_gt_name = "umm_road_00000"
png_gt_name2 = "umm_road_0000"

for i in umm_index:
    if i < 10:
        file = file_name + png_name
        gt = gt_name + png_gt_name
    else:
        file = file_name + png_name2
        gt = gt_name + png_gt_name2
    listText.write(file+str(i) + ".png" + " ")
    listText.write(gt+str(i) + ".png" + "\n")
png_name = "uu_00000"
png_name2 = "uu_0000"
png_gt_name = "uu_road_00000"
png_gt_name2 = "uu_road_0000"

for i in uu_index:
    if i < 10:
        file = file_name + png_name
        gt = gt_name + png_gt_name
    else:
        file = file_name + png_name2
        gt = gt_name + png_gt_name2
    listText.write(file+str(i) + ".png" + " ")
    listText.write(gt+str(i) + ".png" + "\n")

# 点云数据准备
listText = open(file_name_cloud + 'ADI_point_cloud_image_train_index0414.txt', 'w+')
png_name = "um_00000"
png_name2 = "um_0000"
for i in um_index:
    if i < 10:
        file = file_name_cloud + png_name
    else:
        file = file_name_cloud + png_name2
    listText.write(file+str(i) + ".png" + "\n")
png_name = "umm_00000"
png_name2 = "umm_0000"
for i in umm_index:
    if i < 10:
        file = file_name_cloud + png_name
    else:
        file = file_name_cloud + png_name2
    listText.write(file+str(i) + ".png" + "\n")
png_name = "uu_00000"
png_name2 = "uu_0000"
for i in uu_index:
    if i < 10:
        file = file_name_cloud + png_name
    else:
        file = file_name_cloud + png_name2
    listText.write(file+str(i) + ".png" + "\n")

# 测试集
listText = open(file_name+'single_image_test_index0414.txt', 'w+')
for i in number:
    if i < 10:
        file = file_name + png_name
        gt = gt_name + png_gt_name
    else:
        file = file_name + png_name2
        gt = gt_name + png_gt_name2
    listText.write(file+str(i) + ".png" + " ")
    listText.write(gt+str(i) + ".png" + "\n")
png_name = "umm_00000"
png_name2 = "umm_0000"
png_gt_name = "umm_road_00000"
png_gt_name2 = "umm_road_0000"

for i in number:
    if i < 10:
        file = file_name + png_name
        gt = gt_name + png_gt_name
    else:
        file = file_name + png_name2
        gt = gt_name + png_gt_name2
    listText.write(file+str(i) + ".png" + " ")
    listText.write(gt+str(i) + ".png" + "\n")
png_name = "uu_00000"
png_name2 = "uu_0000"
png_gt_name = "uu_road_00000"
png_gt_name2 = "uu_road_0000"

for i in number:
    if i < 10:
        file = file_name + png_name
        gt = gt_name + png_gt_name
    else:
        file = file_name + png_name2
        gt = gt_name + png_gt_name2
    listText.write(file+str(i) + ".png" + " ")
    listText.write(gt+str(i) + ".png" + "\n")

# 点云数据准备
listText = open(file_name_cloud + 'ADI_point_cloud_image_test_index0414.txt', 'w+')
png_name = "um_00000"
png_name2 = "um_0000"
for i in number:
    if i < 10:
        file = file_name_cloud + png_name
    else:
        file = file_name_cloud + png_name2
    listText.write(file+str(i) + ".png" + "\n")
png_name = "umm_00000"
png_name2 = "umm_0000"
for i in number:
    if i < 10:
        file = file_name_cloud + png_name
    else:
        file = file_name_cloud + png_name2
    listText.write(file+str(i) + ".png" + "\n")
png_name = "uu_00000"
png_name2 = "uu_0000"
for i in number:
    if i < 10:
        file = file_name_cloud + png_name
    else:
        file = file_name_cloud + png_name2
    listText.write(file+str(i) + ".png" + "\n")



















