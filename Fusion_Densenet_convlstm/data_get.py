import xlrd
import math


file_name = 'E:/dataset/kitti/data_road/training/image_2/'
gt_name = 'E:/dataset/kitti/data_road/training/gt_image_2/'
file_name_cloud = 'E:/dataset/kitti/data_road/point_cloud_train/'
png_name = "um_00000"
png_name2 = "um_0000"
png_gt_name = "um_road_00000"
png_gt_name2 = "um_road_0000"
# 获取单帧数据




# 获取不同数据index
data = xlrd.open_workbook(file_name + "index.xlsx")  # 打开excel
table = data.sheet_by_name("Sheet1")  # 读sheet
nrows = table.nrows  # 获得行数
result = []
result2 = []
result3 = []
for i in range(1, nrows):  #

    rows = table.row_values(i)  # 行的数据放在数组里
    sku = rows[0]
    keyword = rows[1]
    start2 = rows[2]
    end = rows[3]
    uu_start = rows[4]
    uu_end = rows[5]
    result.append([sku, keyword])
    result2.append([start2, end])
    result3.append([uu_start, uu_end])
# 获取图片数据
um_index = result[:-4]
umm_index = result2
uu_index = result3[:-8]
# um_data = []

listText = open(file_name+'image_train_index.txt', 'w+')
number_um = 0

for item in um_index:
    number = item[1] - item[0]-1
    if item[1] < 10:
        file = file_name + png_name
        gt = gt_name + png_gt_name
    else:
        file = file_name + png_name2
        gt = gt_name + png_gt_name2
    for i in range(int(number)):
        listText.write(file+str(int(item[1]-i-2)) + ".png" + " ")
        listText.write(file+str(int(item[1]-i-1)) + ".png" + " ")
        listText.write(file+str(int(item[1]-i)) + ".png" + " ")
        listText.write(gt+str(int(item[1]-i)) + ".png" + "\n")
        number_um += 1
listText = open(file_name +'image_train_index.txt', 'a')
for item in umm_index:
    number = item[1] - item[0]-1
    if item[1] < 10:
        file = file_name + png_name
        gt = gt_name + png_gt_name
    else:
        file = file_name + png_name2
        gt = gt_name + png_gt_name2
    for i in range(int(number)):
        listText.write(file+str(int(item[1]-i-2)) + ".png" + " ")
        listText.write(file+str(int(item[1]-i-1)) + ".png" + " ")
        listText.write(file+str(int(item[1]-i)) + ".png" + " ")
        listText.write(gt + str(int(item[1] - i)) + ".png" + "\n")
        number_um += 1
listText = open(file_name +'image_train_index.txt', 'a')
for item in uu_index:
    number = item[1] - item[0]-1
    if item[1] < 10:
        file = file_name + png_name
        gt = gt_name + png_gt_name
    else:
        file = file_name + png_name2
        gt = gt_name + png_gt_name2
    for i in range(int(number)):
        listText.write(file+str(int(item[1]-i-2)) + ".png" + " ")
        listText.write(file+str(int(item[1]-i-1)) + ".png" + " ")
        listText.write(file+str(int(item[1]-i)) + ".png" + " ")
        listText.write(gt + str(int(item[1] - i)) + ".png" + "\n")
        number_um += 1
print(number_um)

# 点云数据准备
listText = open(file_name_cloud + 'point_cloud_image_train_index.txt', 'w+')
number_um = 0
for item in um_index:
    number = item[1] - item[0]-1
    if item[1] < 10:
        file = file_name_cloud + png_name
    else:
        file = file_name_cloud + png_name2
    for i in range(int(number)):
        listText.write(file+str(int(item[1]-i-2)) + ".png" + " ")
        listText.write(file+str(int(item[1]-i-1)) + ".png" + " ")
        listText.write(file+str(int(item[1]-i)) + ".png" + "\n")
        number_um += 1
listText = open(file_name_cloud + 'point_cloud_image_train_index.txt', 'a')
for item in umm_index:
    number = item[1] - item[0]-1
    if item[1] < 10:
        file = file_name_cloud + png_name
    else:
        file = file_name_cloud + png_name2
    for i in range(int(number)):
        listText.write(file+str(int(item[1]-i-2)) + ".png" + " ")
        listText.write(file+str(int(item[1]-i-1)) + ".png" + " ")
        listText.write(file+str(int(item[1]-i)) + ".png" + "\n ")
        number_um += 1
listText = open(file_name_cloud + 'point_cloud_image_train_index.txt', 'a')
for item in uu_index:
    number = item[1] - item[0]-1
    if item[1] < 10:
        file = file_name_cloud + png_name
    else:
        file = file_name_cloud + png_name2
    for i in range(int(number)):
        listText.write(file+str(int(item[1]-i-2)) + ".png" + " ")
        listText.write(file+str(int(item[1]-i-1)) + ".png" + " ")
        listText.write(file+str(int(item[1]-i)) + ".png" + "\n ")
        number_um += 1






















