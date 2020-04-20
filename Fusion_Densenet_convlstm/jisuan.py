import os
import re
save_dir = r'.\train_model\first'
file_dir = r'.\train_model\second'
# file_name = r'um_000000.png'

file_list = os.listdir(file_dir)
a = 0
b = 0
i = 0
# 制作len(file_list)张label
for file_name in file_list:
    number = re.sub("\D", '', file_name)
    a += int(number)/10 ** 14
    i += 1
    print(number)
print(i)
print(a/30)