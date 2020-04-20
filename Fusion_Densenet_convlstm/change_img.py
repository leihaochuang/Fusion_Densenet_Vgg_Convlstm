from PIL import Image


def produceImage(file_in, width, height, file_out):
    image = Image.open(file_in)
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    # resized_image = resized_image.convert('L')
    resized_image.save(file_out)


if __name__ == '__main__':
    gt_name = './data_road/ADI/'
    png_gt_name = "um_00000"
    png_gt_name2 = "um_0000"
    n = 1/3
    width = int(1242 * n)
    height = int(375 * n)
    for i in range(95):
        if i < 10:
            gt = gt_name + png_gt_name
        else:
            gt = gt_name + png_gt_name2
        file_in = gt + str(i) + ".png"
        file_out = gt + str(i) + ".png"
        produceImage(file_in, width, height, file_out)
    png_gt_name = "umm_00000"
    png_gt_name2 = "umm_0000"
    for i in range(96):
        if i < 10:
            gt = gt_name + png_gt_name
        else:
            gt = gt_name + png_gt_name2
        file_in = gt + str(i) + ".png"
        file_out = gt + str(i) + ".png"
        produceImage(file_in, width, height, file_out)
    png_gt_name = "uu_00000"
    png_gt_name2 = "uu_0000"
    for i in range(98):
        if i < 10:
            gt = gt_name + png_gt_name
        else:
            gt = gt_name + png_gt_name2
        file_in = gt + str(i) + ".png"
        file_out = gt + str(i) + ".png"
        produceImage(file_in, width, height, file_out)
