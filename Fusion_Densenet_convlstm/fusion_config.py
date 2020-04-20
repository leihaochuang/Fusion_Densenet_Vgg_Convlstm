import argparse

# globel param
# dataset setting
img_width = 1242/3
img_height = 375/3
img_channel = 3
label_width = 1242/3
label_height = 375/3
label_channel = 1
data_loader_numworkers = 0
class_num = 2
growth_rate = 32

# path
image_train_path = "./data/single_image_train_index0414.txt"
point_cloud_image_train_path = "./data/ADI_point_cloud_image_train_index0414.txt"
# point_cloud_image_train_path = "./data/single_point_cloud_image_train_index_train.txt"
val_path = "./data/single_image_test_index0414.txt"
file_path_Point_cloud_image = "./data/ADI_point_cloud_image_test_index0414.txt"
# image_test_path = "./data/image_test_index.txt"
# point_cloud_image_test_path = "./data/image_test_index.txt"
image_test_path = "./data/single_image_test_index0414.txt"
point_cloud_image_test_path = "./data/ADI_point_cloud_image_test_index0414.txt"
save_path = "./result/"
pretrained_path = './train_model/6and7/95.58933030216917.pth'


# weight
class_weight = [0.25, 1]
# class_weight = [1, 0.004]


def args_setting():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch UNet-ConvLSTM')
    parser.add_argument('--model', type=str, default='Densenet',
                        help='(Densenet||Densenet-Vgg)')
    parser.add_argument('--layers', type=int, default=121,
                        help='(121|161|169|201)'),
    parser.add_argument('--pretrained', default=True,
                        help='use pretrained densenet model'),
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    return args
