from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np


def readTxt(file_path):
    img_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip().split()
            img_list.append(item)
    file_to_read.close()
    return img_list


class RoadSequenceDatasetList(Dataset):

    def __init__(self, file_path, file_path_Point_cloud_image, transforms):

        self.img_list = readTxt(file_path)
        self.img_list_Point_cloud_image = readTxt(file_path_Point_cloud_image)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        img_path_list_Point_cloud_image = self.img_list_Point_cloud_image[idx]
        data = []
        data2 = []
        # print(len(img_path_list))
        # print(len(img_path_list_Point_cloud_image))
        for i in range(1):
            data.append(torch.unsqueeze(self.transforms(Image.open(img_path_list[i])), dim=0))
            data2.append(torch.unsqueeze(self.transforms(Image.open(img_path_list_Point_cloud_image[i])), dim=0))
        # for i in range(4):
        #     print(data[i])
        #     print(data2[i])

        data = torch.cat(data, 0)

        data2 = torch.cat(data2, 0)

        label = Image.open(img_path_list[1])
        label = torch.squeeze(self.transforms(label))

        sample = {'data': data, 'data2': data2, 'label': label}
        return sample


class RoadSequenceDataset(Dataset):

    def __init__(self, file_path, transforms):

        self.img_list = readTxt(file_path)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        data = Image.open(img_path_list[0])
        label = Image.open(img_path_list[1])
        data = self.transforms(data)
        label = torch.squeeze(self.transforms(label))
        sample = {'data': data, 'label': label}
        return sample


