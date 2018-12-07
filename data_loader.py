import torch
import torch.utils.data
import PIL
import os
import numpy as np

def My_loader(path):
	return PIL.Image.open(path).convert('RGB')

class customData(torch.utils.data.Dataset):
    def __init__(self, img_folder, img_path, cls_path, class_num, data_transforms=None, loader=My_loader):
        with open(cls_path) as cls_file:
            lines = cls_file.readlines()
            self.cls_dict = [(int(line.split(' ')[0]) - 1) for line in lines]

        with open(img_path) as img_file:
            lines = img_file.readlines()
            self.img_name = [os.path.join(img_folder, line[:-1]) for line in lines]
            self.label = [(int(line.split('.')[0]) - 1) for line in lines]
    
        self.class_num = class_num
        self.data_transforms = data_transforms
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        img = self.loader(img_name)
        label = -1
        for i in range(self.class_num):
            if self.cls_dict[i] == self.label[item]:
                label = i
                break

        if self.data_transforms is not None:
            img = self.data_transforms(img)

        return img, label