import torch
import numpy as np
import data_loader
import model
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

from utils import *

img_folder = "/home/disk3/zhousiyu/dataset/CUB_200_2011/images"
att_path = "/home/disk3/zhousiyu/dataset/CUB_200_2011/attributes/class_attribute_labels_continuous.txt"
train_img_path = "/home/disk3/zhousiyu/dataset/CUB_200_2011/zeroshot/train.txt"
train_cls_path = "/home/disk3/zhousiyu/dataset/CUB_200_2011/zeroshot/train_classes.txt"
testR_img_path = "/home/disk3/zhousiyu/dataset/CUB_200_2011/zeroshot/testRecg.txt"
testZ_img_path = "/home/disk3/zhousiyu/dataset/CUB_200_2011/zeroshot/testZS.txt"
test_cls_path = "/home/disk3/zhousiyu/dataset/CUB_200_2011/zeroshot/test_classes.txt"
finetune = False
mse = False

train_class_num = 150
test_class_num = 50
img_size = 224
att_size = 312
batch_size = 64
weight_decay = 0.01

data_transforms = transforms.Compose([
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

trainset = data_loader.customData(img_folder, train_img_path, train_cls_path, train_class_num, data_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testRset = data_loader.customData(img_folder, testR_img_path, train_cls_path, train_class_num, data_transforms)
testRloader = torch.utils.data.DataLoader(testRset, batch_size=1, shuffle=True, num_workers=2)
testZset = data_loader.customData(img_folder, testZ_img_path, test_cls_path, test_class_num, data_transforms)
testZloader = torch.utils.data.DataLoader(testZset, batch_size=1, shuffle=True, num_workers=2)

bk_net = model.Backbone()
mp_net = model.Mapping(att_size)

set_parameter_requires_grad(bk_net, finetune)
set_parameter_requires_grad(mp_net, True)

train_cls_file = open(train_cls_path)
test_cls_file = open(test_cls_path)
lines = train_cls_file.readlines()
train_cls_dict = [(int(line.split(' ')[0]) - 1) for line in lines]
lines = test_cls_file.readlines()
test_cls_dict = [(int(line.split(' ')[0]) - 1) for line in lines]

att_dict = np.loadtxt(att_path)
if att_dict.max() > 1.:
    att_dict /= 100.
att_mean = att_dict[train_cls_dict, :].mean(axis=0)
for i in range(att_size):
    att_dict[att_dict[:, i] < 0, i] = att_mean[i]
for i in range(att_size):
    att_dict[:, i] = att_dict[:, i] - att_mean[i] + 0.5

train_att_dict = att_dict[train_cls_dict, :]
test_att_dict = att_dict[test_cls_dict, :]
train_att_dict = torch.Tensor(train_att_dict).cuda()
test_att_dict = torch.Tensor(test_att_dict).cuda()

if mse:
    criterion = nn.MSELoss()
else:
    criterion = nn.CrossEntropyLoss()

if finetune:
    optimizer = optim.Adam([
                    {'params': bk_net.parameters(), 'lr': 0.00001},
                    {'params': mp_net.parameters(), 'weight_decay': weight_decay},
                    ], lr=0.0001, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
else:
    optimizer = optim.Adam([
                    {'params': mp_net.parameters(), 'weight_decay': weight_decay},
                    ], lr=0.0001, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)

bk_net = bk_net.cuda()
mp_net = mp_net.cuda()

for epoch in range(1000):
      train_ac = 0
      if finetune:
        bk_net.train()
      else:
        bk_net.eval()
      mp_net.train()

      for i, data in enumerate(trainloader, 0):
          img, label = data
          att = train_att_dict[label]
          img = img.cuda()
          label = label.cuda()
          att = att.cuda()

          optimizer.zero_grad()

          feature = bk_net(img)
          fake_att = mp_net(feature)

          '''similarity_up = torch.mm(fake_att, train_att_dict.t())
          similarity_down = torch.mm(torch.norm(fake_att, dim=1, keepdim=True), torch.norm(train_att_dict, dim=1, keepdim=True).t())
          similarity = similarity_up / similarity_down'''
          similarity = torch.mm(fake_att, train_att_dict.t())
          if mse:
              loss = criterion(fake_att, att)
          else:
              loss = criterion(similarity, label)

          loss.backward()
          optimizer.step()
          if i % 10 == 0:
            print(epoch, i, loss.item())

          predict = torch.argmax(similarity, 1)
          train_ac += (predict == label).sum().item()
      train_ac = train_ac / len(trainloader.dataset)

      testR_ac = 0
      bk_net.eval()
      mp_net.eval()
      for i, data in enumerate(testRloader, 0):
          img, label = data
          img = img.cuda()
          label = label.cuda()

          feature = bk_net(img)
          fake_att = mp_net(feature)

          '''similarity_up = torch.mm(fake_att, train_att_dict.t())
          similarity_down = torch.mm(torch.norm(fake_att, dim=1, keepdim=True), torch.norm(train_att_dict, dim=1, keepdim=True).t())
          similarity = similarity_up / similarity_down'''
          similarity = torch.mm(fake_att, train_att_dict.t())
          predict = torch.argmax(similarity, 1)
          testR_ac += (predict == label).sum().item()
      testR_ac = testR_ac / len(testRloader.dataset)

      testZ_ac = 0
      for i, data in enumerate(testZloader, 0):
          img, label = data
          img = img.cuda()
          label = label.cuda()

          feature = bk_net(img)
          fake_att = mp_net(feature)
 
          '''similarity_up = torch.mm(fake_att, test_att_dict.t())
          similarity_down = torch.mm(torch.norm(fake_att, dim=1, keepdim=True), torch.norm(test_att_dict, dim=1, keepdim=True).t())
          similarity = similarity_up / similarity_down'''
          similarity = torch.mm(fake_att, test_att_dict.t())
          predict = torch.argmax(similarity, 1)
          testZ_ac += (predict == label).sum().item()
      testZ_ac = testZ_ac / len(testZloader.dataset)

      print(epoch, train_ac, testR_ac, testZ_ac)