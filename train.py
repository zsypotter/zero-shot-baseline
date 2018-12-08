import torch
import numpy as np
import data_loader
import model
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from utils import *

parser = argparse.ArgumentParser()

parser.add_argument("--img_folder", type=str, default="/home/disk3/zhousiyu/dataset/CUB_200_2011/images")
parser.add_argument("--att_path", type=str, default="/home/disk3/zhousiyu/dataset/CUB_200_2011/attributes/class_attribute_labels_continuous.txt")
parser.add_argument("--train_img_path", type=str, default="/home/disk3/zhousiyu/dataset/CUB_200_2011/zeroshot/train.txt")
parser.add_argument("--train_cls_path", type=str, default="/home/disk3/zhousiyu/dataset/CUB_200_2011/zeroshot/train_classes.txt")
parser.add_argument("--testR_img_path", type=str, default="/home/disk3/zhousiyu/dataset/CUB_200_2011/zeroshot/testRecg.txt")
parser.add_argument("--testZ_img_path", type=str, default="/home/disk3/zhousiyu/dataset/CUB_200_2011/zeroshot/testZS.txt")
parser.add_argument("--test_cls_path", type=str, default="/home/disk3/zhousiyu/dataset/CUB_200_2011/zeroshot/test_classes.txt")
parser.add_argument("--finetune", type=bool, default=False)
parser.add_argument("--mse", type=bool, default=False)
parser.add_argument("--train_class_num", type=int, default=150)
parser.add_argument("--test_class_num", type=int, default=50)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--att_size", type=int, default=312)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--m_lr", type=float, default=0.001)
parser.add_argument("--b_lr", type=float, default=0.00001)

args = parser.parse_args()

data_transforms = transforms.Compose([
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

trainset = data_loader.customData(args.img_folder, args.train_img_path, args.train_cls_path, args.train_class_num, data_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testRset = data_loader.customData(args.img_folder, args.testR_img_path, args.train_cls_path, args.train_class_num, data_transforms)
testRloader = torch.utils.data.DataLoader(testRset, batch_size=1, shuffle=True, num_workers=2)
testZset = data_loader.customData(args.img_folder, args.testZ_img_path, args.test_cls_path, args.test_class_num, data_transforms)
testZloader = torch.utils.data.DataLoader(testZset, batch_size=1, shuffle=True, num_workers=2)

bk_net = model.Backbone()
mp_net = model.Mapping(args.att_size)

set_parameter_requires_grad(bk_net, args.finetune)
set_parameter_requires_grad(mp_net, True)

train_cls_file = open(args.train_cls_path)
test_cls_file = open(args.test_cls_path)
lines = train_cls_file.readlines()
train_cls_dict = [(int(line.split(' ')[0]) - 1) for line in lines]
lines = test_cls_file.readlines()
test_cls_dict = [(int(line.split(' ')[0]) - 1) for line in lines]

att_dict = np.loadtxt(args.att_path)
if att_dict.max() > 1.:
    att_dict /= 100.
att_mean = att_dict[train_cls_dict, :].mean(axis=0)
for i in range(args.att_size):
    att_dict[att_dict[:, i] < 0, i] = att_mean[i]
for i in range(args.att_size):
    att_dict[:, i] = att_dict[:, i] - att_mean[i] + 0.5

train_att_dict = att_dict[train_cls_dict, :]
test_att_dict = att_dict[test_cls_dict, :]
train_att_dict = torch.Tensor(train_att_dict).cuda()
test_att_dict = torch.Tensor(test_att_dict).cuda()

if args.mse:
    criterion = nn.MSELoss()
else:
    criterion = nn.CrossEntropyLoss()

if args.finetune:
    optimizer = optim.Adam([
                    {'params': bk_net.parameters(), 'lr': args.b_lr},
                    {'params': mp_net.parameters(), 'weight_decay': args.weight_decay},
                    ], lr=args.m_lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
else:
    optimizer = optim.Adam([
                    {'params': mp_net.parameters(), 'weight_decay': args.weight_decay},
                    ], lr=args.m_lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)

bk_net = bk_net.cuda()
mp_net = mp_net.cuda()

for epoch in range(1000):
      train_ac = 0
      if args.finetune:
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

          similarity = torch.mm(fake_att, train_att_dict.t())
          if args.mse:
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
 
          similarity = torch.mm(fake_att, test_att_dict.t())
          predict = torch.argmax(similarity, 1)
          testZ_ac += (predict == label).sum().item()
      testZ_ac = testZ_ac / len(testZloader.dataset)

      print(epoch, train_ac, testR_ac, testZ_ac)