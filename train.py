import torch
import os
import numpy as np
import data_loader
import model
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from tensorboardX import SummaryWriter

from utils import *

# h-parameter
parser = argparse.ArgumentParser()
parser.add_argument("--img_folder", type=str, default="/home/disk3/zhousiyu/dataset/CUB_200_2011/images")
parser.add_argument("--dataset_root", type=str, default="/home/disk3/zhousiyu/dataset/CUB_200_2011/zeroshot")
parser.add_argument("--finetune", type=bool, default=False)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--weight_decay", type=float, default=0.0005)
parser.add_argument("--m_lr", type=float, default=0.001)
parser.add_argument("--b_lr", type=float, default=0.0001)
parser.add_argument("--modify_att", type=bool, default=True)
args = parser.parse_args()

# set path
att_path = os.path.join(args.dataset_root, "class_attribute_labels_continuous.txt")
train_img_path = os.path.join(args.dataset_root, "train.txt")
train_cls_path = os.path.join(args.dataset_root, "train_classes.txt")
testR_img_path = os.path.join(args.dataset_root, "testRecg.txt")
testZ_img_path = os.path.join(args.dataset_root, "testZS.txt")
test_cls_path = os.path.join(args.dataset_root, "test_classes.txt")

# att-preprocess
att_dict = np.loadtxt(att_path)
train_cls_file = open(train_cls_path)
test_cls_file = open(test_cls_path)
lines = train_cls_file.readlines()
train_cls_dict = [(int(line.split(' ')[0]) - 1) for line in lines]
lines = test_cls_file.readlines()
test_cls_dict = [(int(line.split(' ')[0]) - 1) for line in lines]

train_class_num = len(train_cls_dict)
test_class_num = len(test_cls_dict)
num_class, att_size = att_dict.shape

if args.modify_att:
    if att_dict.max() > 1.:
        att_dict /= 100.
    att_mean = att_dict[train_cls_dict, :].mean(axis=0)
    for i in range(att_size):
        att_dict[att_dict[:, i] < 0, i] = att_mean[i]
    for i in range(att_size):
        att_dict[:, i] = att_dict[:, i] - att_mean[i] + 0.5

train_att_dict = att_dict[train_cls_dict, :]
test_att_dict = att_dict[test_cls_dict, :]

# img-preprocess
data_transforms = transforms.Compose([
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

# load data
trainset = data_loader.customData(args.img_folder, train_img_path, train_cls_path, train_class_num, data_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testRset = data_loader.customData(args.img_folder, testR_img_path, train_cls_path, train_class_num, data_transforms)
testRloader = torch.utils.data.DataLoader(testRset, batch_size=1, shuffle=True, num_workers=2)
testZset = data_loader.customData(args.img_folder, testZ_img_path, test_cls_path, test_class_num, data_transforms)
testZloader = torch.utils.data.DataLoader(testZset, batch_size=1, shuffle=True, num_workers=2)

# load model
bk_net = model.Backbone()
mp_net = model.Mapping(att_size)
set_parameter_requires_grad(bk_net, args.finetune)
set_parameter_requires_grad(mp_net, True)

# define loss
criterion = nn.CrossEntropyLoss()

# define opt
if args.finetune:
    optimizer = optim.Adam([
                    {'params': bk_net.parameters(), 'lr': args.b_lr},
                    {'params': mp_net.parameters(), 'weight_decay': args.weight_decay},
                    ], lr=args.m_lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
else:
    optimizer = optim.Adam([
                    {'params': mp_net.parameters(), 'weight_decay': args.weight_decay},
                    ], lr=args.m_lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)

# move tensor to cuda
train_att_dict = torch.Tensor(train_att_dict).cuda()
test_att_dict = torch.Tensor(test_att_dict).cuda()
bk_net = bk_net.cuda()
mp_net = mp_net.cuda()

# main step
writer = SummaryWriter()
for epoch in range(1000):

    # train step
    train_ac = 0
    if args.finetune:
        bk_net.train()
    else:
        bk_net.eval()
        mp_net.train()

    for i, data in enumerate(trainloader, 0):
        img, label = data
        img = img.cuda()
        label = label.cuda()

        optimizer.zero_grad()

        feature = bk_net(img)
        fake_att = mp_net(feature)

        similarity = torch.mm(fake_att, train_att_dict.t())
        loss = criterion(similarity, label)

        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(epoch, i, loss.item())

        predict = torch.argmax(similarity, 1)
        train_ac += (predict == label).sum().item()
    train_ac = train_ac / len(trainloader.dataset)
       
    # testR step
    bk_net.eval()
    mp_net.eval()
    predict_list = []
    label_list = []
    for i, data in enumerate(testRloader, 0):
        img, label = data
        img = img.cuda()
        label = label.cuda()

        feature = bk_net(img)
        fake_att = mp_net(feature)
        similarity = torch.mm(fake_att, train_att_dict.t())
        predict = torch.argmax(similarity, 1)
        predict_list.append(predict.view(-1).item())
        label_list.append(label.view(-1).item())
    predict_list = np.array(predict_list)
    label_list = np.array(label_list)
    testR_ac = np.array([(predict_list[label_list == l] == l).mean() for l in range(train_class_num)]).mean()
     
    # testZ step
    predict_list = []
    label_list = []
    for i, data in enumerate(testZloader, 0):
        img, label = data
        img = img.cuda()
        label = label.cuda()

        feature = bk_net(img)
        fake_att = mp_net(feature)
 
        similarity = torch.mm(fake_att, test_att_dict.t())
        predict = torch.argmax(similarity, 1)
        predict_list.append(predict.view(-1).item())
        label_list.append(label.view(-1).item())
    predict_list = np.array(predict_list)
    label_list = np.array(label_list)
    testZ_ac = np.array([(predict_list[label_list == l] == l).mean() for l in range(test_class_num)]).mean()

    # print accurancy
    print(epoch, train_ac, testR_ac, testZ_ac)
    writer.add_scalar("train_ac", train_ac, epoch)
    writer.add_scalar("testR_ac", testR_ac, epoch)
    writer.add_scalar("testZ_ac", testZ_ac, epoch)
writer.close()