# -*- coding: utf-8 -*-
# @Author  : Lan Zhang
# @Time    : 2022/4/7 12:52
# @File    : regression_AVEC2014.py
# @Software: PyCharm
import math
import os
import argparse
import logging
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from my_dataset import MyDataSet
from LKCT import LKCT
from fintune_utils import read_split_data, train_one_epoch, evaluate
#os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./RMSE") is False:
        os.makedirs("./RMSE")
    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    if os.path.exists("./logs") is False:
        os.makedirs("./logs")
    logger = get_logger('./logs/pre.log')
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 10])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=nw,
                              collate_fn=train_dataset.collate_fn,
                              drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=nw,
                            collate_fn=val_dataset.collate_fn,
                            drop_last=True)

    model = LKCT(num_classes=1).to(device)
    # model = torch.nn.DataParallel(model, [0, 1, 2])
    print("The model has been on the cuda !!!")

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        # weights_dict = torch.load(args.weights, map_location=device)["model"]
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除有关分类类别的权重
        # for k in list(weights_dict.keys()):
            # if "stem" in k:
            #     del weights_dict['model'][k]
            # if "teacher" in k:
            #     del weights_dict['model'][k]
            # if "fc" in k:
            #     del weights_dict[k]
            # if "LK.stages.2" in k:
            #     del weights_dict[k]
            # elif "LK.stages.3" in k:
            #     del weights_dict[k]
#             if "head" in k:                                                                                            
#                 del weights_dict[k]
#             elif "fc.weight" in k:
#                 del weights_dict[k]
#             elif "fc.bias" in k:
#                 del weights_dict[k]
            
        model.load_state_dict(weights_dict, strict=False)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "stem" in name:
                para.requires_grad_(False)
            elif "LK.stages.0" in name:
                para.requires_grad_(False)
            elif "LK.stages.1" in name:
                para.requires_grad_(False)
            elif "LK.stages.2" in name:
                para.requires_grad_(False)
            # elif "fc.bias" in name:
                # para.requires_grad_(True)
            elif "LIT.patch_embed" in name:
                para.requires_grad_(False)
            elif "LIT.layers.0" in name:
                para.requires_grad_(False)
            elif "LIT.layers.1" in name:
                para.requires_grad_(False)
            elif "LIT.layers.2" in name:
                para.requires_grad_(False)
            # elif "LK" in name:
            #     para.requires_grad_(False)
            # elif "LIT" in name:
            #     para.requires_grad_(False)
            else:
                para.requires_grad_(True)
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    # T_max = 100
    optimizer = optim.AdamW(pg, lr=args.lr, eps=1e-8, weight_decay=0.05)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=1e-8, last_epoch=- 1, verbose=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 3, 5, 6, 9], gamma=0.5)
    best_loss = 100.0
    for epoch in range(args.epochs):
        # train
        train_loss = train_one_epoch(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     device=device,
                                     epoch=epoch)

        # validate
        val_loss = evaluate(model=model,
                            data_loader=val_loader,
                            device=device,
                            epoch=epoch)

        logger.info('Epoch:[{}]\t '
                    'train_loss={:.4f}\t '
                    'validation_loss={:.4f}'.format(epoch + 1, train_loss, val_loss))
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     torch.save(model.state_dict(), "./RMSE_mobile/model-best.pth".format(best_loss))
        torch.save(model.state_dict(), "./RMSE/model-{}.pth".format(epoch+1))
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)

    # TODO 数据集所在根目录
    parser.add_argument('--data-path', type=str,
                        default="./train")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./RMSE/testt.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
