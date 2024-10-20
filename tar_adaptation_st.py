import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import time
from datetime import datetime
import  shutil
from autoaugment import ImageNetPolicy
from utils import *
 


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=-1)
    return entropy


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    # decay = 1
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"]  * decay
        param_group["weight_decay"] = 1e-3
        param_group["momentum"] = 0.9
        param_group["nesterov"] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def image_aug(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    # aug_policy = ImageNetPolicy()
    # return aug_policy
    )


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsize = len(txt_src)
    tr_size = int(0.9 * dsize)
    # print(dsize, tr_size, dsize - tr_size)
    _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    tr_txt = txt_src

    '''
    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(
        dsets["source_tr"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(
        dsets["source_te"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    '''

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(
        dsets["target"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["aug"] = ImageList_idx(txt_tar, transform=image_aug())
    dset_loaders["aug"] = DataLoader(
        dsets["aug"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(
        dsets["test"],
        batch_size=train_bs * 3,
        shuffle=False,
        num_workers=args.worker,
        drop_last=False,
    )

    return dset_loaders



def hyper_decay(x, beta=-2, alpha=1):
    weight = (1 + 10 * x) ** (-beta) * alpha
    return weight

'''
def train_target(args):
    dset_loaders = data_load(args)

    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck(
        type=args.classifier,
        feature_dim=netF.in_features,
        bottleneck_dim=args.bottleneck,
    ).cuda()
    netC = network.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck
    ).cuda()

    # modelpath = args.output_dir_src + "/source_F.pt"
    modelpath = "/home/bowei/projects/AaD_SFDA/source_model/T/source_F.pt"
    netF.load_state_dict(torch.load(modelpath))
    # modelpath = args.output_dir_src + "/source_B.pt"
    modelpath = "/home/bowei/projects/AaD_SFDA/source_model/T/source_B.pt"
    netB.load_state_dict(torch.load(modelpath))
    # modelpath = args.output_dir_src + "/source_C.pt"
    modelpath = "/home/bowei/projects/AaD_SFDA/source_model/T/source_C.pt"
    netC.load_state_dict(torch.load(modelpath))

    param_group = []
    param_group_c = []
    for k, v in netF.named_parameters():
        # if k.find('bn')!=-1:
        if True:
            param_group += [{"params": v, "lr": args.lr * 0.1}]  # 0.1

    for k, v in netB.named_parameters():
        if True:
            param_group += [{"params": v, "lr": args.lr * 1}]  # 1
    for k, v in netC.named_parameters():
        param_group_c += [{"params": v, "lr": args.lr * 1}]  # 1

    optimizer = optim.SGD(param_group)          #,lr=0.02, weight_decay=5e-4
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)          #,lr=0.02, weight_decay=5e-4
    optimizer_c = op_copy(optimizer_c)

    # building feature bank and score bank
    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    fea_bank = torch.randn(num_sample, 256)
    score_bank = torch.randn(num_sample, 12).cuda()
    fea_bank_aug = torch.randn(num_sample, 256)
    score_bank_aug = torch.randn(num_sample, 12).cuda()

    netF.eval()
    netB.eval()
    netC.eval()
    print('**********')
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            indx = data[-1]
            # labels = data[1]
            inputs = inputs.cuda()
            output = netB(netF(inputs))
            output_norm = F.normalize(output)
            outputs = netC(output)
            outputs = nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  # .cpu()

            # aug_policy = ImageNetPolicy()
            # inputs_aug = aug_policy(data[0]).cuda()
            inputs_aug = data[0].cuda()
            output_aug = netB(netF(inputs_aug))
            output_norm_aug = F.normalize(output_aug)
            outputs_aug = netC(output_aug)
            outputs_aug = nn.Softmax(-1)(outputs_aug)
            fea_bank_aug[indx] = output_norm_aug.detach().clone().cpu()
            score_bank_aug[indx] = outputs_aug.detach().clone()  # .cpu()

            
    print('**********************')
    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()
    acc_log = 0

    real_max_iter = max_iter

    while iter_num < real_max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()
        if True:
            alpha = (1 + 10 * iter_num / max_iter) ** (-args.beta) * args.alpha
        else:
            alpha = args.alpha

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)    #(B,C)
        # output_re = softmax_out.unsqueeze(1)

        with torch.no_grad():
            output_f_norm = F.normalize(features_test)     #(B,256)
            output_f_ = output_f_norm.cpu().detach().clone()
            pseudo_label = torch.argmax(score_bank[tar_idx], dim=1)
            confidence = torch.max(score_bank[tar_idx], dim=1)


            pred_bs = softmax_out

            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = softmax_out.detach().clone()

            distance = output_f_ @ fea_bank.T      #(B, N)
            distance_near, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
            nsp_score = torch.sum(distance_near[:, 1:], dim=-1)/args.K

            idx_near = idx_near[:, 1:]  # batch x K    #except the sample itself
            # distance_near = nn.Softmax(dim=1)(distance_near[:, 1:])  * args.K    # batch x K
            score_near = score_bank[idx_near]  # batch x K x C
            neigh_pred = torch.sum(score_near.cpu() * distance_near[:, 1:].unsqueeze(-1), dim=1).cuda()
            neigh_pred = nn.Softmax(dim=-1)(neigh_pred)
            model_pred = score_bank[tar_idx]
            epsilon = 1e-5
            nsc_score = neigh_pred * torch.log(model_pred + epsilon) + model_pred * torch.log(neigh_pred + epsilon)
            nsc_score = torch.sum(nsc_score, dim=-1)
            weight_score = nsp_score.unsqueeze(-1).cuda() * torch.exp(nsc_score).unsqueeze(-1)  #(B,1)

            # score_near_max = torch.max(score_near, dim=-1)[0]   # batch x K
            # score_near_entropy = torch.exp(-Entropy(torch.mean(score_near,dim=1))/torch.log(torch.tensor(12,dtype=torch.float)))  #(batch)

            distance_neg = output_f_ @ output_f_.T  #(B,B)
            aug_pred = score_bank_aug[tar_idx]
            reg_pred = weight_score * neigh_pred + (1-weight_score) * aug_pred
 

        loss_st = F.cross_entropy(softmax_out, pseudo_label, reduction="none") * weight_score
        loss_st = torch.mean(loss_st)
        loss_na = F.kl_div(F.log_softmax(softmax_out, dim=1), F.softmax(reg_pred, dim=1), reduction="batchmean")


        mask = torch.ones((inputs_test.shape[0], inputs_test.shape[0]))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag
        copy = softmax_out.T    #.detach().clone()    #(C,B)
        dot_neg = softmax_out @ copy  # batch x batch

        
        dot_neg = (dot_neg * mask.cuda()).sum(-1) 

        neg_pred = torch.mean(dot_neg)
        loss_dis = neg_pred * alpha

        # softmax_out_un = softmax_out.unsqueeze(1).expand(-1, args.K, -1)
        # loss_pos = torch.mean((F.kl_div(softmax_out_un, score_near, reduction="none").sum(-1)).sum(1) 

        loss = loss_st + loss_na + loss_dis

        if iter_num % 30 == 0:
            log_str = (
                        "Iter:{}/{};  loss: {:.4f}".format(
                        iter_num, max_iter, loss
                        ))
            args.out_file.write(log_str + "\n")
            args.out_file.flush()

        optimizer.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_c.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset == "visda-2017":
                acc, accc = cal_acc_(
                    dset_loaders["test"],
                    fea_bank,
                    score_bank,
                    netF,
                    netB,
                    netC,
                    args,
                    flag=True,
                )
                log_str = (
                    "Time:{}, Task: {}, Iter:{}/{};  Acc on target: {:.2f}".format(
                        str(datetime.now()), args.name, iter_num, max_iter, acc
                    )
                    + "\n"
                    + "T: "
                    + accc
                )

            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str + "\n")
            netF.train()
            netB.train()
            netC.train()
            """
            if acc>acc_log:
                acc_log = acc
                torch.save(
                    netF.state_dict(),
                    osp.join(args.output_dir, "target_F_" + str(args.tag) + ".pt"))
                torch.save(
                    netB.state_dict(),
                    osp.join(args.output_dir,
                                "target_B_" + str(args.tag) + ".pt"))
                torch.save(
                    netC.state_dict(),
                    osp.join(args.output_dir,
                                "target_C_" + str(args.tag) + ".pt"))
            """

    return netF, netB, netC
'''

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPA")
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="3", help="device id to run"
    )
    parser.add_argument("--s", type=int, default=0, help="source")
    parser.add_argument("--t", type=int, default=1, help="target")
    parser.add_argument("--max_epoch", type=int, default=20, help="max iterations")
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--worker", type=int, default=12, help="number of workers")
    parser.add_argument("--dset", type=str, default="visda-2017")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--net", type=str, default="resnet101")
    parser.add_argument("--seed", type=int, default=2021, help="random seed")

    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--layer", type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument("--output", type=str, default="./output/")
    parser.add_argument("--output_src", type=str, default="./weight/source/")
    parser.add_argument("--tag", type=str, default="reg_ST")
    parser.add_argument("--da", type=str, default="uda")
    parser.add_argument("--issave", type=bool, default=True)
    parser.add_argument("--cc", default=False, action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--alpha_decay", default=True)
    parser.add_argument("--nuclear", default=False, action="store_true")
    parser.add_argument("--var", default=False, action="store_true")
    args = parser.parse_args()

    if args.dset == "office-home":
        names = ["Art", "Clipart", "Product", "RealWorld"]
        args.class_num = 65
    if args.dset == "visda-2017":
        names = ["train", "validation"]
        args.class_num = 12

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = "/home/ying/data2/xbw/datasets/VisDA"
        args.s_dset_path = folder  + "/" + names[args.s] + "_list.txt"
        args.t_dset_path = folder  + "/" + names[args.t] + "_list.txt"
        args.test_dset_path = folder  + "/" + names[args.t] + "_list.txt"

        args.output_dir_src = osp.join(
            args.output_src, args.da, args.dset, names[args.s][0].upper()
        )
        args.output_dir = osp.join(
            args.output,
            args.da,
            args.dset,
            names[args.s][0].upper() + names[args.t][0].upper(),
            args.tag,
        )
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system("mkdir -p " + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        shutil.copy('tar_adaptation.py', str(args.output_dir))
        args.out_file = open(
            osp.join(args.output_dir, "log_{}.txt".format(args.tag)), "w"
        )
        args.out_file.write(str(datetime.now()) + "\n")
        args.out_file.write(print_args(args) + "\n")
        args.out_file.flush()
        train_target(args)
