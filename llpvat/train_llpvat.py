import logging
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import os
from utils.utils import calc_proportion
from llpvat.utils_llpvat import get_rampup_weight
import random

def llpvat_train_net(args, 
              model,
              train_loader,
              epoch,
              optimizer, 
              criterion_train):
    '''Training: the proportion loss with confidential interval
        Args:
            args (argparse): contain parameters
            train_loader (torch.utils.data): train dataloader
            model (torch.tensor): ResNet18 
            epoch (int): current epoch
            optimizer (torch.optim): optimizer such as Adam
            criterion_train: loss function for training
            
        Returns:
            train_loss (float): average of train loss
            train_acc (float): train accuracy
    '''
    
    model.train()
    losses = []
    gt, pred, diff_prop_list = [], [], []
    pos, neg, same = 0, 0, 0
    cnt0, cnt1 = 0, 0
    heatmap_data = np.zeros((int(args.bags_num*4/5), args.minibags_num))
    for iteration, (data, label, new_lp, std_list, weight, bag_idx, minibag_idx) in enumerate(tqdm(train_loader, leave=False)):
        (b, n, c, w, h) = data.size()
        data = data.reshape(-1, c, w, h)
        label = label.reshape(-1)    
        data, new_lp = data.to(args.device), new_lp.to(args.device)
        weight = weight.to(args.device)
        y = model(data)

        gt.extend(label.cpu().detach().numpy())
        pred.extend(y.argmax(1).cpu().detach().numpy())

        confidence = F.softmax(y, dim=1)
        confidence = confidence.reshape(b, n, -1)
        pred_prop = confidence.mean(dim=1)

        label = label.reshape(b, n)
        label = label.to(args.device)

        for i in range(b):
            true_lp = calc_proportion(args.classes, label[i])
            true_lp = true_lp.to(args.device)
            diff = torch.round((true_lp - new_lp[i]) * 1000) / 1000
            diff = diff.cpu().detach().numpy()
            diff_prop_list.append(diff)
            for j in range(args.classes):
                if diff[j] > 0:
                    pos += 1
                elif diff[j] < 0:
                    neg += 1
                else:
                    same += 1
            for k in range(args.classes):
                if new_lp[i][k] < pred_prop[i][k] < true_lp[k]:
                    cnt0 += 1
                elif true_lp[k] < pred_prop[i][k] < new_lp[i][k]:
                    cnt0 += 1
                else:
                    cnt1 += 1

            correct = (label[i] == y.argmax(1)[i * n:(i + 1) * n]).sum().item()
            total = n
            heatmap_data[bag_idx, minibag_idx] = correct / total
            
        cnt_diff = [pos, neg, same]
        cnt_direction = [cnt0, cnt1]
        
        weight_median = torch.median(weight)
        weight = weight / weight_median

        alpha = get_rampup_weight(0.05, iteration, -1)
        
        loss = criterion_train(model, data, alpha, pred_prop, 
                                new_lp, 
                                weight,
                                )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

        

    train_loss = np.array(losses).mean()
    gt, pred = np.array(gt), np.array(pred)
    train_acc = (gt == pred).mean()


    logging.info(f'[Epoch: {epoch+1}/{args.epochs}] train loss: {np.round(train_loss, 4)}, acc: {np.round(train_acc, 4)}')

 

    

    return train_loss, train_acc, diff_prop_list, cnt_diff, cnt_direction, heatmap_data