import logging
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import os
from utils.utils import calc_proportion
from llpvat.utils_llpvat import get_rampup_weight
import random

def llpvat_eval_net(args, 
             epoch,
             model,
             val_loader, 
             loss_function_val):
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
    
    model.eval()
    losses = []
    gt, pred = [], []
    for iteration, (data, label, lp) in enumerate(tqdm(val_loader, leave=False)):
        (b, n, c, w, h) = data.size()
        data = data.reshape(-1, c, w, h)
        label = label.reshape(-1)
        data, lp = data.to(args.device), lp.to(args.device)
        y = model(data)

        gt.extend(label.cpu().detach().numpy())
        pred.extend(y.argmax(1).cpu().detach().numpy())

        confidence = F.softmax(y, dim=1)
        confidence = confidence.reshape(b, n, -1)
        pred_prop = confidence.mean(dim=1)

        alpha = get_rampup_weight(0.05, iteration, -1)
        weight = torch.tensor(1).float()
        loss = loss_function_val(model, data, alpha, pred_prop, lp, weight)
        losses.append(loss.item())

    val_loss = np.array(losses).mean()

    gt, pred = np.array(gt), np.array(pred)
    val_acc = (gt == pred).mean()

    logging.info('[Epoch: %d/%d] val loss: %.4f, acc: %.4f' %
                    (epoch+1, args.epochs,
                    val_loss, val_acc))

    return val_loss, val_acc