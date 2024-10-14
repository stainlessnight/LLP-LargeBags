import logging
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import os
import random

def llpfc_eval_net(args, 
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
    with torch.no_grad():
        for iteration, (data, label, noisy_label, trans_m, weights) in enumerate(tqdm(val_loader, leave=False)):
            data = data.to(args.device)
            noisy_label = noisy_label.to(args.device)
            trans_m = trans_m.to(args.device)
            weights = weights.to(args.device)
            y = model(data)

            gt.extend(label.cpu().detach().numpy())
            pred.extend(y.argmax(1).cpu().detach().numpy())

            prob = F.softmax(y, dim=1)
            prob_corrected = torch.bmm(trans_m.float(), prob.reshape(prob.shape[0], -1, 1)).reshape(prob.shape[0], -1)
            loss = loss_function_val(prob_corrected, noisy_label, weights)
            losses.append(loss.item())

        val_loss = np.array(losses).mean()

        gt, pred = np.array(gt), np.array(pred)
        val_acc = (gt == pred).mean()

        logging.info('[Epoch: %d/%d] val loss: %.4f, acc: %.4f' %
                        (epoch+1, args.epochs,
                        val_loss, val_acc))

    return val_loss, val_acc