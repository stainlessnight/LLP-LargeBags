import logging
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import os
from llpfc.utils_llpfc import *
import random

def llpfc_train_net(args, 
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
    gt, pred = [], []
    for iteration, (data, label, noisy_label, trans_m, weights) in enumerate(tqdm(train_loader, leave=False)):
        data = data.to(args.device)
        noisy_label = noisy_label.to(args.device)
        trans_m = trans_m.to(args.device)
        weights = weights.to(args.device)
        y = model(data)

        gt.extend(label.cpu().detach().numpy())
        pred.extend(y.argmax(1).cpu().detach().numpy())

        prob = F.softmax(y, dim=1)
        prob_corrected = torch.bmm(trans_m.float(), prob.reshape(prob.shape[0], -1, 1)).reshape(prob.shape[0], -1)
        loss = criterion_train(prob_corrected, noisy_label, weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    train_loss = np.array(losses).mean()
    gt, pred = np.array(gt), np.array(pred)
    train_acc = (gt == pred).mean()

    logging.info(f'[Epoch: {epoch+1}/{args.epochs}] train loss: {np.round(train_loss, 4)}, acc: {np.round(train_acc, 4)}')

    return train_loss, train_acc