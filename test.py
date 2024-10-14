import logging
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def test_net(args, 
             epoch,
             model,
             test_loader
             ):
    
    model.eval()
    gt, pred = [], []
    with torch.no_grad():
        for data, label in tqdm(test_loader, leave=False):
            data = data.to(args.device)
            y = model(data)

            gt.extend(label.cpu().detach().numpy())
            pred.extend(y.argmax(1).cpu().detach().numpy())

    gt, pred = np.array(gt), np.array(pred)
    test_acc = (gt == pred).mean()
    test_cm = confusion_matrix(y_true=gt, y_pred=pred, normalize='true')

    # Calculating micro and macro F1 scores
    micro_f1 = f1_score(gt, pred, average='micro')
    macro_f1 = f1_score(gt, pred, average='macro')

    logging.info('[Epoch: %d/%d] test acc: %.4f' %(epoch+1, args.epochs, test_acc))
    logging.info('Micro F1: {:.4f}, Macro F1: {:.4f}'.format(
        micro_f1, macro_f1))
    logging.info('===============================')

    return test_acc, test_cm, micro_f1, macro_f1