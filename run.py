import os
import sys
import logging
from tqdm import tqdm
from argument import SimpleArgumentParser

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18

from hydra.utils import to_absolute_path as abs_path 
from utils.utils import *
from utils.losses import *

from train import pl_train_net
from val import pl_eval_net
from test import test_net

from llpvat.losses_llpvat import *
from llpvat.train_llpvat import *
from llpvat.val_llpvat import *

from llpfc.net_llpfc import *
from llpfc.losses_llpfc import *
from llpfc.train_llpfc import *
from llpfc.val_llpfc import *




def net(args,
        model,
        test_acc_list
        ):

    fix_seed(args.seed)
    # Generating dataloader
    train_loader, val_loader, test_loader = load_data_bags(args)

    if args.llp_method == 'pl':
        criterion_train = ProportionLoss_Weighted()
        criterion_val = ProportionLoss()
        train_net = pl_train_net
        eval_net = pl_eval_net
    elif args.llp_method == 'llpvat':
        criterion_train = LLPVATLoss()
        criterion_val = LLPVATLoss()
        train_net = llpvat_train_net
        eval_net = llpvat_eval_net
    else:
        raise ValueError('llp_method is not correct')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    logging.info(f'''Start training: 
        Epochs:                {args.epochs}
        Patience:              {args.patience}
        Mini Batch size:       {args.mini_batch}
        Learning rate:         {args.lr}
        Dataset:               {args.dataset}
        Bag size:              {args.bag_size}
        Bag Num:               {args.bags_num}
        Training size:         {len(train_loader)}
        Validation size:       {len(val_loader)}
        Test size:             {len(test_loader)}
        Checkpoints:           {args.output_path + str(args.fold)}
        Device:                {args.device}
        Optimizer              {optimizer.__class__.__name__}
        Confidence Interval:   {args.confidence_interval}
    ''')

    best_val_loss = float('inf')

    train_loss_list, train_acc_list, train_diff_list = [], [], []
    val_loss_list, val_acc_list = [], []
    
    for epoch in range(args.epochs):
        # Trainning
        train_loss, train_acc, train_diff_prop, cnt_diff, cnt_direction, heatmap_data = train_net(args,
                                          model,
                                          train_loader,
                                          epoch,
                                          optimizer,
                                          criterion_train
                                          )
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        train_diff_prop = np.concatenate(train_diff_prop)
        train_diff_list.append(train_diff_prop)
        os.makedirs(args.output_path + '/heatmap' + str(args.fold)) if os.path.exists(args.output_path + '/heatmap' + str(args.fold)) is False else None
        save_minibag_heatmap(heatmap_data, 
                              path=args.output_path + '/heatmap' + str(args.fold) + '/heatmap' + str(epoch) + '.png')
        save_minibag_avg_heatmap(heatmap_data,
                                path=args.output_path + '/heatmap' + str(args.fold) + '/heatmap_avg' + str(epoch) + '.png')


        # Validation 
        val_loss, val_acc = eval_net(args, 
                                     epoch, 
                                     model, 
                                     val_loader,
                                     criterion_val)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        save_separate_loss_plots(train_loss_list, val_loss_list, args.output_path + '/train_loss' + str(args.fold) + '.png', args.output_path + '/val_loss' + str(args.fold) + '.png')
        save_acc_plot(train_acc_list, val_acc_list, path=args.output_path + '/acc' + str(args.fold) + '.png')
        save_diff_prop_histogram(train_diff_prop, bins=30, filename=args.output_path + '/train_diff_prop' + str(args.fold) + '.png')  # shape > (epoch*bagsize,)
        std = np.std(train_diff_prop)
        diff_abs_avg = np.mean(np.abs(train_diff_prop))
        with open(args.output_path + '/std' + str(args.fold) + '.txt', mode='a') as f:
            f.write(str(epoch) + ': ' + 'std ' + str(std) + ', diff_abs_avg ' + str(diff_abs_avg) +'\n')
        pos, neg, same = cnt_diff
        with open(args.output_path + '/cnt_diff' + str(args.fold) + '.txt', mode='a') as f:
            f.write(str(epoch) + ': ' + 'pos ' + str(pos) + ', neg ' + str(neg) + ', same ' + str(same) +'\n')
        cnt0, cnt1 = cnt_direction
        with open(args.output_path + '/cnt_direction' + str(args.fold) + '.txt', mode='a') as f:
            f.write(str(epoch) + ': ' + 'different direction ' + str(cnt0) + ', same direction ' + str(cnt1) +'\n')

        # save best val loss model
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_path = args.output_path +  str(args.fold) + f'/CP_epoch{best_epoch + 1}.pkl'
            torch.save(model.state_dict(), 
                       best_path)
    
    # Load Best Parameters 
    model.load_state_dict(
        torch.load(best_path, map_location=args.device)
    )
    logging.info(f'Model loaded from {best_path}')

    # Test 
    test_acc, test_cm, micro_f1, macro_f1 = test_net(args, 
                                 epoch, 
                                 model, 
                                 test_loader)
    test_acc_list.append(test_acc)
    logging.info(f'Test acc means: {np.mean(test_acc_list)}')
    save_confusion_matrix(cm=test_cm, 
                          path=args.output_path  + '/Confusion_matrix' + str(args.fold) + '.png',
                          title='test: acc: %.4f' % test_acc)
    # save test_acc, micro_f1, macro_f1 to txt
    with open(args.output_path + '/test' + '.txt', mode='a') as f:
        f.write(str(args.fold) + 'test_acc: ' + str(test_acc) + '\n')
        f.write(str(args.fold) + 'micro_f1: ' + str(micro_f1) + '\n')
        f.write(str(args.fold) + 'macro_f1: ' + str(macro_f1) + '\n')
        f.write("----------------------------------\n")


def main(args):
    fix_seed(args.seed)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {args.device}')

    logging.info(f'Network:\n'
                 f'\t{args.channels} input channels\n'
                 f'\t{args.classes} output channels\n'
                 )
    args.output_path = args.output_path + args.dataset
    os.makedirs(args.output_path) if os.path.exists(args.output_path) is False else None
    test_acc_list = []

    for fold in range(5):
        args.fold = fold
        model = resnet18(pretrained=args.pretrained)
        if args.channels != 3:
            model.conv1 = nn.Conv2d(args.channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, args.classes)
        model = model.to(args.device)
        os.makedirs(args.output_path + str(args.fold)) if os.path.exists(args.output_path + str(args.fold)) is False else None

        try:
            if args.llp_method == 'llpfc':
                llpfc_net(args,
                    model,
                    test_acc_list
                    )
            else:
                net(args,
                    model,
                    test_acc_list
                    )
        except KeyboardInterrupt:
            torch.save(model.state_dict(), abs_path('INTERRUPTED.pth'))
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)


if __name__ == '__main__':
    args = SimpleArgumentParser().parse_args()
    main(args)
