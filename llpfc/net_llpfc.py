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
from test import test_net
from llpfc.utils_llpfc import *
from llpfc.losses_llpfc import *
from llpfc.train_llpfc import *
from llpfc.val_llpfc import *


def llpfc_net(args,
        model,
        test_acc_list
        ):

    fix_seed(args.seed)

    test_data = np.load(to_abs_path('data/%s/test_data.npy' % (args.dataset))) 
    test_label = np.load(to_abs_path('data/%s/test_label.npy' % (args.dataset)))

    train_bags = np.load(to_abs_path('data/%s/%d/train_bags.npy' % (args.dataset, args.fold)))
    train_labels = np.load(to_abs_path('data/%s/%d/train_labels.npy' % (args.dataset, args.fold)))
    train_lps = np.load(to_abs_path('data/%s/%d/train_original_lps.npy' % (args.dataset, args.fold)))
    train_bag2indices, train_bag2size, train_bag2prop = make_bags_dict(train_bags.shape[1], train_bags.shape[0], train_lps)

    val_bags = np.load(to_abs_path('data/%s/%d/val_bags.npy' % (args.dataset, args.fold)))
    val_labels = np.load(to_abs_path('data/%s/%d/val_labels.npy' % (args.dataset, args.fold)))
    val_lps = np.load(to_abs_path('data/%s/%d/val_original_lps.npy' % (args.dataset, args.fold)))
    val_bag2indices, val_bag2size, val_bag2prop = make_bags_dict(val_bags.shape[1], val_bags.shape[0], val_lps)

    test_dataset = Dataset_Test(data=test_data, label=test_label)
    test_loader = torch.utils.data.DataLoader(
                                            test_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=False,  
                                            num_workers=args.num_workers
                                            )

    
    criterion_train = LLPFCLoss()
    criterion_val = LLPFCLoss()
    train_net = llpfc_train_net
    eval_net = llpfc_eval_net
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    logging.info(f'''Start training: 
        Epochs:                {args.epochs}
        Patience:              {args.patience}
        Mini Batch size:       {args.mini_batch}
        Learning rate:         {args.lr}
        Dataset:               {args.dataset}
        Bag size:              {args.bag_size}
        Bag Num:               {args.bags_num}
        Test size:             {len(test_loader)}
        Checkpoints:           {args.output_path + str(args.fold)}
        Device:                {args.device}
        Optimizer              {optimizer.__class__.__name__}
        Confidence Interval:   {args.confidence_interval}
    ''')

    best_val_loss = float('inf')

    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []
    
    for epoch in range(args.epochs):
        # Trainning
        train_instance2group, train_group2transition, train_instance2weight, train_noisy_y = make_groups_forward(args.classes,
                                                                                                             train_bag2indices,
                                                                                                             train_bag2size,
                                                                                                             train_bag2prop,
                                                                                                             args.noisy_prior_choice,
                                                                                                             args.weights
                                                                                                             )
        train_dataset = Dataset_Train(
                                    args, 
                                    train_bags, 
                                    train_labels,
                                    train_instance2group, 
                                    train_group2transition, 
                                    train_instance2weight, 
                                    train_noisy_y
                                    )
        
        train_loader = torch.utils.data.DataLoader(
                                                train_dataset, 
                                                batch_size=args.batch_size, 
                                                worker_init_fn = worker_init_fn(args.seed),
                                                shuffle=True,  
                                                num_workers=args.num_workers
                                                )
        train_loss, train_acc = train_net(args,
                                          model,
                                          train_loader,
                                          epoch,
                                          optimizer,
                                          criterion_train
                                          )
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # Validation 
        val_instance2group, val_group2transition, val_instance2weight, val_noisy_y = make_groups_forward(args.classes,
                                                                                                            val_bag2indices,
                                                                                                            val_bag2size,
                                                                                                            val_bag2prop,
                                                                                                            args.noisy_prior_choice,
                                                                                                            args.weights
                                                                                                            )
        val_dataset = Dataset_Train(
                                    args, 
                                    val_bags, 
                                    val_labels,
                                    val_instance2group, 
                                    val_group2transition, 
                                    val_instance2weight, 
                                    val_noisy_y
                                    )
        val_loader = torch.utils.data.DataLoader(
                                            val_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=False,  
                                            num_workers=args.num_workers
                                            )     
        val_loss, val_acc = eval_net(args, 
                                     epoch, 
                                     model, 
                                     val_loader,
                                     criterion_val)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        save_separate_loss_plots(train_loss_list, val_loss_list, args.output_path + '/train_loss' + str(args.fold) + '.png', args.output_path + '/val_loss' + str(args.fold) + '.png')
        save_acc_plot(train_acc_list, val_acc_list, path=args.output_path + '/acc' + str(args.fold) + '.png')
        

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