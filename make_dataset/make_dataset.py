import argparse
import random
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from load_svhn import load_svhn
from load_cifar10 import load_cifar10
from load_NCT_CRC_HE_100K import load_nct_crc_data
from glob import glob
from PIL import Image
from tqdm import tqdm

def make_folder(path):
    p = ''
    for x in path.split('/'):
        p += x+'/'
        if not os.path.exists(p):
            os.mkdir(p)

def get_label_proportion(num_bags=100, num_classes=10):
    proportion = np.random.rand(num_bags, num_classes)
    proportion /= proportion.sum(axis=1, keepdims=True)
    return proportion

def get_N_label_proportion(proportion, num_instances, num_classes):
    N = np.zeros(proportion.shape)
    for i in range(len(proportion)):
        p = proportion[i]
        for c in range(len(p)):
            if (c+1) != num_classes:
                num_c = int(np.round(num_instances*p[c]))
                if sum(N[i])+num_c >= num_instances:
                    num_c = int(num_instances-sum(N[i]))
            else:
                num_c = int(num_instances-sum(N[i]))

            N[i][c] = int(num_c)
        np.random.shuffle(N[i])
    print(N.sum(axis=0))
    print((N.sum(axis=1) != num_instances).sum())
    return N


def create_bags(data, label, num_bags, args):
    # make proportion
    proportion = get_label_proportion(num_bags, args.num_classes)
    proportion_N = get_N_label_proportion(proportion, args.num_instances, args.num_classes)
    # make index
    idx = np.arange(len(label))
    idx_c = []
    for c in range(args.num_classes):
        x = idx[label[idx] == c]
        np.random.shuffle(x)
        idx_c.append(x)
    bags_idx = []
    for n in range(len(proportion_N)):
        bag_idx = []
        for c in range(args.num_classes):
            sample_c_index = np.random.choice(idx_c[c], size=int(proportion_N[n][c]), replace=False)
            bag_idx.extend(sample_c_index)
        np.random.shuffle(bag_idx)
        bags_idx.append(bag_idx)
    # make data, label, proportion
    bags, labels = data[bags_idx], label[bags_idx]
    original_lps = proportion_N / args.num_instances

    return bags, labels, original_lps



def main(args):
    # load dataset
    if args.dataset == 'cifar10':
        data, label, test_data, test_label = load_cifar10()
    elif args.dataset == 'svhn':
        data, label, test_data, test_label = load_svhn()
    elif args.dataset == 'NCT_CRC_HE_100K':
        data, label, test_data, test_label = load_nct_crc_data()

    # k-fold cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(skf.split(data, label)):
        train_data, train_label = data[train_idx], label[train_idx]
        val_data, val_label = data[val_idx], label[val_idx]

        output_path = 'data/%s/%d/' % (
            args.dataset, i)
        make_folder(output_path)

        # train
        bags, labels, original_lps = create_bags(train_data, train_label,
                                                args.train_num_bags,
                                                args)
        np.save('%s/train_bags' % (output_path), bags)
        np.save('%s/train_labels' % (output_path), labels)
        np.save('%s/train_original_lps' % (output_path), original_lps)

        # val
        bags, labels, original_lps = create_bags(val_data, val_label,
                                                args.val_num_bags,
                                                args)
        np.save('%s/val_bags' % (output_path), bags)
        np.save('%s/val_labels' % (output_path), labels)
        np.save('%s/val_original_lps' % (output_path), original_lps)


    test_data_path = 'data/%s/' % (args.dataset)
    # test
    used_test_data, used_test_label = [], []
    for c in range(args.num_classes):
        used_test_data.extend(test_data[test_label == c])
        used_test_label.extend(test_label[test_label == c])
    test_data, test_label = np.array(
        used_test_data), np.array(used_test_label)

    np.save('%s/test_data' % (test_data_path), test_data)
    np.save('%s/test_label' % (test_data_path), test_label)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--dataset', default='NCT_CRC_HE_100K', type=str)
    parser.add_argument('--num_classes', default=9, type=int)
    parser.add_argument('--num_instances', default=1000, type=int)

    parser.add_argument('--train_num_bags', default=80, type=int)
    parser.add_argument('--val_num_bags', default=20, type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)
    main(args)
