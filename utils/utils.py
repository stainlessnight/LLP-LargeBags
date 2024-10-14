import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch.nn.functional as F
from scipy.stats import norm
from hydra.utils import to_absolute_path as to_abs_path 
import japanize_matplotlib
import math
from scipy.special import comb


def fix_seed(seed):
    np.random.seed(seed)
    rng = np.random.Generator(np.random.PCG64(seed))
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm
    return rng

def worker_init_fn(seed):
    random.seed(seed)
    np.random.seed(seed)


def save_confusion_matrix(cm, path, title=''):
    sns.heatmap(cm, annot=True, cmap='Blues_r', fmt='.1f')
    plt.xlabel('pred')
    plt.ylabel('GT')
    plt.title(title)
    plt.savefig(path, bbox_inches='tight', dpi = 220, transparent = True)
    plt.close()

def save_minibag_heatmap(heatmap_data, path):
    sns.heatmap(heatmap_data, annot=False, cmap='Blues_r', vmin=0, vmax=1)
    plt.xlabel('minibag_idx')
    plt.ylabel('bag_idx')
    plt.savefig(path, bbox_inches='tight', dpi = 220, transparent = True)
    plt.close()

def save_minibag_avg_heatmap(heatmap_data, path):
    avg_heatmap_data = np.mean(heatmap_data, axis=1).reshape(-1, 1)
    sns.heatmap(avg_heatmap_data, annot=False, cmap='Blues_r', vmin=0, vmax=1)
    plt.xlabel('Average')
    plt.ylabel('bag_idx')
    plt.savefig(path, bbox_inches='tight', dpi=220, transparent=True)
    plt.close()

def load_data_bags(args):  # Toy
    ######### load data #########
    test_data = np.load(to_abs_path('data/%s/test_data.npy' % (args.dataset))) 
    test_label = np.load(to_abs_path('data/%s/test_label.npy' % (args.dataset)))

    train_bags = np.load(to_abs_path('data/%s/%d/train_bags.npy' % (args.dataset, args.fold)))
    train_labels = np.load(to_abs_path('data/%s/%d/train_labels.npy' % (args.dataset, args.fold)))
    train_lps = np.load(to_abs_path('data/%s/%d/train_original_lps.npy' % (args.dataset, args.fold)))  
    val_bags = np.load(to_abs_path('data/%s/%d/val_bags.npy' % (args.dataset, args.fold)))
    val_labels = np.load(to_abs_path('data/%s/%d/val_labels.npy' % (args.dataset, args.fold)))
    val_lps = np.load(to_abs_path('data/%s/%d/val_original_lps.npy' % (args.dataset, args.fold)))

    train_dataset = Dataset_Train(
                                    args=args, 
                                    data=train_bags, 
                                    label=train_labels, 
                                    lp=train_lps, 
                                    )
        
    train_loader = torch.utils.data.DataLoader(
                                                train_dataset, 
                                                batch_size=args.mini_batch, 
                                                worker_init_fn = worker_init_fn(args.seed),
                                                shuffle=True,  
                                                num_workers=args.num_workers
                                                )
    val_dataset = Dataset_Val(
                            args=args,
                            data=val_bags,
                            label=val_labels,
                            lp=val_lps,
                            )
    val_loader = torch.utils.data.DataLoader(
                                            val_dataset, 
                                            batch_size=args.mini_batch,
                                            shuffle=False,  
                                            num_workers=args.num_workers
                                            )
    test_dataset = Dataset_Test(data=test_data, label=test_label)
    test_loader = torch.utils.data.DataLoader(
                                            test_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=False,  
                                            num_workers=args.num_workers
                                            )

    return train_loader, val_loader, test_loader


class Dataset_Val(torch.utils.data.Dataset):
    def __init__(self, args, data, label, lp):
        np.random.seed(args.seed)
        self.data = data
        self.label = label
        self.lp = lp
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071), (0.2673))])
        self.len = data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        lp = self.lp[idx]

        if len(data.shape) == 3:
            data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
            (b, w, h, c) = data.shape
            trans_data = torch.zeros((b, c, w, h))
            for i in range(b):
                trans_data[i] = self.transform2(data[i])
        else:
            (b, w, h, c) = data.shape
            trans_data = torch.zeros((b, c, w, h))
            for i in range(b):
                trans_data[i] = self.transform(data[i])
        
        data = trans_data
        label = torch.tensor(label).long()
        lp = torch.tensor(lp).float()
        lp = torch.round(lp * 10000) / 10000
        return data, label, lp


class Dataset_Test(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071), (0.2673))])
        self.len = self.data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]
        if len(data.shape) != 3:
            data = data.reshape(data.shape[0], data.shape[1], 1)
            data = self.transform2(data)
        else:
            data = self.transform(data)
        label = self.label[idx]
        label = torch.tensor(label).long()
        return data, label

class Dataset_Train(torch.utils.data.Dataset):
    def __init__(self, args, data, label, lp):
        self.rng = fix_seed(args.seed)
        self.CI = args.confidence_interval
        self.data = data
        self.label = label
        self.lp = lp
        self.classes = args.classes
        self.bag_size = args.bag_size
        self.bags_num = data.shape[0]
        self.minibags_num = args.minibags_num
        self.num_sampled_instances = args.num_sampled_instances
        self.label_sampling_method = args.label_sampling_method
        self.minibags_instances = args.minibags_instances
        self.apply_loss_weights = args.apply_loss_weights
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071), (0.2673))])
        self.len = self.bags_num * self.minibags_num
        self.random_index = np.array(  # [bags_num][minibags_num][bag_size]
            [[np.random.permutation(np.arange(self.bag_size)) for _ 
              in range(self.minibags_num)] for _ in range(self.bags_num)])
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        bag_idx = idx // self.minibags_num
        minibag_idx = idx % self.minibags_num
        data, label, lp = self.data[bag_idx], self.label[bag_idx], self.lp[bag_idx]
        label = torch.tensor(label).long() 
        std_list = calc_class_std(self.bag_size, self.num_sampled_instances, lp)
        std_list = torch.tensor(std_list).float()
    
        if len(data.shape) == 3:
            data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
            (b, w, h, c) = data.shape
            trans_data = torch.zeros((b, c, w, h))
            for i in range(b):
                trans_data[i] = self.transform2(data[i])
        else:
            (b, w, h, c) = data.shape
            trans_data = torch.zeros((b, c, w, h))
            for i in range(b):
                trans_data[i] = self.transform(data[i])
        data = trans_data

        # sampling instances from a population bag
        if self.minibags_instances == 'unfixed':
            random_index = np.random.permutation(np.arange(self.bag_size))
            minibag_data = data[random_index[:self.num_sampled_instances]]
            minibag_label = label[random_index[:self.num_sampled_instances]]
        else:  # fixed
            minibag_data = data[self.random_index[bag_idx][minibag_idx][:self.num_sampled_instances]]
            minibag_label = label[self.random_index[bag_idx][minibag_idx][:self.num_sampled_instances]]

        # sampling label proportions from a population bag
        if self.label_sampling_method == 'population':
            sampled_lp = lp
        elif self.label_sampling_method == 'hypergeometric':
            sampled_lp, weight = sampling_multivariate_hypergeometric_label(
                lp, self.bag_size, self.num_sampled_instances, self.rng)
        elif self.label_sampling_method == 'uniform':
            sampled_lp = sampling_uniform_distribution(lp, 0.3)
        elif self.label_sampling_method == 'normal':
            sampled_lp = sampling_normal_distribution(lp, 0.10)
        elif self.label_sampling_method == 'mode':
            sampled_lp = find_max_hypergeom_combination(lp, self.bag_size, self.num_sampled_instances)
        elif self.label_sampling_method == 'discrete_uniform':
            sampled_lp = sampling_discrete_uniform_distribution(lp, 0.2, self.num_sampled_instances)
        else:
            raise NotImplementedError
        
        sampled_lp = torch.tensor(sampled_lp).float()
        sampled_lp = torch.round(sampled_lp * 10000) / 10000
        
        # give weighit to the loss function
        if self.apply_loss_weights == True:
            weight = torch.tensor(weight).float()
        else:
            weight = torch.tensor(1).float()

        return minibag_data, minibag_label, sampled_lp, std_list, weight, bag_idx, minibag_idx
    


def save_loss_plot(train_loss_list, val_loss_list, path):
    plt.plot(train_loss_list, label='train loss')
    plt.plot(val_loss_list, label='val loss')
    plt.legend(fontsize=18) 
    plt.rcParams["font.size"] = 18
    plt.savefig(path, transparent=True, bbox_inches='tight', dpi = 220)
    plt.close()

def save_separate_loss_plots(train_loss_list, val_loss_list, train_path, val_path):
    # 訓練損失のプロットと保存
    plt.plot(train_loss_list, label='train loss', color='tab:blue')
    plt.legend(fontsize=18)
    plt.rcParams["font.size"] = 18
    plt.savefig(train_path, transparent=True, bbox_inches='tight', dpi=220)
    plt.close()
    # 検証損失のプロットと保存
    plt.plot(val_loss_list, label='val loss', color='tab:orange')
    plt.legend(fontsize=18)
    plt.rcParams["font.size"] = 18
    plt.savefig(val_path, transparent=True, bbox_inches='tight', dpi=220)
    plt.close()

def save_acc_plot(train_acc_list, val_acc_list, path):
    plt.plot(train_acc_list, label='train accuracy')
    plt.plot(val_acc_list, label='val accuracy')
    plt.legend(fontsize=18)
    plt.rcParams["font.size"] = 18
    plt.savefig(path, transparent=True, bbox_inches='tight', dpi = 220)
    plt.close()

def calc_proportion(classes, label):
    label_prop = np.zeros(classes)
    for i in range(len(label)):
        for c in range(classes):
            if label[i].item() == c:
                label_prop[c] += 1
    label_prop /= len(label)
    label_prop = torch.tensor(label_prop).float()
    label_prop = torch.round(label_prop * 10000) / 10000
    return label_prop


def calc_diff_prop(classes, bag_prop, minibag_prop):
    diff_prop = np.zeros(classes)
    for i in range(classes):
        diff_prop[i] = bag_prop[i] - minibag_prop[i]
    return diff_prop  # diff_prop.shape = (classes,)

def save_diff_prop_histogram(diff_prop, bins, filename='histogram.png'):
    plt.hist(diff_prop, bins, range=(-0.25, 0.25))
    plt.xlabel('Difference in Proportions')
    plt.ylabel('Frequency')
    plt.title('Histogram of Difference in Proportions')
    plt.rcParams["font.size"] = 18
    plt.savefig(filename, bbox_inches='tight', dpi = 220, transparent=True)
    plt.close()

# 比率の差から標準偏差を計算する関数
def calc_std(diff_prop, epoch):
    std = np.std(diff_prop)
    return std


def calc_class_std(bag_size, minibag_size, proportion): 
    std_list = []
    N, n = bag_size, minibag_size
    for k in proportion:
        K = k*N
        std_list.append(math.sqrt((n*K*(N-K)*(N-n))/(N*N*(N-1))) / n)
    return std_list


def sampling_success_ratio(lp: np.ndarray, bag_size: int, num_sampled_instances: int) -> torch.Tensor:
    successes = (lp * bag_size).astype(int)
    sampled_successes = np.random.hypergeometric(successes, bag_size - successes, num_sampled_instances)
    success_ratios = sampled_successes / num_sampled_instances
    if np.sum(success_ratios) == 0:
        success_ratios = sampling_success_ratio(lp, bag_size, num_sampled_instances)
    else:
        success_ratios = success_ratios / np.sum(success_ratios)
    return success_ratios

def pmf_multivariate_hypergeometric(N, n, x):
    total_combinations = comb(np.sum(N), n)
    category_combinations = np.prod([comb(N_i, x_i) for N_i, x_i in zip(N, x)])
    return category_combinations / total_combinations

def pmf_hypergeometric(N, K, n, k):
    return comb(K, k) * comb(N - K, n - k) / comb(N, n)

def sampling_multivariate_hypergeometric_label(lp: np.ndarray, bag_size: int, num_sampled_instances: int, rng: np.random.Generator) -> torch.Tensor:
    instances = (lp * bag_size).astype(int)
    sampled_instances = rng.multivariate_hypergeometric(instances, num_sampled_instances)
    sampled_lp = sampled_instances / num_sampled_instances
    probability = pmf_multivariate_hypergeometric(instances, num_sampled_instances, sampled_instances)
    return sampled_lp, probability

def find_max_hypergeom_combination(lp, bag_size, num_sampled_instances):
    Ni = np.round(lp * bag_size).astype(int)

    def adjust_to_sum_to_n_proportional(expected, N, n):
        while n > 0:
            proportions = N / N.sum()
            expected += np.round(proportions * n).astype(int)
            n -= expected.sum()
            N -= expected
            for i in proportions.argsort()[::-1]:
                if n <= 0:
                    break
                if N[i] > 0:
                    expected[i] += 1
                    n -= 1
        return expected

    expected_values = np.zeros_like(Ni)
    expected_values = adjust_to_sum_to_n_proportional(expected_values, Ni.copy(), num_sampled_instances) / num_sampled_instances
    return expected_values

def sampling_uniform_distribution(lp: np.ndarray, width: float):
    random_values = []
    for i in lp:
        lower_bound = max(0, i - width)
        upper_bound = min(1, i + width)
        random_value = np.random.uniform(lower_bound, upper_bound)
        random_values.append(random_value)
    random_values = np.array(random_values)
    return random_values


def sampling_normal_distribution(lp: np.ndarray, std: float):
    random_values = []
    for i in lp:
        while True:
            random_value = np.random.normal(i, std)
            if random_value >= 0:
                break
        random_values.append(random_value)
    random_values = np.array(random_values)
    random_values = random_values / random_values.sum()  # 正規化
    return random_values

def random_label_proportions(classes: int, num_sampled_instances: int):
    lp = np.random.rand(classes)
    lp = np.floor(lp / lp.sum() * num_sampled_instances).astype(int)
    while lp.sum() < num_sampled_instances:
        lp[np.random.randint(classes)] += 1
    lp = lp / num_sampled_instances
    return lp

def continuous_prop_to_discrete_prop(lp: np.array, num_sampled_instances: int):
    discrete_lp = np.floor(lp * num_sampled_instances) / num_sampled_instances
    return np.round(discrete_lp, decimals=6)

def sampling_discrete_uniform_distribution(lp: np.ndarray, width: float, num_sampled_instances: int):
    random_values = []

    for prob in lp:
        lower_bound = max(0, prob - width)
        upper_bound = min(1, prob + width)
        random_value = np.random.uniform(lower_bound, upper_bound)
        random_values.append(random_value)

    random_values = np.array(random_values)
    random_values = continuous_prop_to_discrete_prop(random_values, num_sampled_instances)
    return random_values
