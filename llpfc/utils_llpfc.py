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
from numpy.linalg import matrix_rank
import math
from scipy.special import comb
from scipy.optimize import minimize, Bounds, LinearConstraint


class InvalidChoiceOfWeights(Exception):
    pass


class InvalidChoiceOfNoisyPrior(Exception):
    pass


def make_bags_dict(bag_size, bags_num, lps):
    bag2indices, bag2size, bag2prop = dict(), dict(), dict()
    for i in range(bags_num):
        bag2indices[i] = np.arange(bag_size) + i * bag_size
        bag2size[i] = bag_size
        bag2prop[i] = lps[i]
    return bag2indices, bag2size, bag2prop


def approx_noisy_prior(gamma_m, clean_prior, seed):
    def ls_error(x, A, b):
        return 0.5 * np.sum((np.matmul(A, x) - b) ** 2)

    def grad(x, A, b):
        return np.matmul(np.matmul(np.transpose(A), A), x) - np.matmul(np.transpose(A), b)

    def hess(x, A, b):
        return np.matmul(np.transpose(A), A)

    x0 = np.random.rand(clean_prior.shape[0])
    x0 /= np.sum(x0)

    res = minimize(ls_error,
                   x0,
                   args=(np.transpose(gamma_m), clean_prior),
                   method='trust-constr',
                   jac=grad,
                   hess=hess,
                   bounds=Bounds(np.zeros(x0.shape), np.ones(x0.shape)),
                   constraints=LinearConstraint(np.ones(x0.shape), np.ones(1), np.ones(1)),
                   )
    return res.x


def make_a_group(classes, clean_prior, bag_ids, bag2prop, noisy_prior_choice):
    bags_list = random.sample(bag_ids, classes)
    gamma_m = np.zeros((classes, classes))
    for row_idx in range(classes):
        gamma_m[row_idx, :] = bag2prop[bags_list[row_idx]]
    if noisy_prior_choice == 'approx':
        noisy_prior_approx = approx_noisy_prior(np.transpose(gamma_m), clean_prior, seed)
    elif noisy_prior_choice == 'uniform':
        noisy_prior_approx = np.ones((classes,)) / classes
    else:
        raise InvalidChoiceOfNoisyPrior("Unknown choice of noisy prior: %s" % noisy_prior_choice)
    assert np.all(noisy_prior_approx >= 0)
    assert (np.sum(noisy_prior_approx) - 1) < 1e-4
    clean_prior_approx = np.matmul(np.transpose(gamma_m), noisy_prior_approx)

    transition_m = np.zeros((classes, classes))
    for i in range(classes):
        for j in range(classes):
            transition_m[i, j] = gamma_m[i, j] * noisy_prior_approx[i] / clean_prior_approx[j]  # clean_prior can't be 0 in this case

    return bags_list, noisy_prior_approx, transition_m


def make_groups_forward(classes, bag2indices, bag2size, bag2prop, noisy_prior_choice, weights):
    bag_ids = set(bag2indices.keys())
    num_groups = len(bag_ids) // classes
    assert num_groups > 0

    clean_prior = np.zeros((classes, ))
    for bag_id in bag2size.keys():
        clean_prior += bag2prop[bag_id] * bag2size[bag_id]
    clean_prior /= np.sum(clean_prior)

    group2bag = {}
    group2noisyp = {}
    group2transition = {}
    group_id = 0
    groups = []
    while len(bag_ids) >= classes:
        bags_list, noisy_prior, transition_m = make_a_group(classes,
                                                            clean_prior,
                                                            bag_ids,
                                                            bag2prop,
                                                            noisy_prior_choice)
        bag_ids = bag_ids - set(bags_list)
        group2bag[group_id], group2noisyp[group_id], group2transition[group_id] = bags_list, noisy_prior, transition_m
        groups.append(group_id)
        group_id += 1
    group2bag[-1] = list(bag_ids)  # bags that are not in a group
    groups.append(-1)

    instance2group = {instance_id: group_id for group_id in groups for bag_id in group2bag[group_id] for
                      instance_id in bag2indices[bag_id]}

    # calculate the weights of groups
    if weights == 'uniform':
        group2weights = {group_id: 1.0 for group_id, trans_m in group2transition.items()}
    else:
        raise InvalidChoiceOfWeights("Unknown way to determine weights %s, use either ch_vol or uniform" % weights)

    # set the noisy labels
    noisy_y = -np.ones((sum([len(instances) for instances in bag2indices.values()]),))
    instance2weight = np.zeros((sum([len(instances) for instances in bag2indices.values()]),))
    for group_id in groups:
        if group_id == -1:
            continue

        noisy_prior = group2noisyp[group_id]
        noisy_prop = np.zeros((classes, ))
        for noisy_class, bag_id in enumerate(group2bag[group_id]):
            noisy_prop[noisy_class] = bag2size[bag_id]
        noisy_prop /= np.sum(noisy_prop)
        weights = np.divide(noisy_prior, noisy_prop)
        weights /= np.sum(weights)

        for noisy_class, bag_id in enumerate(group2bag[group_id]):
            for instance_id in bag2indices[bag_id]:
                noisy_y[instance_id] = noisy_class
                instance2weight[instance_id] = weights[noisy_class] * group2weights[group_id]

    return instance2group, group2transition, instance2weight, noisy_y


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


def truncate_data_group(x, y, instance2group):
	idx_list = []
	for i in range(x.shape[0]):
		if instance2group[i] != -1:
			idx_list.append(i)
	x_truncated = x[idx_list]
	y_truncated = y[idx_list]
	idx2new = {idx_list[i]: i for i in range(len(idx_list))}
	instance2group_new = {}
	for old, new in idx2new.items():
		instance2group_new[new] = instance2group[old]
	new2idx = {idx2new[idx]: idx for idx in idx2new.keys()}
	return x_truncated, y_truncated, instance2group_new, new2idx



class Dataset_Train(torch.utils.data.Dataset):
    def __init__(self, args, data, label, instance2group, group2transition, instance2weight, noisy_y):
        self.data = data.reshape(-1, data.shape[2], data.shape[3], data.shape[4])
        self.label = label.reshape(-1)
        self.data, self.noisy_y, self.instance2group, self.new2idx = truncate_data_group(self.data, noisy_y, instance2group)
        self.group2transition = group2transition
        self.instance2weight = instance2weight
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
        data, label = self.data[idx], self.label[idx]
        y_ = self.noisy_y[idx]
        label = torch.tensor(label).long()
        trans_m = self.group2transition[self.instance2group[idx]]
        weight = self.instance2weight[self.new2idx[idx]]
    
        if len(data.shape) == 2:
            data = data.reshape(data.shape[0], data.shape[1], 1)
            (w, h, c) = data.shape
            trans_data = torch.zeros((c, w, h))
            trans_data = self.transform2(data)
        else:
            (w, h, c) = data.shape
            trans_data = torch.zeros((c, w, h))
            trans_data = self.transform(data)
        data = trans_data



        return data, label, int(y_), torch.tensor(trans_m, dtype=None), weight
    



class Dataset_Val(torch.utils.data.Dataset):
    def __init__(self, args, data, label, instance2group, group2transition, instance2weight, noisy_y):
        self.data = data.reshape(-1, data.shape[2], data.shape[3], data.shape[4])
        self.label = label.reshape(-1)
        self.data, self.noisy_y, self.instance2group, self.new2idx = truncate_data_group(self.data, noisy_y, instance2group)
        self.group2transition = group2transition
        self.instance2weight = instance2weight
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
        data, label = self.data[idx], self.label[idx]
        y_ = self.noisy_y[idx]
        label = torch.tensor(label).long()
        trans_m = self.group2transition[self.instance2group[idx]]
        weight = self.instance2weight[self.new2idx[idx]]
    
        if len(data.shape) == 2:
            data = data.reshape(data.shape[0], data.shape[1], 1)
            (w, h, c) = data.shape
            trans_data = torch.zeros((c, w, h))
            trans_data = self.transform2(data)
        else:
            (w, h, c) = data.shape
            trans_data = torch.zeros((c, w, h))
            trans_data = self.transform(data)
        data = trans_data



        return data, label, int(y_), torch.tensor(trans_m, dtype=None), weight


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