from collections import defaultdict, OrderedDict
import copy
from datetime import datetime
import json
import os
import sys
import time
import random

import numpy as np
import torch
import torch.utils
from torchvision import datasets, transforms

import classify
import dataset
import loss

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        if not '...' in data:
            self.file.write(data)
        self.stdout.write(data)
        self.flush()
    def flush(self):
        self.file.flush()

class RandomIdentitySampler(torch.utils.data.sampler.Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    """

    def __init__(self, dataset, batch_size, num_instances):
        self.data_source = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        # changed according to the dataset
        for index, inputs in enumerate(self.data_source):
            self.index_dic[inputs[1]].append(index)

        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

def init_dataloader(args, file_path=None, batch_size=64, mode="gan"):
    tf = time.time()

    if args['dataset']['name'] == "celeba":
        data_set = dataset.CelebA(args, file_path, mode)
    elif args['dataset']['name'] == "mnist":
        # Expand chennel from 1 to 3 to fit pretrained models
        re_size = 64
        raw_data = datasets.MNIST(
            root=args["dataset"]["img_path"],
            train=mode != 'test',
            transform=transforms.Compose([
                transforms.Resize((re_size, re_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.expand(3, -1, -1)),
            ])
        )
        if args['dataset'].get('eval', False):
            # Use full dataset to train evaluation model
            data_set = raw_data
        else:
            # Take samples with label 0, 1, 2, 3, 4 as the private data
            indices = torch.where(raw_data.targets <= 4)[0]
            data_set = torch.utils.data.Subset(raw_data, indices)
        print(f"Load {len(data_set)} images")
    elif args['dataset']['name'] == "cifar10":
        re_size = 64
        raw_data = datasets.CIFAR10(
            root=args["dataset"]["img_path"],
            train=mode != 'test',
            transform=transforms.Compose([
                transforms.Resize((re_size, re_size)),
                transforms.ToTensor()
            ])
        )
        if args['dataset'].get('eval', False):
            # Use full dataset to train evaluation model
            data_set = raw_data
        else:
            # Take samples with label 0, 1, 2, 3, 4 as the private data
            indices = torch.where(torch.tensor(raw_data.targets) <= 4)[0]
            data_set = torch.utils.data.Subset(raw_data, indices)
        print(f"Load {len(data_set)} images")
    else:
        raise NotImplementedError(f"Dataset {args['dataset']['name']} not implemented")

    if 'bido' in args and args['dataset']['name'] == "celeba":
        sampler = RandomIdentitySampler(data_set, batch_size, 4)
        shuffle = None
    else:
        sampler = None
        shuffle = True
    data_loader = torch.utils.data.DataLoader(data_set,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                sampler=sampler,
                                drop_last=True,
                                num_workers=2,
                                pin_memory=True)
    interval = time.time() - tf
    print('Initializing data loader took %ds' % interval)
    
    return data_set, data_loader

def load_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)


def load_feature_extractor(net, state_dict):
    print("load_pretrained_feature_extractor!!!")
    net_state = net.state_dict()
    new_state_dict = OrderedDict()
    for name, param in state_dict.items():
        if "running_var" in name:
            new_state_dict[name] = param
            new_item = name.replace("running_var", "num_batches_tracked")
            new_state_dict[new_item] = torch.tensor(0)
        else:
            new_state_dict[name] = param

    for ((name, param), (new_name, mew_param)) in zip(net_state.items(), new_state_dict.items()):
        if "classifier" in new_name:
            break
        if "num_batches_tracked" in new_name:
            continue
        net_state[name].copy_(mew_param.data)

def init_model(model_name, n_classes, pretrained_path, bido=False):
    if bido and model_name not in ("VGG16", "IR18"):
        raise NotImplementedError(f"Model {model_name} not implemented for BiDO.")

    if model_name == "VGG16":
        if bido:
            net = classify.VGG16_BiDO(n_classes)
            BACKBONE_RESUME_ROOT = os.path.join(pretrained_path, "vgg16_bn-6c64b313.pth")
            checkpoint = torch.load(BACKBONE_RESUME_ROOT)
            load_feature_extractor(net, checkpoint)
            return net
        return classify.VGG16(n_classes)

    if model_name == "FaceNet64":
        net = classify.FaceNet64(n_classes)
        BACKBONE_RESUME_ROOT = os.path.join(pretrained_path, "backbone_ir50_ms1m_epoch120.pth")
        print("Loading Backbone Checkpoint ")
        load_state_dict(net.feature, torch.load(BACKBONE_RESUME_ROOT))
        return net

    if model_name == "IR152":
        net = classify.IR152(n_classes)
        BACKBONE_RESUME_ROOT = os.path.join(pretrained_path, "Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth")
        print("Loading Backbone Checkpoint ")
        load_state_dict(net.feature, torch.load(BACKBONE_RESUME_ROOT))
        return net

    if model_name == "IR18":
        if bido:
            return classify.IR18_BiDO(n_classes)
        return classify.IR18(n_classes)

    raise NotImplementedError(f"Model {model_name} not implemented.")


def init_optimizer(model_args, parameters):
    optimizer_name = model_args.get('optimizer', 'sgd')
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(params=parameters,
                               lr=model_args['lr'], 
                               momentum=model_args['momentum'], 
                               weight_decay=model_args['weight_decay'])
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(params=parameters,
                                lr=model_args['lr'],
                                weight_decay=model_args['weight_decay'])
    else:
        raise NotImplementedError(f'Optimizer {optimizer_name} not implemented.')

    if 'scheduler' in model_args:
        adjust_epochs = model_args['scheduler']['adjust_epochs']
        gamma = model_args['scheduler']['gamma']
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=adjust_epochs, gamma=gamma)
    else:
        scheduler = None

    return optimizer, scheduler

def init_criterion(negls, dataset_name='celeba'):
    if negls == 0:
        return torch.nn.CrossEntropyLoss().cuda()
    if dataset_name == 'mnist':
        ls_scheduler = loss.mnist_ls_scheduler
    elif dataset_name == 'cifar10':
        ls_scheduler = loss.cifar10_ls_scheduler
    else:
        ls_scheduler = loss.ls_scheduler
    return loss.NegLSCrossEntropyLoss(negls, scheduler=ls_scheduler)

def load_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data

def print_params(info, params, trap_info=None, bido_info=None):
    print('-----------------------------------------------------------------')
    print("Running time: %s" % datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    for i, (key, value) in enumerate(info.items()):
        print("%s: %s" % (key, str(value)))
    for i, (key, value) in enumerate(params.items()):
        print("%s: %s" % (key, str(value)))
    if trap_info:
        for i, (key, value) in enumerate(trap_info.items()):
            print("%s: %s" % (key, str(value)))
    if bido_info:
        for i, (key, value) in enumerate(bido_info.items()):
            print("%s: %s" % (key, str(value)))
    print('-----------------------------------------------------------------')
