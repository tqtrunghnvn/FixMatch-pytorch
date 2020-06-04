import os
import random
import numpy as np
import torch
import logging

def cdist(x, y):
    '''
    x, y: Tensor
    '''
    return torch.sqrt(torch.sum((x-y)**2))    

def cdists_slow(batch):
    '''
    batch: the size of (N, M)
            N: number of images
            M: number of classes
    '''
    N, M = batch.size()[0], batch.size()[1]
    dists = torch.zeros(N, N)
    
    for i in range(N):
        for j in range(N):
            dists[i,j] = cdist(batch[i], batch[j])
    
    return dists

def cdists_old(batch): # fast
    '''
    batch: the size of (N, M)
            N: number of images
            M: number of classes
    '''
    diff = torch.unsqueeze(batch, 0) - torch.unsqueeze(batch, 1)
    
    return torch.sqrt(torch.sum(diff*diff, axis=-1))

def cdists(batch): # fast --> solve the problem of gradient of sqrt becomes NaN when meeting 0 value.
    '''
    batch: the size of (N, M)
            N: number of images
            M: number of classes
    '''
    diff = torch.unsqueeze(batch, 0) - torch.unsqueeze(batch, 1)
    diff_2 = torch.sum(diff*diff, axis=-1)
#    itself = torch.eye(diff_2.size(0), dtype=torch.bool)
#    diff_2[itself] = 1.0

#    return torch.sqrt(diff_2)
    return diff_2

def batchhard(batch, idens, margin=0.1):
    # soft-margin
    dists = cdists(batch)

    same_iden_ = (torch.unsqueeze(idens,0) == torch.unsqueeze(idens,1))
    other_iden = ~same_iden_
    itself = ~torch.eye(same_iden_.size(0), dtype=torch.bool).cuda()
    same_iden = same_iden_ & itself
    infs = torch.ones_like(dists)*torch.Tensor([float('inf')]).cuda()

    dists_pos = torch.where(same_iden, dists, -infs)
    pos = torch.max(dists_pos, axis=1).values

    dists_neg = torch.where(other_iden, dists, infs)
    neg = torch.min(dists_neg, axis=1).values

    diff = (pos + margin) - neg
    diff = torch.log(torch.exp(diff)+1)

    return torch.mean(diff)

def batchhard2(batch, idens, margin=0.1):
    # use relu
    dists = cdists(batch)

    same_iden_ = (torch.unsqueeze(idens,0) == torch.unsqueeze(idens,1))
    other_iden = ~same_iden_
    itself = ~torch.eye(same_iden_.size(0), dtype=torch.bool).cuda()
    same_iden = same_iden_ & itself
    infs = torch.ones_like(dists)*torch.Tensor([float('inf')]).cuda()

    dists_pos = torch.where(same_iden, dists, -infs)
    pos = torch.max(dists_pos, axis=1).values

    dists_neg = torch.where(other_iden, dists, infs)
    neg = torch.min(dists_neg, axis=1).values

    diff = (pos + margin) - neg
    diff = torch.nn.functional.relu(diff)

    return torch.mean(diff)


def create_logger(out_dir, name, time_str):
    log_file = '{}_{}.log'.format(name, time_str)
    final_log_file = os.path.join(out_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger
