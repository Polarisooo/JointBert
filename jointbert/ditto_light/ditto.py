import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import sklearn.metrics as metrics
import argparse
import pandas as pd

from .dataset import DittoDataset
from torch.utils import data
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from apex import amp

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}


class DittoModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8, class_num=1000):
        super().__init__()
        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)

        self.device = device
        self.alpha_aug = alpha_aug

        # linear layer
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, 2 + class_num * 2)

    def forward(self, x1, x2=None):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1 = x1.to(self.device)  # (batch_size, seq_len)
        if x2 is not None:
            # MixDA
            x2 = x2.to(self.device)  # (batch_size, seq_len)
            enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size]  # (batch_size, emb_size)
            enc2 = enc[batch_size:]  # (batch_size, emb_size)

            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            enc = self.bert(x1)[0][:, 0, :]

        return self.fc(enc)  # .squeeze() # .sigmoid()


def evaluate(model, iterator, threshold=None):
    """Evaluate a model on a validation/test dataset

    Args:
        model (DMModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_p = []
    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            x, y, id1, id2 = batch
            logits = model(x)
            probs = logits[:, 0:2].softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        f1 = metrics.f1_score(all_y, pred)
        return f1
    else:
        best_th = 0.5
        f1 = 0.0  # metrics.f1_score(all_y, all_p)

        for th in np.arange(0.0, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred)
            if new_f1 > f1:
                f1 = new_f1
                best_th = th

        return f1, best_th


def train_step(train_iter, model, optimizer, scheduler, hp, class_num, criterion1, criterion2):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()

        if len(batch) == 4:
            x, y, id1, id2 = batch
            prediction = model(x)
        else:
            x1, x2, y, id1, id2 = batch
            prediction = model(x1, x2)

        loss1 = criterion(prediction[:, 0:2], y.to(model.device))
        loss2 = criterion1(prediction[:, 2:2 + class_num], id1.to(model.device))
        loss3 = criterion2(prediction[:, 2 + class_num:2 + class_num * 2], id2.to(model.device))
        loss = loss1 + loss2 + loss3

        if hp.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 10 == 0:  # monitoring
            print(f"step: {i}, loss: {loss.item()}")
        del loss


def train(trainset, validset, testset, run_tag, hp, classnum):
    """Train and evaluate the model

    Args:
        trainset (DittoDataset): the training set
        validset (DittoDataset): the validation set
        testset (DittoDataset): the test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)

    Returns:
        None
    """
    print(trainset)
    padder = trainset.pad
    # create the DataLoaders
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=padder)

    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size * 16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                batch_size=hp.batch_size * 16,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=padder)

    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DittoModel(device=device,
                       lm=hp.lm,
                       alpha_aug=hp.alpha_aug, class_num=classnum)
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=hp.lr)

    if hp.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    # logging with tensorboardX
    writer = SummaryWriter(log_dir=hp.logdir)
    ID1 = []
    ID2 = []
    for i, batch in enumerate(train_iter):

        if len(batch) == 4:
            x, y, id1, id2 = batch
        else:
            x1, x2, y, id1, id2 = batch
        ID1.extend(id1.cpu().numpy())
        ID2.extend(id2.cpu().numpy())

    multi_labels1_counts = pd.Series(ID1).value_counts()
    max_count_labels1 = multi_labels1_counts.max()
    multi_labels1_counts_dict = multi_labels1_counts.to_dict()
    for k, v in multi_labels1_counts_dict.items():
        multi_labels1_counts_dict[k] = max_count_labels1 / v
    for i in range(classnum):
        if i not in multi_labels1_counts_dict.keys():
            multi_labels1_counts_dict[i] = 1
    ordered_labels1 = OrderedDict(sorted(multi_labels1_counts_dict.items()))
    weights_labels1 = torch.Tensor(list(ordered_labels1.values())).to(model.device)

    multi_labels2_counts = pd.Series(ID2).value_counts()
    max_count_labels2 = multi_labels2_counts.max()
    multi_labels2_counts_dict = multi_labels2_counts.to_dict()
    for k, v in multi_labels2_counts_dict.items():
        multi_labels2_counts_dict[k] = max_count_labels2 / v
    for i in range(classnum):
        if i not in multi_labels2_counts_dict.keys():
            multi_labels2_counts_dict[i] = 1
    ordered_labels2 = OrderedDict(sorted(multi_labels2_counts_dict.items()))
    weights_labels2 = torch.Tensor(list(ordered_labels2.values())).to(model.device)
    criterion1 = nn.CrossEntropyLoss(weight=weights_labels1)
    criterion2 = nn.CrossEntropyLoss(weight=weights_labels2)
    best_dev_f1 = best_test_f1 = 0.0
    for epoch in range(1, hp.n_epochs + 1):
        # train
        model.train()
        train_step(train_iter, model, optimizer, scheduler, hp, classnum, criterion1, criterion2)

        # eval
        model.eval()
        dev_f1, th = evaluate(model, valid_iter)
        test_f1 = evaluate(model, test_iter, threshold=th)

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            if hp.save_model:
                # create the directory if not exist
                directory = os.path.join(hp.logdir, hp.task)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save the checkpoints for each component
                ckpt_path = os.path.join(hp.logdir, hp.task, 'model.pt')
                ckpt = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch}
                torch.save(ckpt, ckpt_path)

        print(f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")

        # logging
        scalars = {'f1': dev_f1,
                   't_f1': test_f1}
        writer.add_scalars(run_tag, scalars, epoch)

    writer.close()
