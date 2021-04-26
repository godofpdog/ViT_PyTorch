import math
import torch
import numpy as np 
from .utils import to_numpy
from torch import sigmoid
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ExponentialLR, StepLR


def calc_accuracy(outputs, targets):
    if outputs.size(1) == 1:
        preds = (sigmoid(outputs) > 0.5) * 1
    else:
        preds = outputs.argmax(-1)

    return torch.mean(1. * (preds == targets)).cpu()


def get_criterion(loss):
    assert loss in ('bce', 'ce')

    if loss == 'bce':
        return torch.nn.BCEWithLogitsLoss()
    else:
        return torch.nn.CrossEntropyLoss()


def get_optimizer(model, args):
    # only support Adam now
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.init_lr, 
                                 betas=(args.beta1, args.beta2), 
                                 eps=1e-8, 
                                 weight_decay=args.weight_decay)
    
    return optimizer


def train_epoch(model, loader, criterion, optimizer, meter, device, epoch):
    criterion.to(device)
    model.train()

    for step, data in enumerate(loader):
        images, targets = data 
        images = images.to(device)
        targets = targets.to(device).float()
        targets = targets.unsqueeze(-1)
        outputs = model(images)
        loss = criterion(outputs, targets)
        acc = calc_accuracy(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        _show_result(epoch, step, len(loader), loss.item(), acc, True)
        _update_meter(meter, loss.item(), acc)

    return meter 


def eval_epoch(model, loader, criterion, meter, device, epoch):
    criterion.to(device)
    model.eval()

    with torch.no_grad():
        for step, data in enumerate(loader):
            images, targets = data 
            images = images.to(device)
            targets = targets.to(device).float()
            targets = targets.unsqueeze(-1)
            outputs = model(images)
            loss = criterion(outputs, targets)
            acc = calc_accuracy(outputs, targets)
            _show_result(epoch, step, len(loader), loss.item(), acc, False)
            _update_meter(meter, loss.item(), acc)

    return meter 


def _show_result(epoch, step, total_batch, loss, acc, is_train):
    task = 'Train' if is_train else 'Eval'

    print('[Epoch %s][%d/%d][%s][Loss %.6f][Acc %.6f]' \
        % (str(epoch), step, total_batch, task, loss, acc))

    return None


def _update_meter(meter, loss, acc):
    updates = {'loss': loss, 'acc': acc}
    meter.update(updates)

    return None 


def _calc_warmup_lr(epoch, warmup_epochs):
    return float(epoch) / float(max(1., warmup_epochs))


class WarmupScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        super(WarmupScheduler, self).__init__(optimizer, 
                                              self.lr_lambda,
                                              last_epoch=last_epoch)

    def lr_lambda(self, epoch):
        if epoch < self.warmup_epochs:
            return _calc_warmup_lr(epoch, self.warmup_epochs)
        else:
            return 1.


class ConstantScheduler(LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantScheduler, self).__init__(optimizer, 
                                                self.lr_lambda,
                                                last_epoch=last_epoch)

    def lr_lambda(self, epoch):
        return 1.


def get_scheduler(optimizer, args):
    if args.scheduler is not None:
        if args.scheduler == 'cosine':
            return CosineAnnealingLR(optimizer, args.t_max, args.eta_min)
        elif args.scheduler == 'step':
            return StepLR(optimizer, args.step_size, args.gamma)
        elif args.scheduler == 'exp':
            return ExponentialLR(optimizer, args.gamma)
        else:
            raise ValueError('Invalid scheduler.')
    else:
        return ConstantScheduler(optimizer)


class EarlyStopper:
    def __init__(self, monitor, patient, min_delta=0):
        assert monitor in ('acc', 'loss')

        if monitor == 'loss':
            self._best_metric = np.Inf
        else:
            self._best_metric = -np.Inf

        self.monitor = monitor
        self.patient = patient
        self.min_delta = min_delta
        self._not_improved_cnt = 0 
        self._is_best = False 
        self._is_stop = False

    def step(self, metric):
        if self._is_improved(metric):
            self._is_best = True 
            self._not_improved_cnt = 0
        else:
            self._is_best = False
            self._not_improved_cnt += 1
        
        if self._not_improved_cnt >= self.patient:
            self._is_stop = True

    def _is_improved(self, metric):
        if self._best_metric is None:
            self._best_metric = metric
            _is_improved = True
        else:
            if self.monitor == 'acc':
                _is_improved = metric > self._best_metric + self.min_delta
            else:
                _is_improved = metric < self._best_metric - self.min_delta
            
        if _is_improved:
            print('Monitored metric `%s` has improved from `%.6f` to `%.6f`' 
                %(self.monitor, self._best_metric, metric))

        return _is_improved

    @property
    def is_best(self):
        return self._is_best

    @property
    def is_early_stop(self):
        return self._is_stop

    @property
    def not_improved_cnt(self):
        return self._not_improved_cnt