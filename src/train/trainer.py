"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from ..utils import CfgNode as CN

class Trainer:

    # TODO: add resume training option, using default config files
    # model.pt, config.json

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.n_worker = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 16
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0

        return C

    def __init__(self, config, model, train_dataset, eval_dataset=None):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset

        self.eval_dataset = eval_dataset

        self.callbacks = defaultdict(list)

        if config.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

        # epoch iteration
        self.epoch_num = 0
        self.epoch_time = 0.0
        self.epoch_dt = 0.0


        # metrics of training runs
        self.metric = {"train_loss": [], "train_accuracy": [],
                        "val_loss": [], "val_accuracy": [], }


        # setup the optimizer
        self.optimizer = self.model.configure_optimizers_2(self.config)

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    @torch.no_grad()
    def eval_model(self, split: str):

        evaluation_batch_size = self.config.eval_batch_size
        
        dataset = self.train_dataset if split=="train" else self.eval_dataset

        steps_per_epoch = math.ceil(len(dataset) / evaluation_batch_size)

        loader = DataLoader(dataset, batch_size=evaluation_batch_size, num_workers=self.config.n_worker,)

        data_iter = iter(loader)

        running_loss = 0
        running_accuracy = 0

        self.model.eval()
        for batch in data_iter:

            batch = [t.to(self.device) for t in batch]
            x, y = batch


            # forward the model
            logits = self.model(x)

            running_loss += loss_fn(logits, y).item()

            running_accuracy += accuracy(logits, y).item()

        if split=="train":
            self.metric["train_loss"].append(running_loss / steps_per_epoch)
            self.metric["train_accuracy"].append(running_accuracy / steps_per_epoch)

        else:
            self.metric["val_loss"].append(running_loss / steps_per_epoch)
            self.metric["val_accuracy"].append(running_accuracy / steps_per_epoch)

            
    def run(self):

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            #sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=True,
            pin_memory=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.n_worker,
        )

        # batch iteration
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)


        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                # end of epoch

                self.epoch_num += 1
                self.epoch_dt = tnow - self.epoch_time
                self.epoch_time = tnow

                self.trigger_callbacks('on_epoch_end')

                data_iter = iter(train_loader)
                batch = next(data_iter)

            self.model.train()

            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits = self.model(x)


            self.loss = loss_fn(logits, y)

            # backprop and update the parameters
            self.model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')


            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow
            


            # termination conditions
            if self.config.max_iters is not None and self.iter_num >= self.config.max_iters:
                break




def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.
    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns:
        loss (Variable): cross entropy loss for all images in the batch
    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    return nn.CrossEntropyLoss()(outputs, labels)


def loss_fn_kd(outputs, labels, teacher_outputs, config):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = config.alpha
    T = config.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) output of the model
        labels: (np.ndarray) [0, 1, ..., n_class-1]
    Returns: (float) accuracy in [0,1]
    """
    outputs = torch.argmax(outputs, axis=1)
    return torch.mean((outputs == labels).float())