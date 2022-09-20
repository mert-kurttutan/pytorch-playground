"""
Trains a character-level language model.
"""

import os
import sys
import json

import torch

import src.data as data
from src.model import DenseNet, BottleneckUnit
from src.utils import set_seed, setup_logging, CfgNode as CN 

# create a Trainer object
from src.train import Trainer, loss_fn, accuracy

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/densenet_v4'

    # data
    C.train_data = data.get_default_config_cifar10()
    C.eval_data = data.get_default_config_cifar10()
    C.eval_data.augmentation = []

    # model
    C.model = DenseNet.get_default_config()
    C.model.model_type = 'densenet'
    C.model.growth_rate = 24
    C.model.n_channel = C.model.growth_rate * 2
    C.model.img_channel = 3
    C.model.activation = "gelu"
    C.model.p_dropout = 0.16
    C.model.n_class = 10
    C.model.compression = 0.5
    C.model.expansion = 4
    C.model.small_inputs = True
    C.model.efficient = True
    C.model.n_blocks = [31,31,31]


    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.epochs = 120
    C.trainer.batch_size = 64
    C.trainer.eval_batch_size = 512
    C.trainer.lr = 1e-3 * (C.trainer.batch_size/32)
    C.trainer.warmup_epochs = C.trainer.epochs // 10   # % 10 of training is warm-up
    C.trainer.warmup_ratio = 10 #C.trainer.warmup_epochs
    C.trainer.init_lr = C.trainer.lr / C.trainer.warmup_ratio
    C.trainer.min_lr = C.trainer.lr / 1e2
    C.trainer.optim = "sgd"
    
    C.trainer.momentum = 0.9
    
    C.trainer.weight_decay = 1e-4

    C.trainer.max_iters = int(C.trainer.epochs*(50000/C.trainer.batch_size))   
    C.trainer.n_worker = 2
    
    C.trainer.grad_norm_clip = 1
    C.trainer.cudnn_benchmark = True

    return C

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    train_dataset = data.get_dataset_cifar10("train", config.train_data)

    # construct the test dataset
    eval_dataset = data.get_dataset_cifar10("test", config.eval_data)

    # construct the model resnet 50
    model = DenseNet(config.model, BottleneckUnit)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, loss_fn, accuracy, train_dataset, eval_dataset)


    decay_steps = config.trainer.epochs - config.trainer.warmup_epochs
    linear_warm = lambda epoch: ((epoch+1)/config.trainer.warmup_epochs) * config.trainer.warmup_ratio

    scheduler1 = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lr_lambda=linear_warm)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=decay_steps, eta_min=config.trainer.min_lr)


    def batch_end_callback(trainer):
        if trainer.iter_num % 100 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

    def epoch_end_callback(trainer):

        # apply learning rate schedule, cosine decay with lr warmup
        if trainer.epoch_num < config.trainer.warmup_epochs:
            scheduler1.step()
        else:
            scheduler2.step()

        print(f"epoch_dt {trainer.epoch_dt * 1000:.2f}ms; epoch {trainer.epoch_num}")

        trainer.eval_model("train")
        train_loss = trainer.metric["train_loss"][-1]
        train_acc = trainer.metric["train_accuracy"][-1]
        print(f"train loss {train_loss:.5f}, train acc {train_acc:.5f}")

        if trainer.eval_dataset is not None:
            trainer.eval_model("test")
            val_loss = trainer.metric["val_loss"][-1]
            val_acc = trainer.metric["val_accuracy"][-1]
            print(f"val loss {val_loss:.5f}, val acc {val_acc:.5f}")

        # save the latest model
        print("saving model...")
        ckpt_path = os.path.join(config.system.work_dir, "model.pt")
        torch.save(model.state_dict(), ckpt_path)

        # save the metric results
        with open(os.path.join(config.system.work_dir, 'metric.json'), 'w') as f:
            f.write(json.dumps(trainer.metric, indent=4))


    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.set_callback('on_epoch_end', epoch_end_callback)



    # run the optimization
    trainer.run()
