"""
   Baseline CNN, losss function and metrics
   Also customizes knowledge distillation (KD) loss function here
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import get_default_config
from ..utils.utils import CfgNode as CN


class BaseCNN(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.
    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    @staticmethod
    def get_activation(name: str):

        activation_map = {"relu": F.relu,
                        "elu": F.elu,
                        "gelu": F.gelu,
                        "tanh": torch.tanh,
                        "sigmoid": torch.sigmoid}

        return activation_map[name]

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'base_cnn'
        C.n_channel = 32
        C.img_channel = 3
        C.activation = "relu"
        C.fc_pdrop = 0.16
        C.n_class = 10

        return C

    def __init__(self, config):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:
        Args:
            params: (Params) contains n_channels
        """
        super(BaseCNN, self).__init__()
        self.n_channel = config.n_channel



        self.activation = F.relu if config.activation is None else self.get_activation(config.activation)
        
        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        self.conv1 = nn.Conv2d(config.img_channel, self.n_channel, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.n_channel)
        self.conv2 = nn.Conv2d(self.n_channel, self.n_channel*2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.n_channel*2)
        self.conv3 = nn.Conv2d(self.n_channel*2, self.n_channel*4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.n_channel*4)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(4*4*self.n_channel*4, self.n_channel*4)
        self.fcbn1 = nn.BatchNorm1d(self.n_channel*4)
        self.fc2 = nn.Linear(self.n_channel*4, config.n_class)       
        self.dropout_rate = config.fc_pdrop

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.
        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 32 x 32 .
        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.
        Note: the dimensions after each step are provided
        """
        #                                                  -> batch_size x 3 x 32 x 32
        # we apply the convolution layers, followed by batch normalisation, maxpool and activation x 3
        s = self.bn1(self.conv1(s))                         # batch_size x n_channels x 32 x 32
        s = self.activation(F.max_pool2d(s, 2))                      # batch_size x n_channels x 16 x 16
        s = self.bn2(self.conv2(s))                         # batch_size x n_channels*2 x 16 x 16
        s = self.activation(F.max_pool2d(s, 2))                      # batch_size x n_channels*2 x 8 x 8
        s = self.bn3(self.conv3(s))                         # batch_size x n_channels*4 x 8 x 8
        s = self.activation(F.max_pool2d(s, 2))                      # batch_size x n_channels*4 x 4 x 4

        # flatten the output for each image
        s = s.view(-1, 4*4*self.n_channel*4)             # batch_size x 4*4*n_channels*4

        # apply 2 fully connected layers with dropout
        s = self.activation(self.fcbn1(self.fc1(s)))
        s = F.dropout(s, p=self.dropout_rate, training=self.training)    # batch_size x self.n_channels*4
        s = self.fc2(s)                                     # batch_size x 10

        return s


    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        optim_groups = [{
                    "params": self.parameters()
            }]
        optimizer = torch.optim.Adam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)

        return optimizer


    def configure_optimizers_2(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer