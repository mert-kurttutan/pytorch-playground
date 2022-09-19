

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.checkpoint as cp

# local modules
from ..utils import CfgNode as CN


# __all__ = ['densenet']



def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class BottleneckUnit(nn.Module):
    def __init__(self, in_channel, growth_rate, expansion, p_dropout, activation,efficient=False):
        super(BottleneckUnit, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.activation = activation
        self.conv1 = nn.Conv2d(in_channel, expansion * growth_rate,
                        kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(expansion * growth_rate)

        self.conv2 = nn.Conv2d(expansion * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)
        self.p_dropout = p_dropout
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.bn1, self.activation, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.activation(self.bn2(bottleneck_output)))
        if self.p_dropout > 0:
            new_features = F.dropout(new_features, p=self.p_dropout, training=self.training)
        return new_features

class _InitBlock(nn.Module):

    def __init__(self, in_channel, out_channel, small_inputs, activation):
        super(_InitBlock, self).__init__()

        self.activation = activation
        self.small_inputs = small_inputs
        if small_inputs:
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)

        else:
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn = nn.BatchNorm2d(out_channel)
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False)

    def forward(self, x):

        out = self.conv(x)
        if not self.small_inputs:
            out = self.pool(self.activation(self.bn(out)))

        return out


class _TransitionBlock(nn.Module):
    def __init__(self, in_channel, out_channel, activation):
        super(_TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channel)
        self.activation = activation
        self.conv = nn.Conv2d(in_channel, out_channel,
                                          kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        
        out = self.activation(self.bn(x))
        out = self.pool(self.conv(out))

        return out





class _DenseBlock(nn.Module):
    def __init__(self, dense_unit, n_unit, in_channel, expansion, growth_rate, p_dropout, activation, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(n_unit):
            layer = dense_unit(
                in_channel=in_channel + i * growth_rate,
                growth_rate=growth_rate,
                expansion=expansion,
                p_dropout=p_dropout,
                efficient=efficient,
                activation=activation,
            )
            self.add_module('dense_unit%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        n_channel (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        n_class (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """

    @staticmethod
    def get_activation(name: str):

        activation_map = {
            "relu": F.relu,
            "elu": F.elu,
            "gelu": F.gelu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }

        return activation_map[name]

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd)
        # must be given in the config
        C.model_type = 'densenet'
        C.n_channel = 32
        C.img_channel = 3
        C.activation = "gelu"
        C.fc_pdrop = 0.16
        C.n_class = 10

        return C
    def __init__(
        self,
        config: CN,
        unit_module=BottleneckUnit,
    ):

        self.activation = (
            F.relu if config.activation is None
            else self.get_activation(config.activation)
        )


        super(DenseNet, self).__init__()
        assert 0 < config.compression <= 1, 'compression of densenet should be between 0 and 1'

        self.features = nn.Sequential()

        # First convolution
        self.features.add_module("init_block",_InitBlock(config.img_channel, config.n_channel, config.small_inputs, self.activation))

        # Each denseblock
        n_feature = config.n_channel
        for i, n_unit in enumerate(config.n_blocks):
            block = _DenseBlock(
                dense_unit=unit_module,
                n_unit=n_unit,
                in_channel=n_feature,
                expansion=config.expansion,
                growth_rate=config.growth_rate,
                p_dropout=config.p_dropout,
                efficient=config.efficient,
                activation=self.activation,
            )
            self.features.add_module('dense_block_%d' % (i + 1), block)
            n_feature = n_feature + n_unit * config.growth_rate
            if i != len(config.n_blocks) - 1:
                trans = _TransitionBlock(in_channel=n_feature,
                                        out_channel=int(n_feature * config.compression),
                                        activation=self.activation)
                self.features.add_module('transition_%d' % (i + 1), trans)
                n_feature = int(n_feature * config.compression)

        # Final batch norm
        self.bn_final = nn.BatchNorm2d(n_feature)

        # fully connect / classifer layer
        self.fc= nn.Linear(n_feature, config.n_class)

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'bn' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'bn' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'fc' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        out = self.features(x)
        out = self.activation(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def _get_parameter_config(self):

        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        Then returns buckets for decay and no-decay parameters
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

        return decay, no_decay, param_dict

    def _get_optim(optim_groups, train_config):

        if train_config.optim == "adamw":
            optimizer = torch.optim.AdamW(optim_groups, lr=train_config.lr, betas=train_config.betas)
        
        elif train_config.optim == "sgd":
            optimizer = torch.optim.SGD(optim_groups, lr=train_config.lr, momentum=train_config.momentum)

        else:
            raise ValueError(f"The optimizer for {train_config.optim} is not implemented")


        return optimizer
        

    def configure_optimizers(self, train_config):

        decay, no_decay, param_dict = self._get_parameter_config()

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = DenseNet._get_optim(optim_groups, train_config)

        
        return optimizer