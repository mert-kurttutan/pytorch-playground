'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.utils import CfgNode as CN


class BasicBlock(nn.Module):
    # first conv uses different strides to adapt to channel dimension change
    # second conv retains the same channel dim
    # see figure 3 in the original resnet paper

    def __init__(self, in_channel, out_channel, activation=None):
        super(BasicBlock, self).__init__()

        stride = 1 if in_channel == out_channel else 2
        self.activation = F.relu if activation is None else activation
        self.conv1 = nn.Conv2d(
            in_channel, out_channel,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(
            out_channel, out_channel,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channel)

        self.shortcut = nn.Identity()
        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channel, out_channel,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class BottleneckBlock(nn.Module):

    def __init__(
        self, in_channel, out_channel,
        bottleneck_channel=None, activation=None
    ):
        super(BottleneckBlock, self).__init__()

        self.activation = F.relu if activation is None else activation
        stride = 1 if in_channel == out_channel else 2
        if bottleneck_channel is None:
            bottleneck_channel = out_channel // 4

        self.conv1 = nn.Conv2d(
            in_channel, bottleneck_channel,
            kernel_size=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(bottleneck_channel)
        self.conv2 = nn.Conv2d(
            bottleneck_channel, bottleneck_channel,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channel)
        self.conv3 = nn.Conv2d(
            bottleneck_channel, out_channel,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channel)

        self.shortcut = nn.Identity()
        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channel, out_channel,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet(nn.Module):

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
        C.model_type = 'resnet'
        C.n_channel = 64
        C.img_channel = 3
        C.activation = "relu"
        C.fc_pdrop = 0.16
        C.n_class = 10

        return C

    def __init__(
        self,
        config: CN,
        block=BottleneckBlock,
    ):
        super(ResNet, self).__init__()

        n_blocks = config.n_blocks

        self.activation = (
            F.relu if config.activation is None
            else self.get_activation(config.activation)
        )

        self.conv1 = nn.Conv2d(
            config.img_channel, config.n_channel,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(config.n_channel)
        self.layer1 = self._make_layer(
            block, config.n_channel, config.n_channel, n_blocks[0]
        )
        self.layer2 = self._make_layer(
            block, config.n_channel, config.n_channel*2, n_blocks[1]
        )
        self.layer3 = self._make_layer(
            block, config.n_channel*2, config.n_channel*4, n_blocks[2]
        )
        self.layer4 = self._make_layer(
            block, config.n_channel*4, config.n_channel*8, n_blocks[3]
        )
        self.linear = nn.Linear(config.n_channel*8, config.n_class)

    def _make_layer(self, block, in_channel, out_channel, n_blocks):
        channel_dims = (
            [(in_channel, out_channel)]
            + [(out_channel, out_channel)]*(n_blocks-1)
        )

        layers = []
        for c_dim in channel_dims:
            layers.append(block(*c_dim, activation=self.activation))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def configure_optimizers(self, train_config):

        optim_groups = [{
                    "params": self.parameters()
            }]

        optimizer = torch.optim.Adam(
            optim_groups,
            lr=train_config.learning_rate, betas=train_config.betas
        )

        return optimizer

    def configure_optimizers_2(self, train_config):
        """
        This long function is unfortunately doing something
        very simple and is being very defensive:
        We are separating out all parameters of the model
        into two buckets: those that will experience
        weight decay for regularization and those
        that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and
        # won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name
                # random note: because named_modules
                # and named_parameters are recursive
                # we will see the same tensors p many many times.
                # but doing it this way allows us to know
                # which parent module any tensor p belongs to...
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

        assert (
            len(inter_params) == 0
        ), (
            f"parameters {str(inter_params)} made it ",
            "into both decay/no_decay sets!"
        )

        assert (
            len(param_dict.keys() - union_params) == 0
        ), (
            f"parameters {str(param_dict.keys() - union_params)} ",
            "were not separated into either decay/no_decay set!"

        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer


def ResNet18():
    default_config = ResNet.get_default_config()
    default_config.n_blocks = [2, 2, 2, 2]
    return ResNet(default_config, BasicBlock)


def ResNet34():
    default_config = ResNet.get_default_config()
    default_config.n_blocks = [3, 4, 6, 3]
    return ResNet(default_config, BasicBlock)


def ResNet50():
    default_config = ResNet.get_default_config()
    default_config.n_blocks = [3, 4, 6, 3]
    return ResNet(default_config, BottleneckBlock)


def ResNet101():
    default_config = ResNet.get_default_config()
    default_config.n_blocks = [3, 4, 23, 3]
    return ResNet(default_config, BottleneckBlock)


def ResNet152():
    default_config = ResNet.get_default_config()
    default_config.n_blocks = [3, 8, 36, 3]
    return ResNet(default_config, BottleneckBlock)
