import einops
import torch
import torch.nn as nn
import numpy as np
from fast_transformer_pytorch.fast_transformer_pytorch import PreNorm

def default(x, y):
    return x if x is not None else y

class ConvBlock1D(nn.Module):
    def __init__(self, channels, outchannels=None, *, kernel_size=3, stride=None, padding=None, groups=1, dilation=1):
        outchannels = default(outchannels, channels)
        super().__init__()
        stride = default(stride, max(1, (kernel_size // 2) - 1))
        padding = default(padding, max(1, (kernel_size // 2)))
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.seq = nn.Sequential(
                                    nn.Conv1d(in_channels=channels, out_channels=outchannels,
                                              kernel_size=kernel_size, padding=padding),
                                    nn.BatchNorm1d(num_features=outchannels),
                                    nn.ReLU(inplace=True)
                                )

    def forward(self, x):
        return self.seq(x)


def batch_norm(x):
    """match Tensorflow batch norm settings"""
    return nn.BatchNorm1d(x, momentum=0.99, eps=0.001)


class RevBottleneck(nn.Module):
    expansion = 1
    '''
        Adapted from MemCNN's Resnet https://github.com/silvandeleemput/memcnn/blob/afd65198fb41e7339882ec55d35ad041edba13d7/memcnn/models/resnet.py
    '''
    def __init__(self, inchannels, channels=None, stride=1, downsample=None, noactivation=False, expansion=4):
        channels = default(channels, inchannels)
        super(RevBottleneck, self).__init__()
        makelayer = lambda: BottleneckSub(inchannels // 2, channels // 2, stride, noactivation, expansion=expansion)
        from memcnn import create_coupling, InvertibleModuleWrapper
        coupling = create_coupling(Fm=makelayer(), Gm=makelayer(), coupling='additive')
        self.revblock = InvertibleModuleWrapper(fn=coupling, keep_input=False)
        self.downsample = downsample
        self.stride = stride
        self.expansion = expansion

    def forward(self, x):
        if self.downsample is not None:
            out = self.bottleneck_sub(x)
            residual = self.downsample(x)
            out += residual
        else:
            out = self.revblock(x)
        return out


class BottleneckSub(nn.Module):
    '''
        Adapted from MemCNN's Resnet https://github.com/silvandeleemput/memcnn/blob/afd65198fb41e7339882ec55d35ad041edba13d7/memcnn/models/resnet.py
        Unlike their class, this increases then decreases the embedding dimension, returning dat of the same shape.
    '''
    def __init__(self, inchannels, channels=None, kernel_size=3, stride=1, noactivation=False, expansion=4):
        if channels is None:
            channels = inchannels
        super(BottleneckSub, self).__init__()
        self.noactivation = noactivation
        if not self.noactivation:
            self.bn1 = batch_norm(inchannels)
        self.conv1 = nn.Conv1d(inchannels, channels, kernel_size=1, bias=False)
        self.bn2 = batch_norm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=stride,
                               bias=False, padding='same')
        self.bn3 = batch_norm(channels)
        expanded_channels = channels * expansion
        self.conv3 = nn.Conv1d(channels, expanded_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.expansion = expansion
        self.bn4 = batch_norm(expanded_channels)
        self.conv4 = nn.Conv1d(expanded_channels, channels, kernel_size=kernel_size, bias=False, padding='same')

    def forward(self, x):
        if not self.noactivation:
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn4(x)
        x = self.conv4(x)
        return x


class RevConvBlock1D(nn.Module):
    def __init__(self, channels, additive=True, padding=None, kernel_size=3, dilation=1, groups=1, stride=1, depth=2):
        """
        Reversible Conv1DBlock
        Uses Affine coupling by default, but this can be switched to Additive via additive=
        Has padding, kernel_size, dilation, groups, and stride arguments which are passed to Convblock1D
        """
        super().__init__()
        if not additive:
            raise InvalidArgument("Only additive is supported currently. MemCNN seems to have a problem with affine.")
        from memcnn import AffineCoupling, AdditiveCoupling, InvertibleModuleWrapper
        if channels & 1:
            raise RuntimeError("Channels must be divisble by 2 in order to perform a reversible conv block")
        self.channels = channels
        halfchan = channels >> 1
        Coupling = [AffineCoupling, AdditiveCoupling][int(additive)]
        # Set coupling function, then create the invertible block
        makeblock = lambda : ConvBlock1D(channels=halfchan, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride, groups=groups)
        self.invmods = [Coupling(Fm=makeblock(), Gm=makeblock()) for i in range(depth)]
        self.wrappers = [InvertibleModuleWrapper(fn=x, keep_input=True, keep_input_inverse=True) for x in self.invmods]
        self.invmodw = nn.Sequential(*self.wrappers)

    def forward(self, x):
        return self.invmodw(x)


class RevConvNetwork1D(nn.Module):
    def __init__(self, inchannels, channels=None, padding=None, kernel_size=3, revdepth=3, totaldepth=3, noactivation=False):
        super().__init__()
        import itertools
        channels = default(channels, inchannels)
        layers = [ConvBlock1D(inchannels, channels)]
        for _ in range(totaldepth):
            layers.append(RevConvBlock1D(channels, padding=padding, kernel_size=kernel_size, depth=revdepth))
            layers.append(BottleneckSub(channels, kernel_size=kernel_size, noactivation=noactivation))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class RevConvClassifier(nn.Module):
    def __init__(self, inchannels, num_classes, *, channels=None, padding=None, kernel_size=3, revdepth=3, totaldepth=3, noactivation=False, softmax=None):
        super().__init__()
        channels = default(channels, inchannels)
        self.net = RevConvNetwork1D(inchannels=inchannels, channels=channels, padding=padding, kernel_size=kernel_size, revdepth=revdepth, totaldepth=totaldepth, noactivation=noactivation)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)
        softmax = default(softmax, nn.Softmax(-1))

    def logits(self, x):
        '''
        Returns logits
        '''
        embeddings = self.net(x)
        pooled = self.pool(embeddings).squeeze(-1) # Average across data
        return self.fc(pooled)

    def forward(self, x):
        logits = self.logits(x)
        


class ResConvBlock1D(nn.Module):
    def __init__(self, channels, additive=True, padding=None, kernel_size=3, dilation=1, groups=1, stride=1, depth=2, downsample=None):
        """
        Reversible Conv1DBlock
        Uses Affine coupling by default, but this can be switched to Additive via additive=
        Has padding, kernel_size, dilation, groups, and stride arguments which are passed to Convblock1D
        """
        super().__init__()
        self.block = RevConvBlock1D(channels, additive=additive, padding=padding, kernel_size=kernel_size, dilation=dilation, groups=groups, stride=stride, depth=depth)
        self.downsample = downsample
        self.expansion = 1

    def forward(self, x):
        res = x
        out = self.block(x)
        if self.downsample is not None:
            res = self.downsample(x)
        out += res
        return out
