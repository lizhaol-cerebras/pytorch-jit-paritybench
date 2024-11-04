import sys
_module = sys.modules[__name__]
del sys
main = _module
fid_score = _module
flist = _module
inception = _module
metrics = _module
src = _module
config = _module
dataset = _module
edge_connect = _module
loss = _module
metrics = _module
models = _module
networks = _module
utils = _module
test = _module
train = _module

from paritybench._paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import random


import numpy as np


import torch


from scipy import linalg


from torch.autograd import Variable


from torch.nn.functional import adaptive_avg_pool2d


import torch.nn as nn


import torch.nn.functional as F


from torchvision import models


import scipy


import torchvision.transforms.functional as F


from torch.utils.data import DataLoader


import torchvision.models as models


import torch.optim as optim


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {(64): 0, (192): 1, (768): 2, (2048): 3}

    def __init__(self, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=True, normalize_input=True, requires_grad=False):
        """Build pretrained InceptionV3
        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, normalizes the input to the statistics the pretrained
            Inception network expects
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        inception = models.inception_v3(pretrained=True)
        block0 = [inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
            self.blocks.append(nn.Sequential(*block1))
        if self.last_needed_block >= 2:
            block2 = [inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e]
            self.blocks.append(nn.Sequential(*block2))
        if self.last_needed_block >= 3:
            block3 = [inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1))]
            self.blocks.append(nn.Sequential(*block3))
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp
        if self.resize_input:
            x = F.upsample(x, size=(299, 299), mode='bilinear')
        if self.normalize_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            if idx == self.last_needed_block:
                break
        return outp


class AdversarialLoss(nn.Module):
    """
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        """
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()
        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if type == 'nsgan':
            self.criterion = nn.BCELoss()
        elif type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()
        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class VGG19(torch.nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()
        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()
        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()
        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()
        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()
        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])
        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])
        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])
        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])
        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])
        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])
        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])
        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])
        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])
        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])
        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])
        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])
        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])
        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])
        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])
        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)
        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)
        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)
        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)
        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)
        out = {'relu1_1': relu1_1, 'relu1_2': relu1_2, 'relu2_1': relu2_1, 'relu2_2': relu2_2, 'relu3_1': relu3_1, 'relu3_2': relu3_2, 'relu3_3': relu3_3, 'relu3_4': relu3_4, 'relu4_1': relu4_1, 'relu4_2': relu4_2, 'relu4_3': relu4_3, 'relu4_4': relu4_4, 'relu5_1': relu5_1, 'relu5_2': relu5_2, 'relu5_3': relu5_3, 'relu5_4': relu5_4}
        return out


class StyleLoss(nn.Module):
    """
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G

    def __call__(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))
        return style_loss


class PerceptualLoss(nn.Module):
    """
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])
        return content_loss


class EdgeAccuracy(nn.Module):
    """
    Measures the accuracy of the edge map
    """

    def __init__(self, threshold=0.5):
        super(EdgeAccuracy, self).__init__()
        self.threshold = threshold

    def __call__(self, inputs, outputs):
        labels = inputs > self.threshold
        outputs = outputs > self.threshold
        relevant = torch.sum(labels.float())
        selected = torch.sum(outputs.float())
        if relevant == 0 and selected == 0:
            return torch.tensor(1), torch.tensor(1)
        true_positive = ((outputs == labels) * labels).float()
        recall = torch.sum(true_positive) / (relevant + 1e-08)
        precision = torch.sum(true_positive) / (selected + 1e-08)
        return precision, recall


class PSNR(nn.Module):

    def __init__(self, max_val):
        super(PSNR, self).__init__()
        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()
        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)
        if mse == 0:
            return torch.tensor(0)
        return self.max_val - 10 * torch.log(mse) / self.base10


class BaseModel(nn.Module):

    def __init__(self, name, config):
        super(BaseModel, self).__init__()
        self.name = name
        self.config = config
        self.iteration = 0
        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            None
            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)
            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            None
            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)
            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        None
        torch.save({'iteration': self.iteration, 'generator': self.generator.state_dict()}, self.gen_weights_path)
        torch.save({'discriminator': self.discriminator.state_dict()}, self.dis_weights_path)


class BaseNetwork(nn.Module):

    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        """
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        """

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)
        self.apply(init_func)


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


class Discriminator(BaseNetwork):

    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.conv1 = self.features = nn.Sequential(spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm), nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm), nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm), nn.LeakyReLU(0.2, inplace=True))
        self.conv4 = nn.Sequential(spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm), nn.LeakyReLU(0.2, inplace=True))
        self.conv5 = nn.Sequential(spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm))
        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)
        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):

    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(nn.ReflectionPad2d(dilation), spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm), nn.InstanceNorm2d(dim, track_running_stats=False), nn.ReLU(True), nn.ReflectionPad2d(1), spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm), nn.InstanceNorm2d(dim, track_running_stats=False))

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class EdgeGenerator(BaseNetwork):

    def __init__(self, residual_blocks=8, use_spectral_norm=True, init_weights=True):
        super(EdgeGenerator, self).__init__()
        self.encoder = nn.Sequential(nn.ReflectionPad2d(3), spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0), use_spectral_norm), nn.InstanceNorm2d(64, track_running_stats=False), nn.ReLU(True), spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm), nn.InstanceNorm2d(128, track_running_stats=False), nn.ReLU(True), spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), use_spectral_norm), nn.InstanceNorm2d(256, track_running_stats=False), nn.ReLU(True))
        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)
        self.decoder = nn.Sequential(spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm), nn.InstanceNorm2d(128, track_running_stats=False), nn.ReLU(True), spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm), nn.InstanceNorm2d(64, track_running_stats=False), nn.ReLU(True), nn.ReflectionPad2d(3), nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0))
        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


class EdgeModel(BaseModel):

    def __init__(self, config):
        super(EdgeModel, self).__init__('EdgeModel', config)
        generator = EdgeGenerator(use_spectral_norm=True)
        discriminator = Discriminator(in_channels=2, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)
        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)
        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        self.gen_optimizer = optim.Adam(params=generator.parameters(), lr=float(config.LR), betas=(config.BETA1, config.BETA2))
        self.dis_optimizer = optim.Adam(params=discriminator.parameters(), lr=float(config.LR) * float(config.D2G_LR), betas=(config.BETA1, config.BETA2))

    def process(self, images, edges, masks):
        self.iteration += 1
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0
        dis_input_real = torch.cat((images, edges), dim=1)
        dis_input_fake = torch.cat((images, outputs.detach()), dim=1)
        dis_real, dis_real_feat = self.discriminator(dis_input_real)
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2
        gen_input_fake = torch.cat((images, outputs), dim=1)
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
        gen_loss += gen_gan_loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss
        logs = [('l_d1', dis_loss.item()), ('l_g1', gen_gan_loss.item()), ('l_fm', gen_fm_loss.item())]
        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):
        edges_masked = edges * (1 - masks)
        images_masked = images * (1 - masks) + masks
        inputs = torch.cat((images_masked, edges_masked, masks), dim=1)
        outputs = self.generator(inputs)
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        if dis_loss is not None:
            dis_loss.backward()
        self.dis_optimizer.step()
        if gen_loss is not None:
            gen_loss.backward()
        self.gen_optimizer.step()


class InpaintGenerator(BaseNetwork):

    def __init__(self, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()
        self.encoder = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0), nn.InstanceNorm2d(64, track_running_stats=False), nn.ReLU(True), nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(128, track_running_stats=False), nn.ReLU(True), nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(256, track_running_stats=False), nn.ReLU(True))
        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(128, track_running_stats=False), nn.ReLU(True), nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(64, track_running_stats=False), nn.ReLU(True), nn.ReflectionPad2d(3), nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0))
        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2
        return x


class InpaintingModel(BaseModel):

    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)
        generator = InpaintGenerator()
        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)
        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)
        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        self.gen_optimizer = optim.Adam(params=generator.parameters(), lr=float(config.LR), betas=(config.BETA1, config.BETA2))
        self.dis_optimizer = optim.Adam(params=discriminator.parameters(), lr=float(config.LR) * float(config.D2G_LR), betas=(config.BETA1, config.BETA2))

    def process(self, images, edges, masks):
        self.iteration += 1
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0
        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)
        dis_fake, _ = self.discriminator(dis_input_fake)
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss
        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss
        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss
        logs = [('l_d2', dis_loss.item()), ('l_g2', gen_gan_loss.item()), ('l_l1', gen_l1_loss.item()), ('l_per', gen_content_loss.item()), ('l_sty', gen_style_loss.item())]
        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):
        images_masked = images * (1 - masks).float() + masks
        inputs = torch.cat((images_masked, edges), dim=1)
        outputs = self.generator(inputs)
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()
        gen_loss.backward()
        self.gen_optimizer.step()


import torch
from torch.nn import MSELoss, ReLU
from paritybench._paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdversarialLoss,
     lambda: ([], {}),
     lambda: ([], {'outputs': torch.rand([4, 4]), 'is_real': 4}),
     True),
    (Discriminator,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (ResnetBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VGG19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_knazeri_edge_connect(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

