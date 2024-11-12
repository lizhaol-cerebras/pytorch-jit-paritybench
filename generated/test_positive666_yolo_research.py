
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchvision, types, typing, uuid, warnings
import operator as op
from dataclasses import dataclass
import numpy as np
from torch import Tensor
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import torch


import torch.backends.cudnn as cudnn


import torch.nn.functional as F


import time


from copy import deepcopy


import torch.distributed as dist


import torch.hub as hub


import torch.optim.lr_scheduler as lr_scheduler


import torchvision


from torch.cuda import amp


import re


import warnings


import pandas as pd


from torch.utils.mobile_optimizer import optimize_for_mobile


import random


import numpy as np


import math


from collections import OrderedDict


from collections import namedtuple


from copy import copy


import torch.nn as nn


import logging


import tensorflow as tf


from tensorflow import keras


from torch.nn.init import constant_


from torch.nn.init import xavier_uniform_


from torch.nn.init import uniform_


import copy


from itertools import repeat


from torch.utils.data import Dataset


from numpy import random


import matplotlib


import matplotlib.pyplot as plt


import torch.optim as optim


import torch.utils.data


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.tensorboard import SummaryWriter


from torch.optim import Adam


from torch.optim import SGD


from torch.optim import lr_scheduler


from torch.optim import AdamW


import torchvision.transforms as T


import torchvision.transforms.functional as TF


from torch.utils.data import DataLoader


from torch.utils.data import dataloader


from torch.utils.data import distributed


import inspect


import logging.config


from typing import Optional


from typing import Union


from types import SimpleNamespace


from torch.nn import functional as F


from collections import defaultdict


from functools import lru_cache


import torchvision.transforms.functional as F


from torch import nn


from torch import optim


import uuid


from matplotlib import font_manager


class MP(nn.Module):

    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class SP(nn.Module):

    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [(d * (x - 1) + 1) for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [(x // 2) for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DecoupledHead(nn.Module):

    def __init__(self, ch=256, nc=80, anchors=()):
        super().__init__()
        self.nc = nc
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.merge = Conv(ch, 256, 1, 1)
        self.cls_convs1 = Conv(256, 256, 3, 1, 1)
        self.cls_convs2 = Conv(256, 256, 3, 1, 1)
        self.reg_convs1 = Conv(256, 256, 3, 1, 1)
        self.reg_convs2 = Conv(256, 256, 3, 1, 1)
        self.cls_preds = nn.Conv2d(256, self.nc * self.na, 1)
        self.reg_preds = nn.Conv2d(256, 4 * self.na, 1)
        self.obj_preds = nn.Conv2d(256, 1 * self.na, 1)

    def forward(self, x):
        x = self.merge(x)
        x1 = self.cls_convs1(x)
        x1 = self.cls_convs2(x1)
        x1 = self.cls_preds(x1)
        x2 = self.reg_convs1(x)
        x2 = self.reg_convs2(x2)
        x21 = self.reg_preds(x2)
        x22 = self.obj_preds(x2)
        out = torch.cat([x21, x22, x1], 1)
        return out


class Merge(nn.Module):

    def __init__(self, ch=()):
        super(Merge, self).__init__()

    def forward(self, x):
        return [x[0], x[1], x[2]]


class Refine(nn.Module):

    def __init__(self, c2, k, s, ch):
        super(Refine, self).__init__()
        self.refine = nn.ModuleList()
        for c in ch:
            self.refine.append(Conv(c, c2, k, s))

    def forward(self, x):
        for i, f in enumerate(x):
            if i == 0:
                r = self.refine[i](f)
            else:
                r_p = self.refine[i](f)
                r_p = F.interpolate(r_p, r.size()[2:], mode='bilinear', align_corners=False)
                r = r + r_p
        return r


class Proto(nn.Module):

    def __init__(self, c1, c_=256, c2=32):
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class DWConv(Conv):

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class LightConv(nn.Module):
    """Light convolution with args(ch_in, ch_out, kernel).
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class Ensemble(nn.ModuleList):

    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 2)
        return y, None


class ConvFocus(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(ConvFocus, self).__init__()
        slice_kernel = 3
        slice_stride = 2
        self.conv_slice = Conv(c1, c1 * 4, slice_kernel, slice_stride, p, g, act)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        if hasattr(self, 'conv_slice'):
            x = self.conv_slice(x)
        else:
            x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x


LOGGING_NAME = 'yolo_research'


LOGGER = logging.getLogger(LOGGING_NAME)


def attempt_download(file, repo='ultralytics/yolov5'):
    file = Path(str(file).strip().replace("'", '').lower())
    if not file.exists():
        try:
            response = requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json()
            assets = [x['name'] for x in response['assets']]
            tag = response['tag_name']
        except:
            assets = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']
            tag = subprocess.check_output('git tag', shell=True).decode().split()[-1]
        name = file.name
        if name in assets:
            msg = f'{file} missing, try downloading from https://github.com/{repo}/releases/'
            redundant = False
            try:
                url = f'https://github.com/{repo}/releases/download/{tag}/{name}'
                None
                torch.hub.download_url_to_file(url, file)
                assert file.exists() and file.stat().st_size > 1000000.0
            except Exception as e:
                None
                assert redundant, 'No secondary mirror'
                url = f'https://storage.googleapis.com/{repo}/ckpt/{name}'
                None
                os.system(f'curl -L {url} -o {file}')
            finally:
                if not file.exists() or file.stat().st_size < 1000000.0:
                    file.unlink(missing_ok=True)
                    None
                None
                return


GITHUB_ASSET_NAMES = [f'yolov8{k}{suffix}.pt' for k in 'nsmlx' for suffix in ('', '6', '-cls', '-seg', '-pose')] + [f'yolov5{k}u.pt' for k in 'nsmlx'] + [f'yolov3{k}u.pt' for k in ('', '-spp', '-tiny')] + [f'rtdetr-{k}.pt' for k in 'lx']


def is_dir_writeable(dir_path: 'Union[str, Path]') ->bool:
    """
    Check if a directory is writeable.

    Args:
        dir_path (str) or (Path): The path to the directory.

    Returns:
        bool: True if the directory is writeable, False otherwise.
    """
    try:
        with tempfile.TemporaryFile(dir=dir_path):
            pass
        return True
    except OSError:
        return False


def get_user_config_dir(sub_dir='Ultralytics'):
    """
    Get the user config directory.

    Args:
        sub_dir (str): The name of the subdirectory to create.

    Returns:
        Path: The path to the user config directory.
    """
    if WINDOWS:
        path = Path.home() / 'AppData' / 'Roaming' / sub_dir
    elif MACOS:
        path = Path.home() / 'Library' / 'Application Support' / sub_dir
    elif LINUX:
        path = Path.home() / '.config' / sub_dir
    else:
        raise ValueError(f'Unsupported operating system: {platform.system()}')
    if not is_dir_writeable(str(path.parent)):
        path = Path('/tmp') / sub_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def emojis(string=''):
    return string.encode().decode('ascii', 'ignore') if WINDOWS else string


def check_version(current: 'str'='0.0.0', minimum: 'str'='0.0.0', name: 'str'='version ', pinned: 'bool'=False, hard: 'bool'=False, verbose: 'bool'=False) ->bool:
    """
    Check current version against the required minimum version.

    Args:
        current (str): Current version.
        minimum (str): Required minimum version.
        name (str): Name to be used in warning message.
        pinned (bool): If True, versions must match exactly. If False, minimum version must be satisfied.
        hard (bool): If True, raise an AssertionError if the minimum version is not met.
        verbose (bool): If True, print warning message if minimum version is not met.

    Returns:
        bool: True if minimum version is met, False otherwise.
    """
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = current == minimum if pinned else current >= minimum
    warning_message = f'WARNING ⚠️ {name}{minimum} is required by YOLOv8, but {name}{current} is currently installed'
    if hard:
        assert result, emojis(warning_message)
    if verbose and not result:
        LOGGER.warning(warning_message)
    return result


def get_git_dir():
    """
    Determines whether the current file is part of a git repository and if so, returns the repository root directory.
    If the current file is not part of a git repository, returns None.

    Returns:
        (Path) or (None): Git root directory if found or None if not found.
    """
    for d in Path(__file__).parents:
        if (d / '.git').is_dir():
            return d
    return None


def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        dict: YAML data and file name.
    """
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()
        if not s.isprintable():
            s = re.sub('[^\\x09\\x0A\\x0D\\x20-\\x7E\\x85\\xA0-\\uD7FF\\uE000-\\uFFFD\\U00010000-\\U0010ffff]+', '', s)
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)


def yaml_save(file='data.yaml', data=None):
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict, optional): Data to save in YAML format. Default is None.

    Returns:
        None: Data is saved to the specified file.
    """
    file = Path(file)
    if not file.parent.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
    for k, v in data.items():
        if isinstance(v, Path):
            dict[k] = str(v)
    with open(file, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)


TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'


def check_disk_space(url='https://ultralytics.com/assets/coco128.zip', sf=1.5, hard=True):
    """
    Check if there is sufficient disk space to download and store a file.

    Args:
        url (str, optional): The URL to the file. Defaults to 'https://ultralytics.com/assets/coco128.zip'.
        sf (float, optional): Safety factor, the multiplier for the required free space. Defaults to 2.0.
        hard (bool, optional): Whether to throw an error or not on insufficient disk space. Defaults to True.

    Returns:
        (bool): True if there is sufficient disk space, False otherwise.
    """
    with contextlib.suppress(Exception):
        gib = 1 << 30
        data = int(requests.head(url).headers['Content-Length']) / gib
        total, used, free = (x / gib for x in shutil.disk_usage('/'))
        if data * sf < free:
            return True
        text = f'WARNING ⚠️ Insufficient free disk space {free:.1f} GB < {data * sf:.3f} GB required, Please free {data * sf - free:.1f} GB additional disk space and try again.'
        if hard:
            raise MemoryError(text)
        else:
            LOGGER.warning(text)
            return False
    return True


def clean_url(url):
    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    url = str(Path(url)).replace(':/', '://')
    return urllib.parse.unquote(url).split('?')[0]


def is_online() ->bool:
    """
    Check internet connectivity by attempting to connect to a known online host.

    Returns:
        bool: True if connection is successful, False otherwise.
    """
    with contextlib.suppress(Exception):
        host = socket.gethostbyname('www.github.com')
        socket.create_connection((host, 80), timeout=2)
        return True
    return False


def unzip_file(file, path=None, exclude=('.DS_Store', '__MACOSX')):
    if path is None:
        path = Path(file).parent
    with ZipFile(file) as zipObj:
        for f in zipObj.namelist():
            if all(x not in f for x in exclude):
                zipObj.extract(f, path=path)


def url2file(url):
    """Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt."""
    return Path(clean_url(url)).name


def safe_download(url, file=None, dir=None, unzip=True, delete=False, curl=False, retry=3, min_bytes=1.0, progress=True):
    """
    Downloads files from a URL, with options for retrying, unzipping, and deleting the downloaded file.

    Args:
        url (str): The URL of the file to be downloaded.
        file (str, optional): The filename of the downloaded file.
            If not provided, the file will be saved with the same name as the URL.
        dir (str, optional): The directory to save the downloaded file.
            If not provided, the file will be saved in the current working directory.
        unzip (bool, optional): Whether to unzip the downloaded file. Default: True.
        delete (bool, optional): Whether to delete the downloaded file after unzipping. Default: False.
        curl (bool, optional): Whether to use curl command line tool for downloading. Default: False.
        retry (int, optional): The number of times to retry the download in case of failure. Default: 3.
        min_bytes (float, optional): The minimum number of bytes that the downloaded file should have, to be considered
            a successful download. Default: 1E0.
        progress (bool, optional): Whether to display a progress bar during the download. Default: True.
    """
    if '://' not in str(url) and Path(url).is_file():
        f = Path(url)
    else:
        assert dir or file, 'dir or file required for download'
        f = dir / url2file(url) if dir else Path(file)
        desc = f'Downloading {clean_url(url)} to {f}'
        LOGGER.info(f'{desc}...')
        f.parent.mkdir(parents=True, exist_ok=True)
        check_disk_space(url)
        for i in range(retry + 1):
            try:
                if curl or i > 0:
                    s = 'sS' * (not progress)
                    r = subprocess.run(['curl', '-#', f'-{s}L', url, '-o', f, '--retry', '3', '-C', '-']).returncode
                    assert r == 0, f'Curl return value {r}'
                else:
                    method = 'torch'
                    if method == 'torch':
                        torch.hub.download_url_to_file(url, f, progress=progress)
                    else:
                        with request.urlopen(url) as response, tqdm(total=int(response.getheader('Content-Length', 0)), desc=desc, disable=not progress, unit='B', unit_scale=True, unit_divisor=1024, bar_format=TQDM_BAR_FORMAT) as pbar:
                            with open(f, 'wb') as f_opened:
                                for data in response:
                                    f_opened.write(data)
                                    pbar.update(len(data))
                if f.exists():
                    if f.stat().st_size > min_bytes:
                        break
                    f.unlink()
            except Exception as e:
                if i == 0 and not is_online():
                    raise ConnectionError(emojis(f'❌  Download failure for {url}. Environment is not online.')) from e
                elif i >= retry:
                    raise ConnectionError(emojis(f'❌  Download failure for {url}. Retry limit reached.')) from e
                LOGGER.warning(f'⚠️ Download failure, retrying {i + 1}/{retry} {url}...')
    if unzip and f.exists() and f.suffix in ('', '.zip', '.tar', '.gz'):
        unzip_dir = dir or f.parent
        LOGGER.info(f'Unzipping {f} to {unzip_dir}...')
        if is_zipfile(f):
            unzip_dir = unzip_file(file=f, path=unzip_dir)
        elif f.suffix == '.tar':
            subprocess.run(['tar', 'xf', f, '--directory', unzip_dir], check=True)
        elif f.suffix == '.gz':
            subprocess.run(['tar', 'xfz', f, '--directory', unzip_dir], check=True)
        if delete:
            f.unlink()
        return unzip_dir


def attempt_download_asset(file, repo='ultralytics/assets', release='v0.0.0'):

    def github_assets(repository, version='latest'):
        if version != 'latest':
            version = f'tags/{version}'
        response = requests.get(f'https://api.github.com/repos/{repository}/releases/{version}').json()
        return response['tag_name'], [x['name'] for x in response['assets']]
    file = str(file)
    file = checks.check_yolov5u_filename(file)
    file = Path(file.strip().replace("'", ''))
    if file.exists():
        return str(file)
    elif (SETTINGS['weights_dir'] / file).exists():
        return str(SETTINGS['weights_dir'] / file)
    else:
        name = Path(parse.unquote(str(file))).name
        if str(file).startswith(('http:/', 'https:/')):
            url = str(file).replace(':/', '://')
            file = name.split('?')[0]
            if Path(file).is_file():
                LOGGER.info(f'Found {url} locally at {file}')
            else:
                safe_download(url=url, file=file, min_bytes=100000.0)
            return file
        assets = GITHUB_ASSET_NAMES
        try:
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                tag, assets = github_assets(repo)
            except Exception:
                try:
                    tag = subprocess.check_output(['git', 'tag']).decode().split()[-1]
                except Exception:
                    tag = release
        file.parent.mkdir(parents=True, exist_ok=True)
        if name in assets:
            safe_download(url=f'https://github.com/{repo}/releases/download/{tag}/{name}', file=file, min_bytes=100000.0)
        return str(file)


class ASFFV5(nn.Module):

    def __init__(self, level, multiplier=1, rfb=False, vis=False, act_cfg=True):
        """
        ASFF version for YoloV5 .
        different than YoloV3
        multiplier should be 1, 0.5
        which means, the channel of ASFF can be 
        512, 256, 128 -> multiplier=1
        256, 128, 64 -> multiplier=0.5
        For even smaller, you need change code manually.
        """
        super(ASFFV5, self).__init__()
        self.level = level
        self.dim = [int(1024 * multiplier), int(512 * multiplier), int(256 * multiplier)]
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Conv(int(512 * multiplier), self.inter_dim, 3, 2)
            self.stride_level_2 = Conv(int(256 * multiplier), self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, int(1024 * multiplier), 3, 1)
        elif level == 1:
            self.compress_level_0 = Conv(int(1024 * multiplier), self.inter_dim, 1, 1)
            self.stride_level_2 = Conv(int(256 * multiplier), self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, int(512 * multiplier), 3, 1)
        elif level == 2:
            self.compress_level_0 = Conv(int(1024 * multiplier), self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(int(512 * multiplier), self.inter_dim, 1, 1)
            self.expand = Conv(self.inter_dim, int(256 * multiplier), 3, 1)
        compress_c = 8 if rfb else 16
        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_levels = Conv(compress_c * 3, 3, 1, 1)
        self.vis = vis

    def forward(self, x):
        """
        # 128, 256, 512
        512, 256, 128
        from small -> large
        """
        x_level_0 = x[2]
        x_level_1 = x[1]
        x_level_2 = x[0]
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + level_1_resized * levels_weight[:, 1:2, :, :] + level_2_resized * levels_weight[:, 2:, :, :]
        out = self.expand(fused_out_reduced)
        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class Decouple(nn.Module):

    def __init__(self, c1, nc=80, na=3):
        super().__init__()
        c_ = min(c1, 256)
        self.na = na
        self.nc = nc
        self.a = Conv(c1, c_, 1)
        c = [int(x + na * 5) for x in (c_ - na * 5) * torch.linspace(1, 0, 4)]
        self.b1, self.b2, self.b3 = Conv(c_, c[1], 3), Conv(c[1], c[2], 3), nn.Conv2d(c[2], na * 5, 1)
        self.c1, self.c2, self.c3 = Conv(c_, c_, 1), Conv(c_, c_, 1), nn.Conv2d(c_, na * nc, 1)

    def forward(self, x):
        bs, nc, ny, nx = x.shape
        x = self.a(x)
        b = self.b3(self.b2(self.b1(x)))
        c = self.c3(self.c2(self.c1(x)))
        return torch.cat((b.view(bs, self.na, 5, ny, nx), c.view(bs, self.na, self.nc, ny, nx)), 2).view(bs, -1, ny, nx)

