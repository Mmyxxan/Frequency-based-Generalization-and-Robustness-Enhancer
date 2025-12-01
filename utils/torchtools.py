# copied from https://github.com/KaiyangZhou/Dassl.pytorch

import pickle
from functools import partial
import torch
import os.path as osp
import numpy as np
import random
import PIL

import logging
logger = logging.getLogger(__name__)

def count_num_param(model=None, params=None, trainable_only=False):
    r"""Count number of parameters in a model.

    Args:
        model (nn.Module): network model.
        params: network model`s params.
    Examples::
        >>> model_size = count_num_param(model)
    """

    if model is not None:
        return sum(
            p.numel() for p in model.parameters()
            if not trainable_only or p.requires_grad
        )

    if params is not None:
        s = 0
        for p in params:
            if isinstance(p, dict):
                param = p["params"]
            else:
                param = p
            if not trainable_only or param.requires_grad:
                s += param.numel()
        return s

    # logger
    raise ValueError("model and params must provide at least one.")

def load_checkpoint(fpath):
    r"""Load checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError("File path is None")

    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))

    map_location = None if torch.cuda.is_available() else "cpu"

    try:
        checkpoint = torch.load(fpath, map_location=map_location)

    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )

    except Exception:
        logger.info('Unable to load checkpoint from "{}"'.format(fpath))
        raise

    return checkpoint

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collect_env_info():
    """Return env info as a string.

    Code source: github.com/facebookresearch/maskrcnn-benchmark
    """
    from torch.utils.collect_env import get_pretty_env_info

    env_str = get_pretty_env_info()
    env_str += "\n        Pillow ({})".format(PIL.__version__)
    return env_str
