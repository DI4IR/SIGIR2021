import os
import datetime
import torch

from torch.optim import Optimizer
from transformers import BertModel


def unique(seq):
    """
    Source: https://stackoverflow.com/a/480227/1493011
    """
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def print_message(*s: list) -> None:
    """
    Print an output message with time information.

    :param *s: list of printable items

    Used for reporting status information to the user.
    """
    s = ' '.join([str(x) for x in s])
    print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)


def save_checkpoint(path: str, epoch_idx: int, mb_idx,
                    model: BertModel, optimizer: Optimizer) -> None:
    """
    Save a current checkpoint of a training procedure.

    :param path: pathname to the output file
    :param epoch_idx: index of the last training epoch for the saved model
    :param mb_idx: index of the last batch for the saved model
    :param model: the Torch model to save
    :param optmizer: the Torch optimizer to save

    Used for periodically save the model under training.
    """

    print_message("#> Saving a checkpoint in", path)

    checkpoint = {}
    checkpoint['epoch'] = epoch_idx
    checkpoint['batch'] = mb_idx
    checkpoint['model_state_dict'] = model.state_dict()
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(path: str, model: BertModel, optimizer: Optimizer=None):

    print_message("#> Loading a checkpoint in", path)

    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print_message("#> checkpoint['epoch'] =", checkpoint['epoch'])
    print_message("#> checkpoint['batch'] =", checkpoint['batch'])

    return checkpoint


def create_directory(path):

    if not os.path.exists(path):
        print_message("#> Creating directory", path)
        os.makedirs(path)
