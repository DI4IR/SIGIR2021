import os
import torch
import datetime

from collections import OrderedDict

def print_message(*s):
    s = ' '.join([str(x) for x in s])
    print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)


def save_checkpoint(path, epoch_idx, mb_idx, model, optimizer):
    print("#> Saving a checkpoint..")

    checkpoint = {}
    checkpoint['epoch'] = epoch_idx
    checkpoint['batch'] = mb_idx
    checkpoint['model_state_dict'] = model.state_dict()
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None):
    print_message("#> Loading checkpoint", path)

    checkpoint = torch.load(path, map_location='cpu')
    #Missing key(s) in state_dict: "impact_score_encoder.0.weight", "impact_score_encoder.0.bias", "impact_score_encoder.3.weight", "impact_score_encoder.3.bias".
    #Unexpected key(s) in state_dict: "pre_classifier.weight", "pre_classifier.bias", "classifier.weight", "classifier.bias".
    #print(checkpoint['model_state_dict'])
    #input()

    #state_dict_trans = {
    #    "impact_score_encoder.0.weight": "pre_classifier.weight",
    #    "impact_score_encoder.0.bias": "pre_classifier.bias",
    #    "impact_score_encoder.3.weight": "classifier.weight",
    #    "impact_score_encoder.3.bias": "classifier.bias",
    #}
    state_dict_trans = {
        "pre_classifier.weight": "impact_score_encoder.0.weight",
        "pre_classifier.bias": "impact_score_encoder.0.bias",
        "classifier.weight": "impact_score_encoder.3.weight",
        "classifier.bias": "impact_score_encoder.3.bias",
    }
    state_dict = rename_state_dict_keys(checkpoint['model_state_dict'],state_dict_trans)
    
    #model.load_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(state_dict)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print_message("#> checkpoint['epoch'] =", checkpoint['epoch'])
    print_message("#> checkpoint['batch'] =", checkpoint['batch'])

    return checkpoint

def rename_state_dict_keys(state_dict, key_transformation):
    """
    state_dict: original state dict ("para_name": para_value)...
    key_transformation: dict {"orig_para_name": "new_para_name"}
    """
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key not in key_transformation:
            new_state_dict[key] = value
        else:
            new_key = key_transformation[key]
            new_state_dict[new_key] = value
    return new_state_dict


def create_directory(path):
    if not os.path.exists(path):
        print_message("#> Creating", path)
        os.makedirs(path)


def batch(group, bsize):
    offset = 0
    while offset < len(group):
        L = group[offset: offset + bsize]
        yield L
        offset += len(L)
    return
