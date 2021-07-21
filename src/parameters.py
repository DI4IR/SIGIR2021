import torch

DEVICE = torch.device("cuda:0")

DEFAULT_DATA_DIR = './data_download/'

SAVED_CHECKPOINTS = [16000, 32*1000, 64000, 100*1000, 150*1000]
