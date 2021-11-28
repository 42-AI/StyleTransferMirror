from IPython.display import clear_output
from PIL import Image
from matplotlib import cm

import torchvision
import numpy as np
import torch
from torch import nn
from torchvision import transforms

import model
import utils
from utils import FlatFolderDataset

import matplotlib.pyplot as plt

from tqdm import tqdm

import json
import cv2
import os

import torch.distributed as dist

# This is a very useful class that load all the useful config from a json file
class DevEnvironment():
    def __init__(self, config_file):
        '''
        config_file: path to the config file
        '''
        config = json.loads(open(config_file).read())

        # This is the device used for training
        self.device = torch.device(config["device"]) if "device" in config else torch.device("cuda")

        # Setting up the layers we need on the vgg pretrained encoder
        self.vgg_encoder = model.vgg
        self.vgg_encoder.load_state_dict(torch.load("./model_save/vgg_normalised.pth"))
        self.vgg_encoder = nn.Sequential(*list(self.vgg_encoder.children())[:44])

        # Setting up the image decoder
        self.decoder = model.decoder.to(self.device)

        # And then the Style Attention Network
        self.network : model.MultiLevelStyleAttention = model.MultiLevelStyleAttention(self.vgg_encoder, self.decoder)
        self.network.to(self.device)
        self.network.eval()

    def load_save(self, file_path : str):
        # Loading the dict of all parameters dict from the file located at file_path
        saved = torch.load(file_path, map_location=lambda storage, loc: storage)

        # Loading the different part of the model separatly
        self.network.decoder.load_state_dict(saved["decoder"], strict=False)
        self.network.sa_module.load_state_dict(saved["sa_module"], strict=False)

# Loading the configs in env variable
env = DevEnvironment("config.json")

# Load the preceding models if not the first training step
start_iteration = 177000
if start_iteration != 0:
    env.load_save(f"model_save/{str(start_iteration).zfill(6)}.pt")

# Use custom style folder to display sample data
custom_style_dataset = FlatFolderDataset("custom_style/", size=256)

print("Ready for data, please connect the client")
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
dist.init_process_group('gloo', rank=1, world_size=2)

style_len = len(custom_style_dataset)
style_index = 0

with torch.cuda.amp.autocast() and torch.no_grad():
    style = torch.cat([torch.unsqueeze(custom_style_dataset.__getitem__(1, False), 0) for _ in range(6)]).to(env.device)
    _tensor = torch.zeros((6, 3, 256, 256))
    while True:
        dist.recv(_tensor, 0)
        _out = env.network(_tensor.to(env.device), style, train=False)
        dist.send(_out.cpu(), 0)