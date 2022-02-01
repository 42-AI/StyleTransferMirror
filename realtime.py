from PIL import Image

import numpy as np
import torch
from torch import nn

import model
import utils
from utils import FlatFolderDataset
import torch



import json
import cv2

# This is a very useful class that load all the useful config from a json file
class DevEnvironment():
    def __init__(self, config_file):
        '''
        config_file: path to the config file
        '''
        config = json.loads(open(config_file).read())

        self.batch_size = config["batch_size"] if "batch_size" in config else 32

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
        self.network.train()

        # Setting up everything related to the learning rate
        self.lr = config["lr"] if "lr" in config else 1e-4
        self.lr_decay = config["lr_decay"] if "lr_decay" in config else 0.99999
        self.decay_after = config["decay_after"] if "decay_after" in config else 5000

        # Total number of training steps to train
        self.iters = config["iters"] if "iters" in config else 200000

        # Setting up the weights for the loss calculation
        self.style_weight = config["style_weight"] if "style_weight" in config else 5.0
        self.content_weight = config["content_weight"] if "content_weight" in config else 1.0
        self.identity1_weight = config["identity1_weight"] if "identity1_weight" in config else 50.0
        self.identity2_weight = config["identity2_weight"] if "identity2_weight" in config else 1.0

        # And then some variables about log and saving intervals
        self.save_checkpoint_interval = config["save_checkpoint_interval"] if "save_checkpoint_interval" in config else 1000
        self.log_generated_interval = config["log_generated_interval"] if "log_generated_interval" in config else 20
        self.img_generated_interval = config["img_generated_interval"] if "img_generated_interval" in config else 100
    
    def load_save(self, file_path : str):
        # Loading the dict of all parameters dict from the file located at file_path
        saved = torch.load(file_path, map_location=lambda storage, loc: storage)

        # Loading the different part of the model separatly
        self.network.decoder.load_state_dict(saved["decoder"], strict=False)
        self.network.sa_module.load_state_dict(saved["sa_module"], strict=False)

        self.network = self.network.half()

# Loading the configs in env variable
env = DevEnvironment("config.json")

# Load the preceding models if not the first training step
start_iteration = 177000
if start_iteration != 0:
    env.load_save(f"model_save/{str(start_iteration).zfill(6)}.pt")


CONTENT_SIZE = 512
# Use custom style folder to display sample data
custom_style_dataset = FlatFolderDataset("custom_style/", 384)
cam_transform = FlatFolderDataset("custom_style/", CONTENT_SIZE)

def camera_feed():
    cap = cv2.VideoCapture(0)

    if not (cap.isOpened()):
        print("Could not open video device")

    cap.set(3, 1280)
    cap.set(4, 720)

    for param in env.network.parameters():
        param.grad = None

    torch.backends.cudnn.benchmark = True

    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    change_each = 30
    until_change = change_each

    with torch.cuda.amp.autocast() and torch.no_grad():
        style_len = len(custom_style_dataset)
        style_index = 0
        style = torch.unsqueeze(custom_style_dataset.__getitem__(style_index, False), 0).to(env.device).half()
        style_np = cv2.resize(cv2.cvtColor(style[0].permute(1, 2, 0).float().cpu().numpy(), cv2.COLOR_BGR2RGB), (CONTENT_SIZE, CONTENT_SIZE))
        
        while(True):
            utils.clean()

            # Capture frame-by-frame
            ret, frame = cap.read()
            
            frame = cam_transform.transform_test(Image.fromarray(frame))
            out = env.network(torch.unsqueeze(frame.to(env.device), 0), style, train=False)

            # Display the resulting frame
            cv2.imshow('window', np.hstack((frame.permute(1, 2, 0).float().numpy(), out[0, [2, 1, 0]].permute(1, 2, 0).float().cpu().numpy(), style_np)))

            # Loop through styles
            until_change -= 1

            k = cv2.waitKey(33)
            #Waits for a user input to quit the application
            if k == ord('q'):
                break
            elif k == ord('a') or not until_change:
                if (style_index - 1 >= 0):
                    style_index -= 1
                else:
                    style_index = style_len - 1
                style = torch.unsqueeze(custom_style_dataset.__getitem__(style_index, False), 0).to(env.device).half()
                style_np = cv2.resize(cv2.cvtColor(style[0].permute(1, 2, 0).float().cpu().numpy(), cv2.COLOR_BGR2RGB), (CONTENT_SIZE, CONTENT_SIZE), fx=2.5, fy=2.5)
                until_change = change_each

            elif k == ord('d'):
                if (style_index + 1 < style_len):
                    style_index += 1
                else:
                    style_index = 0
                style = torch.unsqueeze(custom_style_dataset.__getitem__(style_index, False), 0).to(env.device).half()
                style_np = cv2.resize(cv2.cvtColor(style[0].permute(1, 2, 0).float().cpu().numpy(), cv2.COLOR_BGR2RGB), (CONTENT_SIZE, CONTENT_SIZE), fx=2.5, fy=2.5)

    cap.release()
    cv2.destroyAllWindows()

camera_feed()