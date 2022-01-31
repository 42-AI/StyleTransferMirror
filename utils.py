import torch
import torch.utils.data as data
import torchvision

import os

from PIL import Image

import gc

def mean_std(X : torch.Tensor):
    size = X.size()

    # except that the feature has 4 dimensions
    assert (len(size) == 4)

    # Bs : batch_size
    # C : channels
    Bs, C = size[:2]

    # Group along channel and batch to find the std and mean
    X = X.view(Bs, C, -1)

    _mean = X.mean(dim=2).view(Bs, C, 1, 1)
    _var = X.var(dim=2) + 1e-5
    _std = _var.sqrt().view(Bs, C, 1, 1)

    return _mean, _std

def normalize(X: torch.Tensor):
    _mean, _std = mean_std(X)
    return (X - _mean) / _std

# clear the gpu memory
def clean():
    gc.collect()
    torch.cuda.empty_cache()

# Load batch from a torch dataload indefinitly
def infinite_loader(dataloader):
    while True:
         for batch in dataloader:
            yield batch
            
class FlatFolderDataset(data.Dataset):
    def __init__(self, _dir : str, size = 512):
        '''
        _dir : path to the folder containing the images
        size : size for inference
        '''

        super().__init__()

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size),
            torchvision.transforms.RandomCrop(size/2),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(.15,.2,.2),
            torchvision.transforms.ToTensor(),
        ])

        # This transformation is used for inference.
        # Because the network is fully convolutional, it can take almost any input size
        # Furthermore we don't want any augmentation transformation on the inference
        self.transform_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size),
            torchvision.transforms.CenterCrop(size),
            torchvision.transforms.RandomHorizontalFlip(1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float16)
        ])
    
        self.dir = _dir

        # list all the image pathes
        self.paths = os.listdir(self.dir)

    def __getitem__(self, index, train : bool = True):
        path = self.paths[index]
        # Try because sometime there is error on image reading 
        # but we don't want the training pipeline to break
        try:
            img = Image.open(os.path.join(self.dir, path)).convert("RGB")
        except:
            return self[index + 1]
        if train:
            return self.transform(img)
        else:
            return self.transform_test(img)

    def __len__(self):
        return len(self.paths)
