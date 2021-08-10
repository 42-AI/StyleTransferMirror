import torch

eps = 1e-5

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
    _var = X.var(dim=2) + eps
    _std = _var.sqrt().view(Bs, C, 1, 1)

    return _mean, _std

def normalize(X: torch.Tensor):
    _mean, _std = mean_std(X)
    return (X - _mean) / _std