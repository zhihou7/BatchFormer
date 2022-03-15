import torch


def get_optimizer(name, params, **kwargs):
    name = name.lower()
    optimizers = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}
    optim_cls = optimizers[name]

    return optim_cls(params, **kwargs)
