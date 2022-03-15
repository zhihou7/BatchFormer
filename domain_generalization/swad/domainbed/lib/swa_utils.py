# Burrowed from https://github.com/pytorch/pytorch/blob/master/torch/optim/swa_utils.py
# modified for the DomainBed.
import copy
import torch
from torch.nn import Module
from copy import deepcopy


class AveragedModel(Module):
    def __init__(self, model, device=None, avg_fn=None, rm_optimizer=False):
        super(AveragedModel, self).__init__()
        self.start_step = -1
        self.end_step = -1
        if isinstance(model, AveragedModel):
            # prevent nested averagedmodel
            model = model.module
        self.module = deepcopy(model)
        if rm_optimizer:
            for k, v in vars(self.module).items():
                if isinstance(v, torch.optim.Optimizer):
                    setattr(self.module, k, None)

        if device is not None:
            self.module = self.module.to(device)

        self.register_buffer("n_averaged", torch.tensor(0, dtype=torch.long, device=device))

        if avg_fn is None:
            def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return averaged_model_parameter + (model_parameter - averaged_model_parameter) / (
                    num_averaged + 1
                )

        self.avg_fn = avg_fn

    def forward(self, *args, **kwargs):
        #  return self.predict(*args, **kwargs)
        return self.module(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @property
    def network(self):
        return self.module.network

    def update_parameters(self, model, step=None, start_step=None, end_step=None):
        """Update averaged model parameters

        Args:
            model: current model to update params
            step: current step. step is saved for log the averaged range
            start_step: set start_step only for first update
            end_step: set end_step
        """
        if isinstance(model, AveragedModel):
            model = model.module
        for p_swa, p_model in zip(self.parameters(), model.parameters()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(
                    self.avg_fn(p_swa.detach(), p_model_, self.n_averaged.to(device))
                )
        self.n_averaged += 1

        if step is not None:
            if start_step is None:
                start_step = step
            if end_step is None:
                end_step = step

        if start_step is not None:
            if self.n_averaged == 1:
                self.start_step = start_step

        if end_step is not None:
            self.end_step = end_step

    def clone(self):
        clone = copy.deepcopy(self.module)
        clone.optimizer = clone.new_optimizer(clone.network.parameters())
        return clone


def cvt_dbiterator_to_loader(dbiterator, n_iter):
    """Convert DB iterator to the loader"""
    for _ in range(n_iter):
        minibatches = [(x, y) for x, y in next(dbiterator)]
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        yield all_x, all_y


@torch.no_grad()
def update_bn(iterator, model, n_steps, device="cuda"):
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for i in range(n_steps):
        # batches_dictlist: [{env0_data_key: tensor, env0_...}, env1_..., ...]
        batches_dictlist = next(iterator)
        x = torch.cat([dic["x"] for dic in batches_dictlist])
        x = x.to(device)

        model(x)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)
