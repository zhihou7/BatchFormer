from torch.optim import lr_scheduler


def get_scheduler(name, optimizer, lr, total_steps, final_div_factor=1e4):
    name = name.lower()
    if name == "onecycle":
        scheduler = lr_scheduler.OneCycleLR(
            optimizer, lr, total_steps=total_steps, pct_start=0.1, final_div_factor=final_div_factor
        )
    elif name in ["cos", "cosine"]:
        eta_min = lr / final_div_factor
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
    elif name == "step":
        decay_pct = 0.6
        decay_step = int(total_steps * decay_pct)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=0.1)
    elif name in ["const", "swa"]:
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    else:
        raise ValueError(name)

    return scheduler
