import torch 


def init_deis(mei,n= 5): # copy MEI 5 times along dim 0 and add additive gaussia white noise to it
    detached_stimulus = mei.detach().clone()
    deis = detached_stimulus.repeat(n, 1, 1, 1, 1)
    noise = torch.randn_like(deis) * 0.01
    deis = deis + noise
    print(deis.size())
    deis.requires_grad_(True)
    return deis