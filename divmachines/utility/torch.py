import torch


def gpu(tensor, gpu=False, device=None):
    if gpu:
        return tensor.cuda(device_id=device)
    else:
        return tensor


def cpu(tensor, cpu=False):
    if cpu:
        return tensor.cpu()
    else:
        return tensor


def assert_no_grad(variable):
    if variable.requires_grad:
        raise ValueError(
            "nn criterions don't compute the gradient w.r.t. targets - please "
            "mark these variables as volatile or not requiring gradients")


def set_seed(seed, cuda=False):
    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)
