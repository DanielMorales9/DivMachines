import torch


def gpu(tensor, gpu=False):
    if gpu:
        return tensor.cuda()
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
