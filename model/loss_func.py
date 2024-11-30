import torch
import torch.nn.functional as F


# For feature
def sce_loss(x, y, alpha=2):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow(alpha)
    loss = loss.mean()
    return loss


def time_diff(x):
    """
    Args:
        x: reconfiguration graph (T, V, C) eg. (frame_nums, vertex_nums, channel`x,y,z`)
    """
    frame_id = 0
    diff_value = 0
    while frame_id < x.shape[0] - 1:
        diff_value += torch.sum(x[frame_id + 1] - x[frame_id])
        frame_id += 1
    return diff_value


def frame_sum_sce(x, y, alpha=2):
    frame_loss = []
    # i represent the different frames
    for i in range(x.shape[2]):
        _loss = sce_loss(x[:, :, i], y[:, :, i], alpha)
        frame_loss.append(_loss)
    loss = sum(frame_loss)
    return loss


def batch_sum_mse(x, y):
    frame_loss = []
    for b in range(x.shape[0]):
        # i represent the different frames
        for i in range(x.shape[2]):
            frame_loss.append(F.mse_loss(x[b, :, i], y[b, :, i]))
    loss = sum(frame_loss)
    return loss


def frame_sum_mse(x, y):
    frame_loss = []
    # i represent the different frames
    for i in range(x.shape[2]):
        frame_loss.append(F.mse_loss(x[:, :, i], y[:, :, i]))
    loss = sum(frame_loss)
    return loss


def frame_mean_mse(x, y):
    frame_loss = []
    for i in range(x.shape[2]):
        frame_loss.append(F.mse_loss(x[:, :, i], y[:, :, i]))
    loss = sum(frame_loss) / len(frame_loss)
    return loss

