import torch
import torch.nn.functional as F


def mask_lambda(x, y, lambda_index=0):
    mask = 1.0 - y[:, lambda_index, :, :]
    x[:, :3, :, :] *= mask.unsqueeze(1)
    return x


def mask_flow(x, y, flow_index=4):
    mask = y[:, flow_index, :, :]
    x[:, :3, :, :] *= mask.unsqueeze(1)
    return x


def pad_to_even(x):
    h, w = x.shape[-2:]
    pad_h = (h % 2)
    pad_w = (w % 2)
    return F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')


def center_crop(enc_feat, target_feat):
    _, _, h, w = target_feat.shape
    # print("Target shape:", h, w)
    enc_h, enc_w = enc_feat.shape[-2:]
    # print("Enc shape:", enc_h, enc_w)
    assert enc_h >= h
    assert enc_w >= w
    crop_top = (enc_h - h) // 2
    crop_left = (enc_w - w) // 2
    return enc_feat[:, :, crop_top:crop_top + h, crop_left:crop_left + w]


def compute_y_stats(data_points):
    # Stack all y's (first 4 channels) along new batch dim
    y_all = torch.stack([dp[:4] for dp in data_points], dim=0)  # shape: [N, 4, H, W]

    mean = y_all.mean(dim=[0, 2, 3])  # mean per channel (4,)
    std = y_all.std(dim=[0, 2, 3])  # std per channel (4,)

    return mean, std


def normalize_y(sample, mean, std, eps=1e-6):
    return (sample - mean.view(-1, 1, 1)) / (std.view(-1, 1, 1) + eps)


def denormalize_y(sample, mean, std, eps=1e-6):
    return sample * (std.view(-1, 1, 1) + eps) + mean.view(-1, 1, 1)