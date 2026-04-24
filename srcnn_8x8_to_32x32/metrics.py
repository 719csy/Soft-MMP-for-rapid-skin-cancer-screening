from __future__ import annotations

import torch
import torch.nn.functional as F


EPS = 1e-8


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(prediction, target)
    value = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=EPS))
    return float(value.item())


def ssim(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    padding = window_size // 2
    mu_x = F.avg_pool2d(prediction, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.avg_pool2d(prediction * prediction, window_size, stride=1, padding=padding) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=padding) - mu_y_sq
    sigma_xy = F.avg_pool2d(prediction * target, window_size, stride=1, padding=padding) - mu_xy
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return float((numerator / torch.clamp(denominator, min=EPS)).mean().item())


def mae(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(prediction - target)).item())


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((prediction - target) ** 2)).item())


def metric_dict(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "psnr": psnr(prediction, target),
        "ssim": ssim(prediction, target),
        "mae": mae(prediction, target),
        "rmse": rmse(prediction, target),
    }

