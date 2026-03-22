import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def _to_btd(tactile: Tensor) -> tuple[Tensor, bool]:
    if tactile.ndim == 2:
        return tactile.unsqueeze(1), True
    if tactile.ndim == 3:
        return tactile, False
    raise ValueError(
        f"Tactile tensor must have shape (batch, dim) or (batch, time, dim). Got {tuple(tactile.shape)}"
    )


def split_tactile_low_high(tactile: Tensor, lowpass_window: int = 5) -> tuple[Tensor, Tensor]:
    """Split tactile signals into low/high-frequency components along the time axis.

    Args:
        tactile: Tensor of shape (B, T, D) or (B, D).
        lowpass_window: Moving-average window size for low-frequency extraction.

    Returns:
        low_freq, high_freq tensors with the same shape as the input tensor.
    """
    if lowpass_window < 1:
        raise ValueError(f"`lowpass_window` must be >= 1. Got {lowpass_window}")

    tactile_btd, was_2d = _to_btd(tactile)
    bsize, tsize, dim = tactile_btd.shape

    if tsize == 1 or lowpass_window == 1:
        low = tactile_btd
        high = torch.zeros_like(tactile_btd)
    else:
        window = min(lowpass_window, tsize)
        pad = (window - 1) // 2

        tactile_bdt = tactile_btd.transpose(1, 2)
        tactile_flat = tactile_bdt.reshape(bsize * dim, 1, tsize)
        tactile_padded = F.pad(tactile_flat, (pad, window - 1 - pad), mode="replicate")
        low_flat = F.avg_pool1d(tactile_padded, kernel_size=window, stride=1)
        low = low_flat.reshape(bsize, dim, tsize).transpose(1, 2)
        high = tactile_btd - low

    if was_2d:
        return low[:, 0, :], high[:, 0, :]
    return low, high