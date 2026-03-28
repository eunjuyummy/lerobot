from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


@dataclass(frozen=True)
class SparshEncoderLoadResult:
    missing_keys: list[str]
    unexpected_keys: list[str]


def _maybe_add_repo_to_syspath(repo_path: str | os.PathLike[str] | None) -> None:
    if repo_path is None:
        return
    repo_path = str(Path(repo_path).expanduser().resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def _find_default_sparsh_repo_from_checkpoint_dir(checkpoint_dir: str | os.PathLike[str]) -> str | None:
    # Heuristic for common layout: <repo>/checkpoints/<ckpt-name>
    ckpt_dir = Path(checkpoint_dir).expanduser().resolve()
    if ckpt_dir.parent.name == "checkpoints":
        return str(ckpt_dir.parent.parent)
    return None


def _load_state_dict_from_checkpoint_dir(checkpoint_dir: str | os.PathLike[str]) -> dict[str, Tensor]:
    ckpt_dir = Path(checkpoint_dir).expanduser().resolve()
    safetensors_path = ckpt_dir / "dino_vitsmall.safetensors"
    ckpt_path = ckpt_dir / "dino_vitsmall.ckpt"

    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file  # type: ignore

            return dict(load_file(str(safetensors_path)))
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Found SPARSH .safetensors weights but failed to load them. "
                "Install `safetensors` or remove the .safetensors file to fall back to the .ckpt loader. "
                f"Path: {safetensors_path}"
            ) from e

    if ckpt_path.exists():
        obj: Any = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj:
            sd = obj["state_dict"]
        else:
            sd = obj
        if not isinstance(sd, dict):
            raise TypeError(f"Unexpected checkpoint payload type: {type(sd)}")
        # Ensure tensors
        return {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}

    raise FileNotFoundError(
        f"No SPARSH weights found in {ckpt_dir}. Expected `dino_vitsmall.safetensors` or `dino_vitsmall.ckpt`."
    )


def _strip_common_prefixes(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    # Prefer student weights if present.
    for preferred in ("student.", "model.student.", "backbone.", "model.backbone."):
        if any(k.startswith(preferred) for k in state_dict):
            return {k[len(preferred) :]: v for k, v in state_dict.items() if k.startswith(preferred)}

    # Otherwise strip generic wrappers.
    prefixes = ("model.", "module.", "encoder.", "teacher.")
    out: dict[str, Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        out[nk] = v
    return out


class SparshDinoSmallTactileEncoder(nn.Module):
    """SPARSH tactile encoder (ViT-Small DINO) for 6-channel tactile image pairs.

    Expects input shaped:
    - (B, 6, H, W) where channels are [I_t (3ch) || I_{t-stride} (3ch)]

    Returns patch tokens (B, N, 384).
    """

    embed_dim: int = 384

    def __init__(
        self,
        checkpoint_dir: str | os.PathLike[str],
        *,
        sparsh_repo_path: str | os.PathLike[str] | None = None,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.checkpoint_dir = str(Path(checkpoint_dir).expanduser().resolve())
        if sparsh_repo_path is None:
            sparsh_repo_path = _find_default_sparsh_repo_from_checkpoint_dir(self.checkpoint_dir)
        self.sparsh_repo_path = str(Path(sparsh_repo_path).expanduser().resolve()) if sparsh_repo_path else None
        self.image_size = int(image_size)

        if self.sparsh_repo_path is not None:
            _maybe_add_repo_to_syspath(self.sparsh_repo_path)

        try:
            from tactile_ssl.model.vision_transformer import vit_small  # type: ignore
        except ModuleNotFoundError as e:  # pragma: no cover
            missing = getattr(e, "name", None)
            if missing:
                raise ImportError(
                    "Failed to import SPARSH (`tactile_ssl`) due to a missing dependency. "
                    f"Missing module: '{missing}'. "
                    "Fix: install it in your environment (e.g. `python -m pip install omegaconf`). "
                    "Also ensure `sparsh_repo_path` points to the SPARSH repo (e.g. /workspace/sparsh)."
                ) from e
            raise
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Failed to import SPARSH (`tactile_ssl`). "
                "Ensure `sparsh_repo_path` points to the SPARSH repo (e.g. /workspace/sparsh) and that SPARSH "
                "python dependencies are installed."
            ) from e

        # SPARSH model: ViT-small, 384-dim patch tokens.
        self.model = vit_small(
            img_size=(self.image_size, self.image_size),
            patch_size=16,
            in_chans=6,
            num_register_tokens=0,
        )

        sd_raw = _load_state_dict_from_checkpoint_dir(self.checkpoint_dir)
        sd = _strip_common_prefixes(sd_raw)
        load_res = self.model.load_state_dict(sd, strict=False)
        self._last_load_result = SparshEncoderLoadResult(
            missing_keys=list(load_res.missing_keys),
            unexpected_keys=list(load_res.unexpected_keys),
        )

        logger = logging.getLogger(__name__)
        if self._last_load_result.missing_keys or self._last_load_result.unexpected_keys:
            logger.warning(
                "Loaded SPARSH tactile encoder with non-strict state_dict. "
                "missing_keys=%d unexpected_keys=%d",
                len(self._last_load_result.missing_keys),
                len(self._last_load_result.unexpected_keys),
            )
        else:
            logger.info("Loaded SPARSH tactile encoder weights successfully (strict match).")

    @property
    def last_load_result(self) -> SparshEncoderLoadResult:
        return self._last_load_result

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected (B,6,H,W) tactile pair tensor, got {tuple(x.shape)}")
        if x.shape[1] != 6:
            raise ValueError(f"Expected 6-channel tactile pair input, got C={x.shape[1]}")

        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        # Return normalized patch tokens.
        out = self.model.forward_features(x)
        return out["x_norm_patchtokens"]
