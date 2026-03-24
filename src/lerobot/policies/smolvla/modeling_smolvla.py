#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import logging
import math
from collections import deque
from typing import TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.policies.smolvla.tactile_frequency import split_tactile_low_high
from lerobot.policies.utils import (
    log_model_loading_keys,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.utils import get_safe_dtype


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"

    def __init__(
        self,
        config: SmolVLAConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self._debug_forward_calls = 0
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Lets create processor if the config provided
        # If RTC is not enabled - we still can track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    @classmethod
    def _load_as_safetensor(cls, model, model_file: str, map_location: str, strict: bool):
        """Override to handle vocab size expansion from added special tokens."""
        from safetensors.torch import load_file as load_safetensor_file

        state_dict = load_safetensor_file(model_file, device=map_location)
        model_state_dict = model.state_dict()

        adjusted_state_dict = {}
        for key, ckpt_param in state_dict.items():
            if key in model_state_dict:
                model_param = model_state_dict[key]
                if ckpt_param.shape != model_param.shape:
                    logging.warning(
                        f"Size mismatch for '{key}': checkpoint {tuple(ckpt_param.shape)} vs "
                        f"model {tuple(model_param.shape)}. "
                        "Copying overlapping rows; extra rows keep random initialization."
                    )
                    new_param = model_param.clone()
                    slices = tuple(slice(0, min(s, m)) for s, m in zip(ckpt_param.shape, model_param.shape))
                    new_param[slices] = ckpt_param[slices].to(dtype=model_param.dtype)
                    adjusted_state_dict[key] = new_param
                else:
                    adjusted_state_dict[key] = ckpt_param
            else:
                adjusted_state_dict[key] = ckpt_param

        missing_keys, unexpected_keys = model.load_state_dict(adjusted_state_dict, strict=strict)
        log_model_loading_keys(missing_keys, unexpected_keys)
        return model

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch, include_tactile_images=False)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, tactile_state=None, tactile_images=None, tactile_img_masks=None, noise=noise, **kwargs
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _use_tactile_for_vlm_training(self) -> bool:
        return self.training and not self.config.train_expert_only and self.config.tactile_input_type != "none"

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Training batch containing observations and actions.
            noise: Optional noise tensor for flow matching.
            time: Optional time tensor for flow matching.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        #if self.config.adapt_to_pi_aloha: # aloha 로봇을 사용하는 경우 state와 action을 변환합니다.
        #    batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
        #    batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])    

        use_tactile_for_vlm = self._use_tactile_for_vlm_training()
        use_tactile_images = use_tactile_for_vlm and self.config.tactile_input_type == "image"
        use_tactile_state = use_tactile_for_vlm and self.config.tactile_input_type == "state"
        print(f"[debug] use_tactile_for_vlm={use_tactile_for_vlm} use_tactile_images={use_tactile_images} use_tactile_state={use_tactile_state}")

        tactile_state = None
        next_tactile_target = None

        tactile_images = None
        tactile_img_masks = None
        next_tactile_images = None
        next_tactile_img_masks = None

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")

        if use_tactile_images:
            tactile_images, tactile_img_masks = self.prepare_tactile_images(batch)
            next_tactile_images, next_tactile_img_masks = self.prepare_next_tactile_images(batch)
        elif use_tactile_state:
            tactile_state = self.prepare_tactile(batch)
            next_tactile_target = self.prepare_next_tactile_target(batch)

        loss_dict = {}
        losses, tactile_loss = self.model.forward(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            actions=actions,
            tactile_state=tactile_state,
            tactile_images=tactile_images,
            tactile_img_masks=tactile_img_masks,
            next_tactile_target=next_tactile_target,
            next_tactile_images=next_tactile_images,
            next_tactile_img_masks=next_tactile_img_masks,
            noise=noise,
            time=time,
        )
        loss_dict["losses_after_forward"] = losses.clone().mean().item()
        if tactile_loss is not None:
            loss_dict["tactile_loss"] = tactile_loss.mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            if tactile_loss is not None:
                per_sample_loss = per_sample_loss + self.config.next_tactile_loss_weight * tactile_loss
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            if tactile_loss is not None:
                loss = loss + self.config.next_tactile_loss_weight * tactile_loss.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]
        print(f"[debug] present_img_keys={present_img_keys} missing_img_keys={missing_img_keys}")

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def prepare_tactile_images(self, batch):
        """Prepare tactile images with a path distinct from regular camera images."""
        tactile_img_keys = [
            key for key, value in batch.items()
            if key.startswith("observation.tactiles.")
            and not key.endswith("_is_pad")
            and not key.endswith("_padding_mask")
        ]
        tactile_img_keys = sorted(tactile_img_keys)
        
        if len(tactile_img_keys) == 0:
            print("[warning!] tactile_img_keys: no tactile keys found in batch")
            return None, None

        tactile_images = []
        tactile_img_masks = []

        for key in tactile_img_keys:
            img = batch[key]

            # [B, T, C, H, W] -> 마지막 current frame 사용
            if img.ndim == 5:
                img = img[:, -1, :, :, :]
            elif img.ndim != 4:
                print(f"[warning] skip non-image tactile key: {key}, shape={img.shape}")
                continue

            if self.config.tactile_image_resize_with_padding is not None:
                img = resize_with_pad(
                    img,
                    *self.config.tactile_image_resize_with_padding,
                    pad_value=0,
                )

            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device

            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
                if mask.ndim > 1:
                    mask = mask[:, -1]
            elif f"{key}_is_pad" in batch:
                mask = ~batch[f"{key}_is_pad"].bool()
                if mask.ndim > 1:
                    mask = mask[:, -1]
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)

            tactile_images.append(img)
            tactile_img_masks.append(mask)

        if len(tactile_images) == 0:
            return None, None

        return tactile_images, tactile_img_masks


    def prepare_next_tactile_images(self, batch):

        tactile_img_keys = [
            key for key, value in batch.items()
            if key.startswith("observation.tactiles.")
            and not key.endswith("_is_pad")
            and not key.endswith("_padding_mask")
            and torch.is_tensor(value)
            and value.ndim in (4, 5)
        ]
        tactile_img_keys = sorted(tactile_img_keys)

        if len(tactile_img_keys) == 0:
            print("[warning] tactile_img_keys: no valid tactile image keys found in batch")
            return None, None

        next_tactile_images = []
        next_tactile_img_masks = []

        for key in tactile_img_keys:
            img = batch[key]

            # [B, T, C, H, W] -> 마지막 current frame 사용
            if img.ndim == 5:
                img = img[:, -1, :, :, :]
            elif img.ndim != 4:
                print(f"[warning] skip non-image tactile key: {key}, shape={img.shape}")
                continue

            if self.config.tactile_image_resize_with_padding is not None:
                img = resize_with_pad(
                    img,
                    *self.config.tactile_image_resize_with_padding,
                    pad_value=0,
                )

            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device

            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
                if mask.ndim > 1:
                    mask = mask[:, -1]
            elif f"{key}_is_pad" in batch:
                mask = ~batch[f"{key}_is_pad"].bool()
                if mask.ndim > 1:
                    mask = mask[:, -1]
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)

            next_tactile_images.append(img)
            next_tactile_img_masks.append(mask)

        if len(next_tactile_images) == 0:
            return None, None

        return next_tactile_images, next_tactile_img_masks

    def _resolve_next_tactile_image_keys(self, batch: dict[str, Tensor]) -> list[str]:
        """Resolve next tactile image keys by deriving from current tactile keys.
        We don't use next_tactile_image_feature_keys config — instead derive from
        the current observation keys (observation.tactiles.* → next_observation.tactiles.*).
        If explicit next keys are absent, return [] so prepare_next_tactile_images
        falls back to the temporal index=1 path.
        """

        tactile_img_keys = [
            key
            for key, value in batch.items()
            if key.startswith("observation.tactiles.") and isinstance(value, torch.Tensor) and value.ndim >= 4
        ]
        tactile_img_keys = sorted(tactile_img_keys)

        derived_next_keys = [
            key.replace("observation.", "next_observation.", 1)
            for key in tactile_img_keys
            if key.startswith("observation.") and
               key.replace("observation.", "next_observation.", 1) in batch
        ]

        return derived_next_keys

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state."""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]

        if state.shape[-1] > self.config.max_state_dim:
            raise ValueError(
                f"Prepared state dim ({state.shape[-1]}) exceeds `max_state_dim` ({self.config.max_state_dim})."
            )

        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_tactile(self, batch): # 텍타일 신호를 저/고주파수로 분리합니다.
        """Prepare optional tactile input as a separate embedding path."""
        if self.config.tactile_input_type != "state":
            return None

        # Look for tactile state keys with the pattern "observation.tactiles.*"
        # and exclude mask/bool tensors (e.g. "*_padding_mask").
        tactile_key = None
        for key, value in batch.items():
            if not key.startswith("observation.tactiles."):
                continue
            if key.endswith("_padding_mask"):
                continue
            if not isinstance(value, torch.Tensor):
                continue
            if value.dtype == torch.bool:
                continue
            if value.ndim < 2 or value.ndim > 3:
                continue
            tactile_key = key
            break
        
        if tactile_key is None:
            return None

        tactile = batch[tactile_key]
        tactile_low, tactile_high = split_tactile_low_high(
            tactile, lowpass_window=self.config.tactile_lowpass_window
        )

        has_explicit_next_tactile = any(
            key.startswith("next_observation.tactiles.") for key in batch.keys()
        )
        use_temporal_next_target = not has_explicit_next_tactile
        select_idx = 0 if use_temporal_next_target else -1

        if self._should_debug_print():
            print(
                "[SmolVLA][debug][TACTILE] "
                f"prepare_tactile(state): key={tactile_key} raw_shape={tuple(tactile.shape)} "
                f"has_explicit_next={has_explicit_next_tactile} select_idx={select_idx}"
            )

        tactile_parts = []
        if self.config.use_tactile_low_freq:
            tactile_parts.append(tactile_low[:, select_idx, :] if tactile_low.ndim > 2 else tactile_low)
        if self.config.use_tactile_high_freq:
            tactile_parts.append(tactile_high[:, select_idx, :] if tactile_high.ndim > 2 else tactile_high)

        if not tactile_parts:
            return None

        tactile_state = torch.cat(tactile_parts, dim=-1)

        if tactile_state.shape[-1] > self.config.max_tactile_dim:
            raise ValueError(
                f"Prepared tactile dim ({tactile_state.shape[-1]}) exceeds `max_tactile_dim` ({self.config.max_tactile_dim}). "
                "Increase `max_tactile_dim` to include selected tactile features."
            )

        tactile_state = pad_vector(tactile_state, self.config.max_tactile_dim)
        return tactile_state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def prepare_next_tactile_target(self, batch):

        tactile_img_keys = [
            key
            for key, value in batch.items()
            if key.startswith("observation.tactiles.") and isinstance(value, torch.Tensor) and value.ndim >= 4
        ]
        tactile_img_keys = sorted(tactile_img_keys)

        tactile_source = batch[tactile_img_keys]
        if tactile_source.ndim <= 2 or tactile_source.shape[1] < 2:
            return None
        tactile_target = tactile_source[:, 1, :]

        if tactile_target.ndim > 2:
            tactile_target = tactile_target[:, -1, :]

        if tactile_target.shape[-1] > self.config.next_tactile_target_dim:
            raise ValueError(
                f"Prepared next tactile target dim ({tactile_target.shape[-1]}) exceeds "
                f"`next_tactile_target_dim` ({self.config.next_tactile_target_dim})."
            )

        tactile_target = pad_vector(tactile_target, self.config.next_tactile_target_dim)
        return tactile_target

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for SmolVLA fine-tuning."""
        common_projections = (
            "state_proj|tactile_proj|tactile_image_connector|tactile_image_connector_out_proj|"
            "next_tactile_head|next_tactile_image_head|"
            "action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }

    def _validate_peft_config(self, peft_config) -> None:
        """Validate PEFT configuration for SmolVLA."""
        super()._validate_peft_config(peft_config)
        if not self.config.load_vlm_weights:
            import logging

            logging.warning(
                "Training SmolVLA from scratch using PEFT. This is unlikely to yield good results. "
                "Set `load_vlm_weights=True` to fine-tune the existing policy."
            )


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class SmallTactileConvEncoder(nn.Module):
    """
    Tiny CNN encoder for tactile images.
    Outputs a single token per image: (B, 1, out_dim).

    This avoids SmolVLM connector pixel_shuffle constraints (e.g., failing on 16x16).
    """

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = int(out_dim)

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Conv2d(256, self.out_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected tactile image tensor (B,C,H,W), got {tuple(x.shape)}")

        b, c, _, _ = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)
        elif c != 3:
            raise ValueError(f"Expected C=1 or C=3 for tactile images, got C={c}")

        feats = self.net(x)  # (B, D, h, w)
        pooled = self.pool(feats).view(b, self.out_dim)  # (B, D)
        return pooled.unsqueeze(1)  # (B, 1, D)
    

class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.tactile_proj = nn.Linear(
            self.config.max_tactile_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        
        self.tactile_image_encoder = SmallTactileConvEncoder(
            out_dim=self.vlm_with_expert.config.text_config.hidden_size
        )
             
        tactile_connector_hidden_dim = self.config.tactile_image_connector_hidden_dim
        tactile_connector_out_dim = self.config.tactile_image_connector_out_dim
        
        tactile_connector_in_dim = self.vlm_with_expert.config.text_config.hidden_size
        self.tactile_image_connector = nn.Sequential(
            nn.Linear(tactile_connector_in_dim, tactile_connector_hidden_dim),
            nn.GELU(),
            nn.Linear(tactile_connector_hidden_dim, tactile_connector_out_dim),
        )
        text_hidden_size = self.vlm_with_expert.config.text_config.hidden_size
        if text_hidden_size != tactile_connector_out_dim:
            self.tactile_image_connector_out_proj = nn.Linear(tactile_connector_out_dim, text_hidden_size)
        else:
            self.tactile_image_connector_out_proj = nn.Identity()
        self.next_tactile_head = nn.Sequential(
            nn.LayerNorm(self.vlm_with_expert.expert_hidden_size),
            nn.Linear(self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size),
            nn.GELU(),
            nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.next_tactile_target_dim),
        )
        self.next_tactile_image_head = nn.Sequential(
            nn.LayerNorm(self.vlm_with_expert.expert_hidden_size),
            nn.Linear(self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size),
            nn.GELU(),
            nn.Linear(self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.config.text_config.hidden_size),
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.add_tactile_special_tokens = self.config.add_tactile_special_tokens
        self.tactile_start_token = None
        self.tactile_end_token = None
        if self.add_tactile_special_tokens:
            tactile_special_tokens = [
                self.config.tactile_start_special_token,
                self.config.tactile_end_special_token,
            ]
            tokenizer = self.vlm_with_expert.processor.tokenizer
            added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": tactile_special_tokens})
            if added_tokens > 0:
                self.vlm_with_expert.vlm.resize_token_embeddings(len(tokenizer))

            tactile_start_id = tokenizer.convert_tokens_to_ids(self.config.tactile_start_special_token)
            tactile_end_id = tokenizer.convert_tokens_to_ids(self.config.tactile_end_special_token)
            self.tactile_start_token = torch.tensor([tactile_start_id], dtype=torch.long)
            self.tactile_end_token = torch.tensor([tactile_end_id], dtype=torch.long)

        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj
        for params in self.tactile_proj.parameters():
            params.requires_grad = self.config.train_state_proj
        for params in self.tactile_image_connector.parameters():
            params.requires_grad = self.config.train_state_proj
        for params in self.tactile_image_connector_out_proj.parameters():
            params.requires_grad = self.config.train_state_proj
        for params in self.next_tactile_head.parameters():
            params.requires_grad = self.config.train_state_proj
        for params in self.next_tactile_image_head.parameters():
            params.requires_grad = self.config.train_state_proj

    def _embed_tactile_image_tokens(self, tactile_img: torch.Tensor) -> torch.Tensor:
        """
        Tactile image -> tokens (B, T, text_hidden)
        Uses small CNN to avoid SmolVLM pixel_shuffle.
        """
        enc_param = next(self.tactile_image_encoder.parameters())
        tactile_img_emb = self.tactile_image_encoder(tactile_img.to(dtype=enc_param.dtype))

        tactile_img_emb = self.tactile_image_connector(tactile_img_emb)
        tactile_img_emb = self.tactile_image_connector_out_proj(tactile_img_emb)

        # Normalize embeddings (same style as existing code)
        d = tactile_img_emb.shape[-1]
        tactile_img_emb = tactile_img_emb * torch.tensor(d**0.5, dtype=tactile_img_emb.dtype, device=tactile_img_emb.device)
        return tactile_img_emb

    def _encode_tactile_image_target_latent(self, tactile_images, tactile_img_masks):
        if tactile_images is None or tactile_img_masks is None:
            return None

        image_latents = []
        image_masks = []
        for tactile_img, tactile_img_mask in zip(tactile_images, tactile_img_masks, strict=False):
            tactile_img_emb = self._embed_tactile_image_tokens(tactile_img)
            pooled_image_latent = tactile_img_emb.mean(dim=1)
            image_latents.append(pooled_image_latent)
            image_masks.append(tactile_img_mask.bool())

        if len(image_latents) == 0:
            return None

        stacked_latents = torch.stack(image_latents, dim=1)
        stacked_masks = torch.stack(image_masks, dim=1)
        valid_counts = stacked_masks.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=stacked_latents.dtype)
        weighted_latents = stacked_latents * stacked_masks.unsqueeze(-1).to(dtype=stacked_latents.dtype)
        return weighted_latents.sum(dim=1) / valid_counts

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state: torch.Tensor = None,
        tactile_state: torch.Tensor | None = None,
        tactile_images=None,
        tactile_img_masks=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1]) # 이미지 종료 토큰들을 0으로 설정합니다.
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        language_like_embs = [lang_emb]
        language_like_masks = [lang_masks]
        tactile_language_embs = []
        tactile_language_masks = []
        bsize = lang_emb.shape[0]
        device = lang_emb.device

        if tactile_state is not None:
            tactile_emb = self.tactile_proj(tactile_state)
            tactile_emb = tactile_emb[:, None, :] if tactile_emb.ndim == 2 else tactile_emb

            tactile_seq_len = tactile_emb.shape[1]
            tactile_mask = torch.ones(bsize, tactile_seq_len, dtype=torch.bool, device=device)

            if self.config.merge_tactile_into_language_tokens:
                tactile_language_embs.append(tactile_emb)
                tactile_language_masks.append(tactile_mask)

        if tactile_images is not None and tactile_img_masks is not None:
            for tactile_img, tactile_img_mask in zip(tactile_images, tactile_img_masks, strict=False):
                tactile_img_emb = self._embed_tactile_image_tokens(tactile_img)

                _, num_tactile_img_embs = tactile_img_emb.shape[:2]
                tactile_img_mask = tactile_img_mask[:, None].expand(bsize, num_tactile_img_embs)

                if self.config.merge_tactile_into_language_tokens:
                    tactile_language_embs.append(tactile_img_emb)
                    tactile_language_masks.append(tactile_img_mask)

        if self.config.merge_tactile_into_language_tokens and len(tactile_language_embs) > 0:
            if self.add_tactile_special_tokens and self.tactile_start_token is not None and self.tactile_end_token is not None:
                tactile_start_emb = (
                    self.vlm_with_expert.embed_language_tokens(self.tactile_start_token.to(device=self.vlm_with_expert.vlm.device))
                    .unsqueeze(0)
                    .expand(bsize, -1, -1)
                )
                tactile_end_emb = (
                    self.vlm_with_expert.embed_language_tokens(self.tactile_end_token.to(device=self.vlm_with_expert.vlm.device))
                    .unsqueeze(0)
                    .expand(bsize, -1, -1)
                )
                tactile_start_mask = torch.ones(
                    bsize,
                    tactile_start_emb.shape[1],
                    dtype=torch.bool,
                    device=device,
                )
                tactile_end_mask = torch.ones(
                    bsize,
                    tactile_end_emb.shape[1],
                    dtype=torch.bool,
                    device=device,
                )
                language_like_embs.append(tactile_start_emb)
                language_like_masks.append(tactile_start_mask)

            language_like_embs.extend(tactile_language_embs)
            language_like_masks.extend(tactile_language_masks)

            if self.add_tactile_special_tokens and self.tactile_start_token is not None and self.tactile_end_token is not None:
                language_like_embs.append(tactile_end_emb)
                language_like_masks.append(tactile_end_mask)

        merged_language_emb = torch.cat(language_like_embs, dim=1)
        merged_language_mask = torch.cat(language_like_masks, dim=1)

        embs.append(merged_language_emb)
        pad_masks.append(merged_language_mask)
        att_masks += [0] * merged_language_emb.shape[1]

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len) # 상태 토큰들을 1로 설정합니다.

        if tactile_state is not None and not self.config.merge_tactile_into_language_tokens:
            tactile_emb = self.tactile_proj(tactile_state)
            tactile_emb = tactile_emb[:, None, :] if tactile_emb.ndim == 2 else tactile_emb
            embs.append(tactile_emb)

            tactile_seq_len = tactile_emb.shape[1]
            tactile_mask = torch.ones(bsize, tactile_seq_len, dtype=torch.bool, device=device)
            pad_masks.append(tactile_mask)
            att_masks += [1] * tactile_seq_len

        if tactile_images is not None and tactile_img_masks is not None and not self.config.merge_tactile_into_language_tokens:
                tactile_img_emb = self._embed_tactile_image_tokens(tactile_img)

                _, num_tactile_img_embs = tactile_img_emb.shape[:2]
                tactile_img_mask = tactile_img_mask[:, None].expand(bsize, num_tactile_img_embs)

                embs.append(tactile_img_emb)
                pad_masks.append(tactile_img_mask)
                att_masks += [1] * num_tactile_img_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        tactile_state=None,
        tactile_images=None,
        tactile_img_masks=None,
        next_tactile_target=None,
        next_tactile_images=None,
        next_tactile_img_masks=None,
        noise=None,
        time=None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
            tactile_state=tactile_state,
            tactile_images=tactile_images,
            tactile_img_masks=tactile_img_masks,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")

        next_tactile_loss = None
        next_tactile_image_loss = None

        if next_tactile_images is not None and next_tactile_img_masks is not None:
            if self.config.next_tactile_image_predict_from == "mean_suffix_tokens":
                tactile_image_context = suffix_out.mean(dim=1)
            else:
                tactile_image_context = suffix_out[:, -1, :]

            next_tactile_image_pred = self.next_tactile_image_head(tactile_image_context)
            with torch.no_grad():
                next_tactile_image_target = self._encode_tactile_image_target_latent(
                    next_tactile_images,
                    next_tactile_img_masks,
                )

            if next_tactile_image_target is not None:
                next_tactile_image_loss = F.mse_loss(
                    next_tactile_image_pred,
                    next_tactile_image_target,
                    reduction="none",
                ).mean(dim=-1)

        tactile_losses = []

        if next_tactile_loss is not None:
            tactile_losses.append(next_tactile_loss)

        if next_tactile_image_loss is not None:
            tactile_losses.append(next_tactile_image_loss)

        if len(tactile_losses) == 0:
            tactile_loss = None
        elif len(tactile_losses) == 1:
            tactile_loss = tactile_losses[0]
        else:
            tactile_loss = sum(tactile_losses) / len(tactile_losses)

        return losses, tactile_loss

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        tactile_state=None,
        tactile_images=None,
        tactile_img_masks=None,
        noise=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state, tactile_state=tactile_state, tactile_images=tactile_images, tactile_img_masks=tactile_img_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
