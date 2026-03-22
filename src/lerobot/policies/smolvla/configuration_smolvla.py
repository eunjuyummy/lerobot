# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Literal

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32
    tactile_state_dim: int = 468
    max_tactile_dim: int | None = None

    tactile_input_type: Literal["none", "state", "image"] = "state"
    tactile_feature_key: str = "observation.tactile"
    tactile_image_feature_keys: tuple[str, ...] = ()
    tactile_image_resize_with_padding: tuple[int, int] | None = (512, 512)
    tactile_image_connector_hidden_dim: int = 4096
    tactile_image_connector_out_dim: int = 4096
    merge_tactile_into_language_tokens: bool = True
    add_tactile_special_tokens: bool = True
    tactile_start_special_token: str = "<TACTILE_START>"
    tactile_end_special_token: str = "<TACTILE_END>"
    enable_next_tactile_loss: bool = True
    next_tactile_target_key: str = "next_observation.tactile"
    next_tactile_target_dim: int = 468
    next_tactile_loss_weight: float = 0.1
    next_tactile_predict_from: Literal["last_suffix_token", "mean_suffix_tokens"] = "last_suffix_token"
    enable_next_tactile_image_loss: bool = True
    next_tactile_image_feature_keys: tuple[str, ...] = ()
    next_tactile_image_loss_weight: float = 0.1
    next_tactile_image_predict_from: Literal["last_suffix_token", "mean_suffix_tokens"] = "last_suffix_token"
    tactile_lowpass_window: int = 5
    use_tactile_low_freq: bool = True
    use_tactile_high_freq: bool = True

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    debug_tactile_pipeline_prints: bool = False
    debug_print_every_n_steps: int = 100

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )
        if self.tactile_lowpass_window < 1:
            raise ValueError(
                f"`tactile_lowpass_window` must be >= 1. Got {self.tactile_lowpass_window}."
            )
        if self.tactile_input_type not in {"none", "state", "image"}:
            raise ValueError(
                f"`tactile_input_type` must be one of ['none', 'state', 'image']. Got {self.tactile_input_type}."
            )
        if self.tactile_input_type == "image" and len(self.tactile_image_feature_keys) == 0:
            raise ValueError(
                "`tactile_image_feature_keys` must be provided when `tactile_input_type='image'`."
            )
        if self.tactile_state_dim < 1:
            raise ValueError(f"`tactile_state_dim` must be >= 1. Got {self.tactile_state_dim}.")
        if self.tactile_image_connector_hidden_dim < 1:
            raise ValueError(
                f"`tactile_image_connector_hidden_dim` must be >= 1. Got {self.tactile_image_connector_hidden_dim}."
            )
        if self.tactile_image_connector_out_dim < 1:
            raise ValueError(
                f"`tactile_image_connector_out_dim` must be >= 1. Got {self.tactile_image_connector_out_dim}."
            )
        if self.add_tactile_special_tokens:
            if not self.tactile_start_special_token:
                raise ValueError("`tactile_start_special_token` must be a non-empty string.")
            if not self.tactile_end_special_token:
                raise ValueError("`tactile_end_special_token` must be a non-empty string.")
        if self.next_tactile_target_dim < 1:
            raise ValueError(f"`next_tactile_target_dim` must be >= 1. Got {self.next_tactile_target_dim}.")
        if self.next_tactile_loss_weight < 0:
            raise ValueError(f"`next_tactile_loss_weight` must be >= 0. Got {self.next_tactile_loss_weight}.")
        if self.next_tactile_predict_from not in {"last_suffix_token", "mean_suffix_tokens"}:
            raise ValueError(
                "`next_tactile_predict_from` must be one of ['last_suffix_token', 'mean_suffix_tokens']. "
                f"Got {self.next_tactile_predict_from}."
            )
        if self.next_tactile_image_loss_weight < 0:
            raise ValueError(
                f"`next_tactile_image_loss_weight` must be >= 0. Got {self.next_tactile_image_loss_weight}."
            )
        if self.next_tactile_image_predict_from not in {"last_suffix_token", "mean_suffix_tokens"}:
            raise ValueError(
                "`next_tactile_image_predict_from` must be one of ['last_suffix_token', 'mean_suffix_tokens']. "
                f"Got {self.next_tactile_image_predict_from}."
            )
        if self.debug_print_every_n_steps < 1:
            raise ValueError(
                f"`debug_print_every_n_steps` must be >= 1. Got {self.debug_print_every_n_steps}."
            )

        if self.max_tactile_dim is None:
            if self.tactile_input_type == "state":
                enabled_components = int(self.use_tactile_low_freq) + int(self.use_tactile_high_freq)
                if enabled_components == 0:
                    raise ValueError(
                        "When `tactile_input_type='state'`, at least one of `use_tactile_low_freq` or "
                        "`use_tactile_high_freq` must be True."
                    )
                self.max_tactile_dim = self.tactile_state_dim * enabled_components
            else:
                # `tactile_proj` is instantiated regardless of modality, so keep a minimal safe size
                # when tactile state inputs are not used.
                self.max_tactile_dim = 1

        if self.max_tactile_dim < 1:
            raise ValueError(f"`max_tactile_dim` must be >= 1. Got {self.max_tactile_dim}.")

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
