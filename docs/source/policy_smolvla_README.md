## Paper

https://arxiv.org/abs/2506.01844

## Citation

```bibtex
@article{shukor2025smolvla,
  title={SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics},
  author={Shukor, Mustafa and Aubakirova, Dana and Capuano, Francesco and Kooijmans, Pepijn and Palma, Steven and Zouitine, Adil and Aractingi, Michel and Pascal, Caroline and Russi, Martino and Marafioti, Andres and Alibert, Simon and Cord, Matthieu and Wolf, Thomas and Cadene, Remi},
  journal={arXiv preprint arXiv:2506.01844},
  year={2025}
}
```

## Using SPARSH as a tactile encoder

If your dataset includes tactile image observations under keys like `observation.tactiles.*` (dtype image/video),
you can embed those tactile inputs with a SPARSH ViT encoder instead of the VLM vision encoder.

Minimal config/CLI example:

```bash
lerobot-train \
  --policy.type=smolvla \
  --policy.tactile_encoder_type=sparsh_dino_small \
  --policy.tactile_encoder_checkpoint_dir=/workspace/sparsh/checkpoints/sparsh-dino-small \
  --policy.sparsh_repo_path=/workspace/sparsh
```

Notes:
- SPARSH expects a 6-channel input formed by concatenating two tactile frames: $I_t \oplus I_{t-\text{stride}}$.
  If the batch does not contain enough history, the current frame is duplicated.
- Tactile keys are detected by prefix (default: `observation.tactiles.`). Override via `--policy.tactile_key_prefix=...`.
