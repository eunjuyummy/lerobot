from __future__ import annotations

import argparse
import os

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check SPARSH tactile encoder wiring in SmolVLA.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.environ.get("SPARSH_CKPT_DIR", "/workspace/sparsh/checkpoints/sparsh-dino-small"),
    )
    parser.add_argument(
        "--sparsh_repo_path",
        type=str,
        default=os.environ.get("SPARSH_REPO", "/workspace/sparsh"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=6, help="tactile history length T")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--token_len", type=int, default=16)
    args = parser.parse_args()

    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

    cfg = SmolVLAConfig(
        device=args.device,
        load_vlm_weights=False,  # avoids downloading large VLM weights
        tactile_encoder_type="sparsh_dino_small",
        tactile_encoder_checkpoint_dir=args.checkpoint_dir,
        sparsh_repo_path=args.sparsh_repo_path,
        sparsh_temporal_stride=args.stride,
    )

    model = VLAFlowMatching(cfg)
    model.eval()
    model.to(args.device)

    # --- 1) Check tactile embedding path with channel-first sequence (B,T,C,H,W), float in [0,1] ---
    tactile_seq = torch.rand(args.batch, args.seq_len, 3, args.height, args.width, device=args.device)
    tactile_token = model._embed_tactile(tactile_seq)
    print("tactile_token (BTCHW float) ->", tuple(tactile_token.shape), tactile_token.dtype)

    # --- 2) Check channel-last input (B,T,H,W,C) ---
    tactile_seq_cl = tactile_seq.permute(0, 1, 3, 4, 2).contiguous()
    tactile_token_cl = model._embed_tactile(tactile_seq_cl)
    print("tactile_token (BTHWC float) ->", tuple(tactile_token_cl.shape), tactile_token_cl.dtype)

    # --- 3) Check uint8 input in [0,255] (should auto-scale) ---
    tactile_seq_u8 = (tactile_seq * 255.0).clamp(0, 255).to(torch.uint8)
    tactile_token_u8 = model._embed_tactile(tactile_seq_u8)
    print("tactile_token (uint8) ->", tuple(tactile_token_u8.shape), tactile_token_u8.dtype)

    # --- 4) End-to-end: embed_prefix with a tactile key in img_keys ---
    # images/img_masks are lists; tactile path accepts 5D.
    images = [tactile_seq]
    img_masks = [torch.ones(args.batch, device=args.device, dtype=torch.bool)]
    img_keys = ["observation.tactiles.left_tactile_little_finger_tip"]

    # Create valid token ids within vocab.
    emb = model.vlm_with_expert.get_vlm_model().text_model.get_input_embeddings()
    vocab = emb.num_embeddings
    lang_tokens = torch.randint(low=0, high=vocab, size=(args.batch, args.token_len), device=args.device)
    lang_masks = torch.ones(args.batch, args.token_len, device=args.device, dtype=torch.bool)
    state = torch.zeros(args.batch, cfg.max_state_dim, device=args.device)

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        img_keys=img_keys,
        state=state,
    )

    print("prefix_embs ->", tuple(prefix_embs.shape), prefix_embs.dtype)
    print("prefix_pad_masks ->", tuple(prefix_pad_masks.shape), prefix_pad_masks.dtype)
    print("prefix_att_masks ->", tuple(prefix_att_masks.shape), prefix_att_masks.dtype)

    # Basic numerical sanity.
    if torch.isnan(prefix_embs).any():
        raise RuntimeError("NaNs detected in prefix embeddings")

    print("OK: SPARSH tactile encoder path is producing embeddings.")


if __name__ == "__main__":
    main()
