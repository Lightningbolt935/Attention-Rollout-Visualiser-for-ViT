"""
Attention Rollout — Implemented from Scratch
=============================================
Paper: "Quantifying Attention Flow in Transformers"
       Abnar & Zuidema, ACL 2020  (arXiv:2005.00928)

The core insight:
  A ViT has L layers, each with H attention heads.
  Each head produces an attention matrix A_l ∈ R^{(N+1) × (N+1)}
  where N = number of image patches, +1 = CLS token.

  Standard attention only shows one layer at a time.
  But information flows through ALL layers — layer 3 attends to
  outputs of layer 2, which attended to layer 1, etc.

  Rollout recursively multiplies attention matrices across layers
  to compute: how much does the CLS token (the final prediction)
  actually attend to each input patch, taking ALL layers into account?

Algorithm (Abnar & Zuidema 2020):
  1. For each layer l, average attention across heads:
       Ā_l = (1/H) Σ_h A_l^h
  2. Add residual connection (identity) to model skip connections:
       Â_l = 0.5 * Ā_l + 0.5 * I
  3. Rollout = Â_L × Â_{L-1} × ... × Â_1
  4. Extract CLS→patch row [0, 1:] and reshape to √N × √N grid
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional


class AttentionRollout:
    """
    Computes Attention Rollout for any ViT model that exposes
    per-layer attention weights.
    """

    def __init__(self, discard_ratio: float = 0.0, head_fusion: str = "mean"):
        """
        Args:
            discard_ratio : fraction of lowest attentions to zero out
                           (noise reduction, 0.0 = keep all)
            head_fusion   : how to combine heads — "mean", "max", or "min"
        """
        assert head_fusion in ("mean", "max", "min")
        self.discard_ratio = discard_ratio
        self.head_fusion   = head_fusion

    def fuse_heads(self, attn: torch.Tensor) -> torch.Tensor:
        """
        attn: (H, N+1, N+1)  — H heads, N+1 tokens (CLS + patches)
        Returns: (N+1, N+1)  — single fused attention matrix
        """
        if self.head_fusion == "mean":
            return attn.mean(dim=0)
        elif self.head_fusion == "max":
            return attn.max(dim=0).values
        else:
            return attn.min(dim=0).values

    def discard_low_attention(self, attn: torch.Tensor) -> torch.Tensor:
        """
        Zero out the bottom `discard_ratio` fraction of attention values
        in each row, then renormalise.  Reduces noise from irrelevant tokens.
        """
        if self.discard_ratio == 0.0:
            return attn
        flat   = attn.view(-1)
        thresh = torch.quantile(flat, self.discard_ratio)
        attn   = attn.clone()
        attn[attn < thresh] = 0.0
        # renormalise each row
        row_sums = attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return attn / row_sums

    def compute(self, attention_maps: List[torch.Tensor]) -> np.ndarray:
        """
        Core rollout algorithm.

        Args:
            attention_maps: list of L tensors, each (H, N+1, N+1)
                            one per transformer layer

        Returns:
            rollout: (N+1, N+1) — accumulated attention from all layers
        """
        n_tokens = attention_maps[0].shape[-1]

        # Start with identity (no attention applied yet)
        result = torch.eye(n_tokens, device=attention_maps[0].device)

        for attn in attention_maps:
            # Step 1: fuse heads → (N+1, N+1)
            fused = self.fuse_heads(attn)

            # Step 2: optional discard + renorm
            fused = self.discard_low_attention(fused)

            # Step 3: add residual (models the skip connection)
            fused = 0.5 * fused + 0.5 * torch.eye(n_tokens, device=fused.device)

            # Step 4: renormalise rows
            fused = fused / fused.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            # Step 5: matrix multiply with running result
            result = torch.matmul(fused, result)

        return result.detach().cpu().numpy()

    def cls_attention_map(
        self,
        attention_maps: List[torch.Tensor],
        grid_size: int,
    ) -> np.ndarray:
        """
        Returns a (grid_size, grid_size) heatmap showing which image
        patches the CLS token attends to after rollout.

        Row 0 of the rollout matrix = CLS token's attention over all tokens.
        Columns 1: = patch tokens (column 0 = CLS attending to itself).
        """
        rollout     = self.compute(attention_maps)   # (N+1, N+1)
        cls_attn    = rollout[0, 1:]                 # (N,)  — N = grid_size²
        cls_attn    = cls_attn / cls_attn.max()      # normalise to [0,1]
        return cls_attn.reshape(grid_size, grid_size)


# ── Layer-wise attention extractor ─────────────────────────────────────────
class ViTAttentionExtractor(nn.Module):
    """
    Wraps a HuggingFace ViT model and registers forward hooks
    on every attention layer to capture attention weights.

    Why hooks?  HuggingFace ViT doesn't return per-layer attention
    by default in the most convenient format; hooks give us direct
    access to the raw attention matrices before they're aggregated.
    """

    def __init__(self, model):
        super().__init__()
        self.model        = model
        self.attention_maps: List[torch.Tensor] = []
        self._hooks       = []
        self._register_hooks()

    def _register_hooks(self):
        """Register a hook on every ViT encoder attention layer."""
        for layer in self.model.vit.encoder.layer:
            hook = layer.attention.attention.register_forward_hook(
                self._hook_fn
            )
            self._hooks.append(hook)

    def _hook_fn(self, module, input, output):
        """
        Called after each attention layer's forward pass.
        `output` from ViTSelfAttention is a tuple; attention weights
        are at index 1 when output_attentions=True.
        Shape: (batch, heads, seq_len, seq_len)
        """
        # output[1] = attention_probs when output_attentions=True
        if isinstance(output, tuple) and len(output) > 1:
            attn = output[1]          # (B, H, N+1, N+1)
            self.attention_maps.append(attn[0])   # take first (only) batch

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(self, pixel_values: torch.Tensor):
        self.attention_maps.clear()
        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                output_attentions=True,
            )
        return outputs, self.attention_maps.copy()
