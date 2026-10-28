"""Whole-stage SmolVLA modules for the end-to-end PyTorchSim run.

Stacks full groups of layers (reusing the single-layer modules of smolvla_phases.py) so
activations and the KV cache move through DRAM between layers exactly as in a real run:

  VisionEncoder : patch embed (GEMM) + 12 SigLIP layers + post-layernorm  (x3 cameras)
  Connector     : pixel-shuffle + projection                             (x3 cameras)
  PrefixStack   : 16 VLM text layers, returns hidden state + 16 (K, V)    (x1)
  DenoiseStep   : action/time MLP + 16 expert layers + norm + out proj    (x10 steps)

Left on the host (not simulated, all tiny): image resize/normalisation and patchify (a pure reshape
that the NPU lowering rejects), token embedding gather and sqrt(dim) scaling, state projection,
token concatenation, the sin/cos timestep embedding (precomputed per step), and the
x_t += dt * v_t update.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import smolvla_phases as sp
from smolvla_phases import (CHUNK, KV_HEADS, HEAD_DIM, N_CAMERAS, N_PATCHES, N_VISION_LAYERS,
                            N_VLM_LAYERS, N_EXPERT_LAYERS, NUM_STEPS, PREFIX_LEN,
                            RMSNorm, VisionPatchEmbed, VisionLayer, Connector, VLMLayer,
                            ExpertLayer, _rope_tables)

NEG = -1e9   # additive mask value for blocked positions (finite so fp32 softmax stays NaN-free)


class VisionEncoder(nn.Module):
    """SigLIP: patch embedding, 12 encoder layers, post-layernorm. patchify(img) (1,1024,768) -> (1,1024,768)."""
    def __init__(self, policy, n_layers=N_VISION_LAYERS):
        super().__init__()
        vm = policy.model.vlm_with_expert.get_vlm_model().vision_model
        self.patch = VisionPatchEmbed(policy)
        self.layers = nn.ModuleList(VisionLayer(policy, i) for i in range(n_layers))
        self.post = vm.post_layernorm

    def forward(self, img):
        x = self.patch(img)
        for layer in self.layers:
            x = layer(x)
        return self.post(x)


class PrefixStack(nn.Module):
    """16 VLM text layers over the 241-token prefix; returns (hidden, k0, v0, ..., k15, v15)."""
    def __init__(self, policy, n_layers=N_VLM_LAYERS):
        super().__init__()
        self.layers = nn.ModuleList(VLMLayer(policy, i) for i in range(n_layers))

    def forward(self, x, cos, sin, mask):
        kv = []
        for layer in self.layers:
            x, k, v = layer(x, cos, sin, mask)
            kv += [k, v]
        return (x, *kv)


class DenoiseStep(nn.Module):
    """One flow-matching step: noisy actions + timestep embedding -> velocity v_t (1,50,32).

    The real model concatenates [action_emb, time_emb] (1440) before action_time_mlp_in; this is
    split into two 720-wide linears that are summed, which is mathematically identical and avoids
    a concat on the NPU. Expert layer i is self-attention for even i, cross-attention for odd i.
    """
    def __init__(self, policy, n_layers=N_EXPERT_LAYERS):
        super().__init__()
        m = policy.model
        self.in_proj = m.action_in_proj
        w = m.action_time_mlp_in.weight.detach()                      # (720, 1440)
        self.mlp_in_a = nn.Linear(720, 720)
        self.mlp_in_a.weight = nn.Parameter(w[:, :720].clone())
        self.mlp_in_a.bias = nn.Parameter(m.action_time_mlp_in.bias.detach().clone())
        self.mlp_in_t = nn.Linear(720, 720, bias=False)
        self.mlp_in_t.weight = nn.Parameter(w[:, 720:].clone())
        self.mlp_out = m.action_time_mlp_out
        self.layers = nn.ModuleList(ExpertLayer(policy, i, cross=(i % 2 == 1)) for i in range(n_layers))
        self.norm = RMSNorm(m.vlm_with_expert.lm_expert.norm)
        self.out_proj = m.action_out_proj

    def forward(self, x_t, time_emb, cos_self, sin_self, cos_cross, sin_cross, mask_self, mask_cross, *kv):
        h = self.mlp_out(F.silu(self.mlp_in_a(self.in_proj(x_t)) + self.mlp_in_t(time_emb)))
        for i, layer in enumerate(self.layers):
            k, v = kv[2 * i], kv[2 * i + 1]
            if layer.cross:
                h = layer(h, k, v, cos_cross, sin_cross, mask_cross)
            else:
                h = layer(h, k, v, cos_self, sin_self, mask_self)
        return self.out_proj(self.norm(h))


def additive(mask_bool):
    """Boolean attention mask (True = may attend) -> additive fp32 mask."""
    return torch.where(mask_bool, 0.0, NEG).float()


def prefix_mask():
    """Image and language tokens attend to each other; the state token (last) attends to all but is
    invisible to them (att_masks = [0]*(3*64+48) + [1], see lerobot make_att_2d_masks)."""
    m = torch.ones(PREFIX_LEN, PREFIX_LEN, dtype=torch.bool)
    m[:-1, -1] = False
    return additive(m)


def denoise_masks():
    """Self layers see the full prefix plus a causal window over the 50 action tokens; cross layers
    see the prefix only."""
    causal = torch.tril(torch.ones(CHUNK, CHUNK, dtype=torch.bool))
    full = torch.cat([torch.ones(CHUNK, PREFIX_LEN, dtype=torch.bool), causal], dim=1)
    return additive(full), additive(full[:, :PREFIX_LEN])


def time_embedding(policy, step):
    """Sinusoidal embedding of the flow time for denoise step `step`, broadcast to (1, 50, 720)."""
    from lerobot.policies.smolvla.modeling_smolvla import create_sinusoidal_pos_embedding
    cfg, m = policy.config, policy.model
    t = torch.tensor([1.0 - step / NUM_STEPS], dtype=torch.float32)
    e = create_sinusoidal_pos_embedding(t, m.vlm_with_expert.expert_hidden_size,
                                        cfg.min_period, cfg.max_period, device=t.device)
    return e.float()[:, None, :].expand(1, CHUNK, -1).contiguous()


def build(stage, policy):
    """Return (module, example_inputs, launches_per_chunk) for a stage."""
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)
    if stage == "vision":
        return VisionEncoder(policy), (sp.patchify(r(1, 3, 512, 512)),), N_CAMERAS
    if stage == "connector":
        return Connector(policy), (r(1, N_PATCHES, 768),), N_CAMERAS
    if stage == "prefix":
        cos, sin = _rope_tables(PREFIX_LEN)
        return PrefixStack(policy), (r(1, PREFIX_LEN, 960), cos, sin, prefix_mask()), 1
    if stage == "denoise":
        cos_s, sin_s = _rope_tables(CHUNK, PREFIX_LEN)
        cos_c, sin_c = _rope_tables(CHUNK, 0)
        m_self, m_cross = denoise_masks()
        kv = [r(1, PREFIX_LEN, KV_HEADS, HEAD_DIM) for _ in range(2 * N_EXPERT_LAYERS)]
        args = (r(1, CHUNK, 32), time_embedding(policy, 0), cos_s, sin_s, cos_c, sin_c, m_self, m_cross, *kv)
        return DenoiseStep(policy), args, NUM_STEPS
    raise ValueError(stage)


STAGES = ["vision", "connector", "prefix", "denoise"]
