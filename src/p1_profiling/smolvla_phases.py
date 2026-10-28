"""Static-shape SmolVLA building blocks for PyTorchSim.

Each phase re-implements one SmolVLA layer in plain torch ops (no dynamic masks,
no HF cache objects) so it can be compiled by the NPU backend. Real checkpoint
weights are used, cast to fp32. Mirrors lerobot's smolvlm_with_expert.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT = "/workspace/3-MDP/smolvla_base"

# SmolVLA (SmolVLM2-500M) shapes, see results/ and lerobot config.
N_CAMERAS = 3
N_PATCHES = 1024  # 512x512 image, 16x16 patches
IMG_TOKENS = 64  # after pixel shuffle (scale 4)
LANG_TOKENS = 48
PREFIX_LEN = N_CAMERAS * IMG_TOKENS + LANG_TOKENS + 1  # + state token
CHUNK = 50
N_VISION_LAYERS = 12
N_VLM_LAYERS = 16
N_EXPERT_LAYERS = 16
NUM_STEPS = 10
Q_HEADS, KV_HEADS, HEAD_DIM = 15, 5, 64


def load_policy():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    return SmolVLAPolicy.from_pretrained(CKPT).to("cpu").eval().float()


class RMSNorm(nn.Module):
    def __init__(self, ref):
        super().__init__()
        self.weight = nn.Parameter(ref.weight.detach().clone())
        self.eps = ref.variance_epsilon

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


def rope(x, cos, sin):
    # x: (B, L, H, D); cos/sin: (1, L, 1, D/2). Same rotate-half form as lerobot apply_rope.
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def attention(q, k, v, mask):
    # q: (B, Lq, 15, D); k, v: (B, Lk, 5, D); mask: (Lq, Lk) additive.
    rep = q.shape[2] // k.shape[2]
    k = (
        k.unsqueeze(3)
        .expand(-1, -1, -1, rep, -1)
        .reshape(k.shape[0], k.shape[1], -1, k.shape[-1])
    )
    v = (
        v.unsqueeze(3)
        .expand(-1, -1, -1, rep, -1)
        .reshape(v.shape[0], v.shape[1], -1, v.shape[-1])
    )
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    s = torch.matmul(q, k.transpose(-1, -2)) * (q.shape[-1] ** -0.5) + mask
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v).transpose(1, 2)
    return o.reshape(o.shape[0], o.shape[1], -1)


class MLP(nn.Module):
    def __init__(self, ref):
        super().__init__()
        self.gate, self.up, self.down = ref.gate_proj, ref.up_proj, ref.down_proj

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def patchify(img):
    """(1,3,512,512) image -> (1,1024,768) patches, features in the conv weight's (C, kh, kw) order.

    stride == kernel, so the 16x16 conv is exactly this reshape followed by a 768x768 GEMM. Runs on the host,
    with the rest of the image preprocessing: on the NPU even a split into two 4-D permutes fails to compile
    ("Not supporting this view operation").
    """
    return (
        img.reshape(1, 3, 32, 16, 32, 16)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(1, 1024, 768)
    )


class VisionPatchEmbed(nn.Module):
    """Patch embedding as a GEMM on patchify(img) + learned position embedding (static ids).

    The SigLIP conv (3->768, 16x16, stride 16) is rewritten as patchify + Linear: PyTorchSim's strided-conv
    template uses only K=3 of the 128 array rows (2.3% PE occupancy).
    """

    def __init__(self, policy):
        super().__init__()
        emb = policy.model.vlm_with_expert.get_vlm_model().vision_model.embeddings
        src = emb.patch_embedding
        self.proj = nn.Linear(3 * 16 * 16, 768)
        self.proj.weight = nn.Parameter(src.weight.detach().reshape(768, -1).clone())
        self.proj.bias = nn.Parameter(src.bias.detach().clone())
        # SigLIP picks position ids with bucketize(h/32 * (1-1e-6)), which for a full 32x32 grid is NOT
        # the identity (row/col k>=1 lands on bucket k-1); reproduce it so the output matches lerobot.
        n = 512 // 16
        bounds = torch.arange(
            1 / emb.num_patches_per_side, 1.0, 1 / emb.num_patches_per_side
        )
        coords = torch.bucketize(
            torch.arange(n).float() / n * (1 - 1e-6), bounds, right=True
        )
        ids = (coords[:, None] * emb.num_patches_per_side + coords).flatten()
        self.register_buffer(
            "pos", emb.position_embedding.weight.detach()[ids].clone()[None]
        )

    def forward(self, x):  # (1, 1024, 768) = patchify(img)
        return self.proj(x) + self.pos


class VisionLayer(nn.Module):
    """One SigLIP encoder layer, 1024 tokens, no padding mask.

    Attention goes through F.scaled_dot_product_attention: under sdpa_kernel(FLASH_ATTENTION) PyTorchSim lowers
    it to its fused flash template (scores stay on chip). Only vision qualifies (L == S, equal Q/KV heads, no
    mask); the VLM and expert attentions use `attention` below, because the template ignores masks.
    """

    def __init__(self, policy, idx=0):
        super().__init__()
        l = policy.model.vlm_with_expert.get_vlm_model().vision_model.encoder.layers[
            idx
        ]
        self.ln1, self.ln2 = l.layer_norm1, l.layer_norm2
        self.q, self.k, self.v, self.o = (
            l.self_attn.q_proj,
            l.self_attn.k_proj,
            l.self_attn.v_proj,
            l.self_attn.out_proj,
        )
        self.fc1, self.fc2 = l.mlp.fc1, l.mlp.fc2
        self.h = l.self_attn.num_heads

    def forward(self, x):  # (1, 1024, 768)
        b, n, d = x.shape
        y = self.ln1(x)
        q, k, v = (
            f(y).view(b, n, self.h, -1).transpose(1, 2)
            for f in (self.q, self.k, self.v)
        )
        o = F.scaled_dot_product_attention(q, k, v)
        x = x + self.o(o.transpose(1, 2).reshape(b, n, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))


class Connector(nn.Module):
    """Connector: x4 pixel-shuffle + modality projection, (1,1024,768) -> (1,64,960), both on the NPU.

    The shuffle is written as one 4-D copy (pixel_shuffle_4d); the original two-permute form does not lower.
    """

    def __init__(self, policy):
        super().__init__()
        self.proj = (
            policy.model.vlm_with_expert.get_vlm_model().connector.modality_projection.proj
        )

    def forward(self, x):
        return self.proj(pixel_shuffle_4d(x))


def pixel_shuffle(x, s=4, g=32):
    b, n, d = x.shape
    # identical to Idefics3Connector.pixel_shuffle (two permutes)
    x = (
        x.view(b, g, g // s, d * s)
        .permute(0, 2, 1, 3)
        .reshape(b, g // s, g // s, d * s * s)
    )
    return x.permute(0, 2, 1, 3).reshape(b, (g // s) ** 2, d * s * s)


def pixel_shuffle_4d(x, s=4, g=32):
    """Same result as pixel_shuffle with a single 4-D permute (batch 1).

    Token (h, w) with h = H*s + hi, w = W*s + wi maps to output token (H, W), feature (hi, wi, d), so the
    two permutes collapse to (H, hi, W, wi*d) -> (H, W, hi, wi*d).
    """
    d = x.shape[-1]
    x = x.reshape(g // s, s, g // s, s * d).permute(0, 2, 1, 3).contiguous()
    return x.reshape(1, (g // s) ** 2, d * s * s)


class VLMLayer(nn.Module):
    """One SmolVLM text layer on the prefix; returns hidden state and the KV pair that fills the cache."""

    def __init__(self, policy, idx=0):
        super().__init__()
        l = policy.model.vlm_with_expert.get_vlm_model().text_model.layers[idx]
        self.n1, self.n2 = RMSNorm(l.input_layernorm), RMSNorm(
            l.post_attention_layernorm
        )
        a = l.self_attn
        self.q, self.k, self.v, self.o = a.q_proj, a.k_proj, a.v_proj, a.o_proj
        self.mlp = MLP(l.mlp)

    def forward(self, x, cos, sin, mask):  # x: (1, 241, 960)
        b, n, _ = x.shape
        y = self.n1(x)
        q = rope(self.q(y).view(b, n, Q_HEADS, HEAD_DIM), cos, sin)
        k = rope(self.k(y).view(b, n, KV_HEADS, HEAD_DIM), cos, sin)
        v = self.v(y).view(b, n, KV_HEADS, HEAD_DIM)
        x = x + self.o(attention(q, k, v, mask))
        return x + self.mlp(self.n2(x)), k, v


class ExpertLayer(nn.Module):
    """Action-expert layer over 50 action tokens attending to the cached prefix KV.

    cross=False (even layers): k/v from the expert's own tokens, concatenated after the cache.
    cross=True  (odd layers):  k/v projections are applied to the cached prefix KV (320->320).
    """

    def __init__(self, policy, idx, cross):
        super().__init__()
        l = policy.model.vlm_with_expert.lm_expert.layers[idx]
        self.cross = cross
        self.n1, self.n2 = RMSNorm(l.input_layernorm), RMSNorm(
            l.post_attention_layernorm
        )
        a = l.self_attn
        self.q, self.k, self.v, self.o = a.q_proj, a.k_proj, a.v_proj, a.o_proj
        self.mlp = MLP(l.mlp)

    def forward(
        self, x, k_cache, v_cache, cos, sin, mask
    ):  # x: (1, 50, 720); cache: (1, 241, 5, 64)
        b, n, _ = x.shape
        y = self.n1(x)
        q = rope(self.q(y).view(b, n, Q_HEADS, HEAD_DIM), cos, sin)
        if self.cross:
            k = self.k(k_cache.reshape(b, k_cache.shape[1], -1)).view(
                b, -1, KV_HEADS, HEAD_DIM
            )
            v = self.v(v_cache.reshape(b, v_cache.shape[1], -1)).view(
                b, -1, KV_HEADS, HEAD_DIM
            )
        else:
            k = torch.cat(
                [k_cache, rope(self.k(y).view(b, n, KV_HEADS, HEAD_DIM), cos, sin)],
                dim=1,
            )
            v = torch.cat([v_cache, self.v(y).view(b, n, KV_HEADS, HEAD_DIM)], dim=1)
        x = x + self.o(attention(q, k, v, mask))
        return x + self.mlp(self.n2(x))


def _rope_tables(n, offset=0):
    inv = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    ang = (torch.arange(n).float() + offset)[:, None] * inv[None]
    return torch.cos(ang)[None, :, None], torch.sin(ang)[None, :, None]


# phase name -> (builder(policy) -> (module, example inputs), multiplier for one full inference)
def build(phase, policy):
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)
    cos_p, sin_p = _rope_tables(PREFIX_LEN)
    if phase == "patch_embed":
        return VisionPatchEmbed(policy), (patchify(r(1, 3, 512, 512)),), N_CAMERAS
    if phase == "vision_layer":
        return VisionLayer(policy), (r(1, N_PATCHES, 768),), N_CAMERAS * N_VISION_LAYERS
    if phase == "connector":
        return Connector(policy), (r(1, N_PATCHES, 768),), N_CAMERAS
    if phase == "vlm_layer":
        mask = torch.zeros(PREFIX_LEN, PREFIX_LEN)
        return (
            VLMLayer(policy),
            (r(1, PREFIX_LEN, 960), cos_p, sin_p, mask),
            N_VLM_LAYERS,
        )
    if phase in ("expert_self_layer", "expert_cross_layer"):
        cross = phase == "expert_cross_layer"
        # cross layers re-base expert positions to 0; self layers continue after the prefix
        cos, sin = _rope_tables(CHUNK, 0 if cross else PREFIX_LEN)
        lk = PREFIX_LEN if cross else PREFIX_LEN + CHUNK
        mask = torch.zeros(CHUNK, lk)
        mod = ExpertLayer(policy, 1 if cross else 0, cross)
        args = (
            r(1, CHUNK, 720),
            r(1, PREFIX_LEN, KV_HEADS, HEAD_DIM),
            r(1, PREFIX_LEN, KV_HEADS, HEAD_DIM),
            cos,
            sin,
            mask,
        )
        return mod, args, (N_EXPERT_LAYERS // 2) * NUM_STEPS
    raise ValueError(phase)


def npu_context():
    """Context for every NPU compile/run: route F.scaled_dot_product_attention to PyTorchSim's flash template.

    The PyTorchSim device disables the flash backend by default (torch_openreg/openreg/__init__.py), so without
    this the vision attention silently decomposes into bmm + softmax kernels that round-trip DRAM.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel([SDPBackend.FLASH_ATTENTION])


PHASES = [
    "patch_embed",
    "vision_layer",
    "connector",
    "vlm_layer",
    "expert_self_layer",
    "expert_cross_layer",
]
