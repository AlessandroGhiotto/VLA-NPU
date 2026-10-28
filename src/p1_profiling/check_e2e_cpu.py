#!/usr/bin/env python3
"""CPU-only check that the stage modules in smolvla_e2e.py reproduce lerobot's SmolVLA.

Runs the real policy (embed_image, VLM prefix pass with KV fill, one denoise step) and the
re-implemented stacks on the same seeded inputs and prints the error of each stage.
Run inside the container:  python src/p1_profiling/check_e2e_cpu.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

import smolvla_e2e as E
import smolvla_phases as P
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks


def report(name, mine, ref, tol=1e-3):
    err = (mine - ref).abs()
    rel = err.max().item() / max(ref.abs().max().item(), 1e-12)
    ok = rel <= tol
    print(f"[{name}] {'PASS' if ok else 'FAIL'} max abs err {err.max().item():.3e}, "
          f"mean abs err {err.mean().item():.3e}, ref max |x| {ref.abs().max().item():.3e}, rel {rel:.2e}")
    return ok


def main():
    torch.manual_seed(0)
    policy = P.load_policy()
    model = policy.model
    vlm = model.vlm_with_expert
    ok = True

    with torch.no_grad():
        # 1. vision: encoder + host pixel-shuffle + connector projection vs embed_image
        img = torch.rand(1, 3, 512, 512) * 2 - 1
        ref_img_emb = vlm.embed_image(img)                                   # (1, 64, 960)
        ref_enc = vlm.get_vlm_model().vision_model(pixel_values=img).last_hidden_state
        enc = E.VisionEncoder(policy)(P.patchify(img))
        ok &= report("vision encoder", enc, ref_enc)
        ok &= report("pixel_shuffle_4d", P.pixel_shuffle_4d(enc), P.pixel_shuffle(enc), tol=0)
        ok &= report("vision+connector", E.Connector(policy)(enc), ref_img_emb)

        # 2. prefix: embed_prefix gives the real 241-token input; compare hidden state and every K/V
        images = [torch.rand(1, 3, 512, 512) * 2 - 1 for _ in range(3)]
        img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(3)]
        lang = torch.randint(0, 1000, (1, 48))
        lang_masks = torch.ones(1, 48, dtype=torch.bool)
        state = torch.randn(1, 32)
        embs, pad, att = model.embed_prefix(images, img_masks, lang, lang_masks, state=state)
        assert embs.shape[1] == P.PREFIX_LEN, embs.shape
        att2d = make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad, dim=1) - 1
        _, pkv = vlm.forward(attention_mask=att2d, position_ids=pos, past_key_values=None,
                             inputs_embeds=[embs, None], use_cache=True, fill_kv_cache=True)
        ok &= report("prefix mask", E.prefix_mask(), E.additive(att2d[0]), tol=0)
        cos, sin = P._rope_tables(P.PREFIX_LEN)
        out = E.PrefixStack(policy)(embs, cos, sin, E.prefix_mask())
        kv = out[1:]
        for i in (0, 7, 15):
            ok &= report(f"prefix K layer {i}", kv[2 * i], pkv[i]["key_states"])
            ok &= report(f"prefix V layer {i}", kv[2 * i + 1], pkv[i]["value_states"])

        # 3. one denoise step against the real cache
        x_t = torch.randn(1, P.CHUNK, 32)
        step = 3
        t = torch.tensor([1.0 - step / P.NUM_STEPS])
        ref_v = model.denoise_step(prefix_pad_masks=pad, past_key_values=pkv, x_t=x_t, timestep=t)
        cos_s, sin_s = P._rope_tables(P.CHUNK, P.PREFIX_LEN)
        cos_c, sin_c = P._rope_tables(P.CHUNK, 0)
        m_self, m_cross = E.denoise_masks()
        kv_args = [kv[j] for j in range(2 * P.N_EXPERT_LAYERS)]
        mine_v = E.DenoiseStep(policy)(x_t, E.time_embedding(policy, step), cos_s, sin_s, cos_c, sin_c,
                                       m_self, m_cross, *kv_args)
        ok &= report("denoise step v_t", mine_v, ref_v)

    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
