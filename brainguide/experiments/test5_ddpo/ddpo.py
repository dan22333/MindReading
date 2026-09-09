#!/usr/bin/env python
"""TEST 5 (option 1) — DDPO: RL reward fine-tuning of SD, encoder as pure score.

Never differentiate the encoder. Sample images with SD (+ trainable LoRA), score
each finished image with the FROZEN DINOv2 brain-encoder (a scalar reward), and use
policy gradient (REINFORCE + PPO clip, per-batch advantage) to make high-reward
trajectories more likely. Encoder frozen (reward only); SD backbone frozen (LoRA only).

Compact demo settings by default (one ROI, modest rounds). Its own experiment.
  python experiments/test5_ddpo/ddpo.py --config configs/universal.yaml \
      --roi faces --rounds 30 --batch 8 --steps 25 --out exp_out/test5
"""
import argparse, math, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, yaml
BG = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(BG))
from run import build

ROIS = ["faces", "places", "words", "bodies", "earlyvisual"]
SD15 = "runwayml/stable-diffusion-v1-5"


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__(); self.base = base
        for p in base.parameters(): p.requires_grad_(False)
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.normal_(self.A.weight, std=1.0 / r); nn.init.zeros_(self.B.weight)
        self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.scale


def add_lora(unet, r=8):
    params = []
    for name, mod in list(unet.named_modules()):
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in ("to_q", "to_k", "to_v"):
            parent = unet.get_submodule(name.rsplit(".", 1)[0]); attr = name.rsplit(".", 1)[1]
            lora = LoRALinear(getattr(parent, attr), r).to(mod.weight.device, dtype=mod.weight.dtype)
            setattr(parent, attr, lora); params += [lora.A.weight, lora.B.weight]
    return params


def ddim_logprob(scheduler, eps, t, prev_t, x_t, x_prev=None, eta=1.0):
    """DDIM stochastic step: return (sampled x_prev, log p(x_prev|x_t)) per sample."""
    ac = scheduler.alphas_cumprod.to(x_t.device)
    a_t = ac[t]; a_prev = ac[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=x_t.device)
    x0 = (x_t - (1 - a_t).sqrt() * eps) / a_t.sqrt()
    sigma = eta * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).clamp(min=0).sqrt()
    mean = a_prev.sqrt() * x0 + (1 - a_prev - sigma**2).clamp(min=0).sqrt() * eps
    if x_prev is None:
        x_prev = mean + sigma * torch.randn_like(mean)
    var = sigma**2 + 1e-8
    lp = -0.5 * (((x_prev - mean)**2) / var + torch.log(2 * math.pi * var))
    return x_prev, lp.flatten(1).sum(1)         # [B]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/universal.yaml")
    ap.add_argument("--roi", default="faces")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1e-3)
    ap.add_argument("--out", default="exp_out/test5")
    ap.add_argument("--seeds", default="0 1 2 3 4")
    args = ap.parse_args()
    dev = "cuda"; torch.manual_seed(0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(args.config).read_text())
    _, encoder, _, _ = build(cfg)                # frozen DINOv2 brain encoder (reward only)
    roi_vox = torch.as_tensor(np.load(BG / "brainguide" / "roi" / f"sub1_{args.roi}.npy").astype("int64"),
                              device=dev)

    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(SD15, torch_dtype=torch.float32, token=False)
    pipe = pipe.to(dev); pipe.safety_checker = None
    unet, vae, sched = pipe.unet, pipe.vae, pipe.scheduler
    unet.requires_grad_(False); vae.requires_grad_(False)
    lora_params = add_lora(unet, r=8)
    print(f"[ddpo] LoRA params: {sum(p.numel() for p in lora_params)/1e6:.2f}M over UNet attn")
    opt = torch.optim.Adam(lora_params, lr=args.lr)
    with torch.no_grad():
        ids = pipe.tokenizer("", padding="max_length", max_length=pipe.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids.to(dev)
        empty = pipe.text_encoder(ids)[0]        # [1,77,768]
    sched.set_timesteps(args.steps, device=dev)
    tsteps = sched.timesteps

    @torch.no_grad()
    def reward(latents):                          # latents [B,4,64,64] -> ROI activation [B]
        img = vae.decode(latents / vae.config.scaling_factor).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        vox = roi_vox.unsqueeze(0).expand(img.shape[0], -1)
        return encoder(img, vox).mean(1)          # encoder does its own resize+norm

    def sample_traj(bs):
        """Roll out a batch; return stored transitions + final reward."""
        x = torch.randn(bs, 4, 64, 64, device=dev) * sched.init_noise_sigma
        emb = empty.repeat(bs, 1, 1)
        xs, xps, ts, pts, lps = [], [], [], [], []
        for i, t in enumerate(tsteps):
            with torch.no_grad():
                eps = unet(x, t, encoder_hidden_states=emb).sample
            prev_t = tsteps[i + 1] if i + 1 < len(tsteps) else torch.tensor(-1)
            x_prev, lp = ddim_logprob(sched, eps, t, prev_t, x, None)
            xs.append(x); xps.append(x_prev); ts.append(t); pts.append(prev_t); lps.append(lp)
            x = x_prev
        R = reward(x)
        return dict(xs=xs, xps=xps, ts=ts, pts=pts, lp_old=torch.stack(lps, 1), R=R)

    hist = []
    for rd in range(args.rounds):
        traj = sample_traj(args.batch)
        R = traj["R"]; adv = (R - R.mean()) / (R.std() + 1e-6)
        emb = empty.repeat(args.batch, 1, 1)
        # one PPO epoch over the stored trajectory
        loss_sum = 0.0
        opt.zero_grad()
        for i in range(len(traj["ts"])):
            eps = unet(traj["xs"][i], traj["ts"][i], encoder_hidden_states=emb).sample
            _, lp_new = ddim_logprob(sched, eps, traj["ts"][i], traj["pts"][i], traj["xs"][i], traj["xps"][i])
            ratio = torch.exp(lp_new - traj["lp_old"][:, i])
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv
            loss = -torch.min(unclipped, clipped).mean() / len(traj["ts"])
            loss.backward()
            loss_sum += float(loss)
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        hist.append(float(R.mean()))
        print(f"[ddpo] round {rd:3d}  meanR {float(R.mean()):+.4f}  advstd {float(R.std()):.3f}  loss {loss_sum:+.4f}")

    np.save(out / f"ddpo_reward_hist_{args.roi}.npy", np.array(hist))
    torch.save({p_i: p.detach().cpu() for p_i, p in enumerate(lora_params)}, out / f"ddpo_lora_{args.roi}.pt")

    # final samples with the trained policy
    for s in [int(x) for x in args.seeds.split()]:
        g = torch.Generator(device=dev).manual_seed(s)
        x = torch.randn(1, 4, 64, 64, device=dev, generator=g) * sched.init_noise_sigma
        emb = empty
        with torch.no_grad():
            for i, t in enumerate(tsteps):
                eps = unet(x, t, encoder_hidden_states=emb).sample
                prev_t = tsteps[i + 1] if i + 1 < len(tsteps) else torch.tensor(-1)
                x, _ = ddim_logprob(sched, eps, t, prev_t, x, None, eta=0.0)  # deterministic for the final image
            img = vae.decode(x / vae.config.scaling_factor).sample
            img = (img / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        from PIL import Image
        Image.fromarray((img * 255).astype("uint8")).save(out / f"t5_{args.roi}_seed{s}_0.png")
    print(f"[ddpo] done: reward {hist[0]:+.3f} -> {hist[-1]:+.3f}; samples in {out}")


if __name__ == "__main__":
    main()
