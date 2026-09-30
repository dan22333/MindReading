#!/usr/bin/env python
"""Phase 1 — calibrate the repel hinge thresholds c*_img and c*_txt.

For ~300 natural NSD images (the baseline test set — overwhelmingly non-face scenes) we
measure their cosine to each anchor SET (E1/E2), aggregated exactly as the loss does:
per-image  mean_i cos(img, img_i)  and  max_i cos(img, img_i)  for the image anchors, and
the same against the text anchors. We then report the distribution (percentiles) so the
chosen c* sits at a meaningful point:

  "novelty achieved = as un-face-like as a natural non-face image"  ->  c* ~ low percentile.

Prints where the picked values (0.2 / 0.1 image, 0.1 text) fall, so we can confirm 0.2 is
near the natural-image edge (not below it, which would demand off-manifold weirdness).

  python calib_cstar.py --anchors weights/anchor_e1.pt weights/anchor_e2.pt \
      --baseline_images data/s1_images.npy --baseline_split data/s1_split.npz --baseline_n 300
"""
import argparse
from pathlib import Path
import numpy as np
import torch


def pct(a, ps=(1, 5, 10, 25, 50, 75, 90)):
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", nargs="+", required=True, help="anchor_*.pt files (build_anchor.py)")
    ap.add_argument("--baseline_images", default="data/s1_images.npy")
    ap.add_argument("--baseline_split", default="data/s1_split.npz")
    ap.add_argument("--baseline_n", type=int, default=300)
    ap.add_argument("--c_img", nargs="+", type=float, default=[0.2, 0.1])
    ap.add_argument("--c_txt", type=float, default=0.1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    import open_clip
    from PIL import Image
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    clip_model = clip_model.eval().to(device)

    def emb(uint8_imgs):
        px = torch.stack([preprocess(Image.fromarray(x)) for x in uint8_imgs]).to(device)
        with torch.no_grad():
            e = clip_model.encode_image(px).float()
            return e / e.norm(p=2, dim=-1, keepdim=True)

    imgs = np.load(args.baseline_images, mmap_mode="r")
    idx = (np.load(args.baseline_split)["test_idx"] if Path(args.baseline_split).exists()
           else np.arange(imgs.shape[0]))[: args.baseline_n]
    E = []
    for i in range(0, len(idx), 16):
        chunk = [np.asarray(imgs[j]).astype("uint8") for j in idx[i:i + 16]]
        E.append(emb(chunk))
    E = torch.cat(E, dim=0)                                            # [Nnat, 768] normed
    print(f"[calib] {E.shape[0]} natural images encoded\n")

    for apath in args.anchors:
        s = torch.load(apath, map_location="cpu")
        name = Path(apath).stem
        img_set = torch.as_tensor(s["img_set"], dtype=torch.float32, device=device)
        img_set = img_set / img_set.norm(p=2, dim=-1, keepdim=True)
        cos = E @ img_set.t()                                          # [Nnat, N]
        img_mean = cos.mean(dim=1).cpu().numpy()                       # per-image mean over anchors
        img_max = cos.max(dim=1).values.cpu().numpy()                  # per-image max over anchors
        print(f"=== {name}  (img_set {tuple(img_set.shape)}) ===")
        print(f"  IMAGE cos mean-over-anchors: {pct(img_mean)}")
        print(f"  IMAGE cos max-over-anchors : {pct(img_max)}")
        for c in args.c_img:
            print(f"    c_img={c}: {100*np.mean(img_mean < c):.0f}% of naturals have mean-cos < c "
                  f"(so hinge would keep pushing past {100*np.mean(img_mean < c):.0f}% of natural imgs)")
        if s.get("text_set") is not None:
            text_set = torch.as_tensor(s["text_set"], dtype=torch.float32, device=device)
            text_set = text_set / text_set.norm(p=2, dim=-1, keepdim=True)
            tcos = E @ text_set.t()
            t_mean = tcos.mean(dim=1).cpu().numpy()
            print(f"  TEXT cos mean-over-anchors : {pct(t_mean)}")
            print(f"    c_txt={args.c_txt}: {100*np.mean(t_mean < args.c_txt):.0f}% of naturals below "
                  f"(text floor; expect most naturals already near/below -> term near-inert)")
        print()
    print("CALIB_DONE")


if __name__ == "__main__":
    main()
