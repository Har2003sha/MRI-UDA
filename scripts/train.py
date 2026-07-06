"""
Train two models for comparison, exactly as a B.Tech project report would
present it:

  1. BASELINE  : U-Net trained only on SOURCE domain (Scanner A) labeled data.
                 No domain adaptation. Expected to perform well on Scanner A
                 but degrade on Scanner B due to domain shift.

  2. UDA (DANN): Same U-Net architecture + gradient-reversal domain
                 classifier. Trained on SOURCE labeled data (segmentation
                 loss) + SOURCE/TARGET unlabeled data (domain loss).
                 Expected to close the performance gap on Scanner B without
                 ever seeing Scanner-B masks during training.

Run:  python scripts/train.py
Outputs:
  models/baseline_unet.pth
  models/uda_unet.pth
  models/metrics.json   (used by the Flask evaluation dashboard)
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.uda_unet import UDAUNet, dice_coefficient, iou_score
from data.synthetic_mri import generate_domain_batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def to_tensor(imgs):
    return torch.from_numpy(imgs).unsqueeze(1).float()  # (N,1,H,W)


def dice_loss(logits, target):
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + 1e-6) / (union + 1e-6)).mean()


def bce_dice_loss(logits, target):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    return bce + dice_loss(logits, target)


@torch.no_grad()
def evaluate(model, imgs, masks, batch_size=16):
    model.eval()
    dices, ious = [], []
    for i in range(0, len(imgs), batch_size):
        x = to_tensor(imgs[i:i + batch_size]).to(DEVICE)
        seg_logits, _ = model(x, return_domain=False)
        preds = (torch.sigmoid(seg_logits) > 0.5).float().cpu().numpy()[:, 0]
        for p, m in zip(preds, masks[i:i + batch_size]):
            dices.append(dice_coefficient(p, m))
            ious.append(iou_score(p, m))
    return float(np.mean(dices)), float(np.mean(ious))


def make_datasets(n_train_source=320, n_train_target=320, n_eval=80):
    src_imgs, src_masks = generate_domain_batch(n_train_source, domain="A", seed_offset=0)
    tgt_imgs, _ = generate_domain_batch(n_train_target, domain="B", seed_offset=100000)  # unlabeled

    src_eval_imgs, src_eval_masks = generate_domain_batch(n_eval, domain="A", seed_offset=200000)
    tgt_eval_imgs, tgt_eval_masks = generate_domain_batch(n_eval, domain="B", seed_offset=300000)  # labels held out, eval only

    return (src_imgs, src_masks, tgt_imgs,
            src_eval_imgs, src_eval_masks, tgt_eval_imgs, tgt_eval_masks)


def train_baseline(src_imgs, src_masks, epochs=12, batch_size=16, lr=1e-3):
    model = UDAUNet().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    n = len(src_imgs)
    history = []
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        ep_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x = to_tensor(src_imgs[idx]).to(DEVICE)
            y = to_tensor(src_masks[idx]).to(DEVICE)
            opt.zero_grad()
            seg_logits, _ = model(x, return_domain=False)
            loss = bce_dice_loss(seg_logits, y)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        history.append(ep_loss / n)
        print(f"[baseline] epoch {ep+1}/{epochs} loss={history[-1]:.4f}")
    return model, history


def train_uda(src_imgs, src_masks, tgt_imgs, epochs=12, batch_size=16, lr=1e-3):
    model = UDAUNet().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    domain_criterion = nn.CrossEntropyLoss()
    n = min(len(src_imgs), len(tgt_imgs))
    history = []
    total_steps = epochs * (n // batch_size)
    step = 0
    for ep in range(epochs):
        model.train()
        perm_s = np.random.permutation(len(src_imgs))
        perm_t = np.random.permutation(len(tgt_imgs))
        ep_seg_loss, ep_dom_loss = 0.0, 0.0
        for i in range(0, n, batch_size):
            # progressively ramp up adaptation strength (standard DANN schedule)
            p = step / max(1, total_steps)
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            step += 1

            idx_s = perm_s[i:i + batch_size]
            idx_t = perm_t[i:i + batch_size]
            xs = to_tensor(src_imgs[idx_s]).to(DEVICE)
            ys = to_tensor(src_masks[idx_s]).to(DEVICE)
            xt = to_tensor(tgt_imgs[idx_t]).to(DEVICE)

            opt.zero_grad()
            # source: segmentation + domain(label=0)
            seg_logits, dom_logits_s = model(xs, alpha=alpha)
            seg_loss = bce_dice_loss(seg_logits, ys)
            dom_labels_s = torch.zeros(xs.size(0), dtype=torch.long, device=DEVICE)
            dom_loss_s = domain_criterion(dom_logits_s, dom_labels_s)

            # target: domain(label=1) only, no mask available (unsupervised)
            _, dom_logits_t = model(xt, alpha=alpha)
            dom_labels_t = torch.ones(xt.size(0), dtype=torch.long, device=DEVICE)
            dom_loss_t = domain_criterion(dom_logits_t, dom_labels_t)

            domain_loss = dom_loss_s + dom_loss_t
            loss = seg_loss + 0.5 * domain_loss
            loss.backward()
            opt.step()

            ep_seg_loss += seg_loss.item() * len(idx_s)
            ep_dom_loss += domain_loss.item() * len(idx_s)
        history.append({"seg_loss": ep_seg_loss / n, "domain_loss": ep_dom_loss / n, "alpha": float(alpha)})
        print(f"[uda] epoch {ep+1}/{epochs} seg_loss={ep_seg_loss/n:.4f} "
              f"domain_loss={ep_dom_loss/n:.4f} alpha={alpha:.2f}")
    return model, history


def main():
    t0 = time.time()
    print("Generating synthetic multi-scanner dataset ...")
    (src_imgs, src_masks, tgt_imgs,
     src_eval_imgs, src_eval_masks,
     tgt_eval_imgs, tgt_eval_masks) = make_datasets()

    print("\n=== Training BASELINE (source-only, no domain adaptation) ===")
    baseline_model, baseline_history = train_baseline(src_imgs, src_masks, epochs=8)

    print("\n=== Training UDA model (DANN: gradient-reversal domain adaptation) ===")
    uda_model, uda_history = train_uda(src_imgs, src_masks, tgt_imgs, epochs=8)

    print("\nEvaluating ...")
    base_src_dice, base_src_iou = evaluate(baseline_model, src_eval_imgs, src_eval_masks)
    base_tgt_dice, base_tgt_iou = evaluate(baseline_model, tgt_eval_imgs, tgt_eval_masks)
    uda_src_dice, uda_src_iou = evaluate(uda_model, src_eval_imgs, src_eval_masks)
    uda_tgt_dice, uda_tgt_iou = evaluate(uda_model, tgt_eval_imgs, tgt_eval_masks)

    metrics = {
        "baseline": {
            "source_dice": base_src_dice, "source_iou": base_src_iou,
            "target_dice": base_tgt_dice, "target_iou": base_tgt_iou,
            "loss_history": baseline_history,
        },
        "uda": {
            "source_dice": uda_src_dice, "source_iou": uda_src_iou,
            "target_dice": uda_tgt_dice, "target_iou": uda_tgt_iou,
            "loss_history": uda_history,
        },
        "target_dice_improvement": uda_tgt_dice - base_tgt_dice,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_seconds": round(time.time() - t0, 1),
    }

    torch.save(baseline_model.state_dict(), os.path.join(MODELS_DIR, "baseline_unet.pth"))
    torch.save(uda_model.state_dict(), os.path.join(MODELS_DIR, "uda_unet.pth"))
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n================ RESULTS ================")
    print(f"Baseline  -> Source Dice: {base_src_dice:.3f} | Target Dice: {base_tgt_dice:.3f}")
    print(f"UDA(DANN) -> Source Dice: {uda_src_dice:.3f} | Target Dice: {uda_tgt_dice:.3f}")
    print(f"Target-domain Dice improvement from UDA: {metrics['target_dice_improvement']:+.3f}")
    print(f"Total training time: {metrics['train_seconds']}s")
    print("Saved: models/baseline_unet.pth, models/uda_unet.pth, models/metrics.json")


if __name__ == "__main__":
    main()
