"""
Sentinel-2 Litter/Debris Detection — Training + QC Script
Optimized for MacBook Air
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for overnight runs

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics import JaccardIndex, F1Score, Precision, Recall
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import segmentation_models_pytorch as smp

# ── Optional: comment out if you don't have wandb yet ──────────────────────
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed — logging to local files only.")
    print("Install with: pip install wandb")

# ───────────────────────────────────────────────────────────────────────────
# 0. CONFIG — edit this section before running
# ───────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Paths
    "data_dir":        "./data/marida/patches",   # folder of .npy patch files
    "label_dir":       "./data/marida/labels",    # folder of .npy label files
    "output_dir":      "./outputs",               # checkpoints + plots saved here

    # Model
    "model":           "unet-resnet34",           # lightweight, good for Mac
    "in_channels":     7,                         # 6 S2 bands + FDI index
    "num_classes":     3,                         # 0=background, 1=litter, 2=other

    # Training
    "epochs":          50,
    "batch_size":      8,                         # safe for 16GB unified memory
    "patch_size":      128,                       # use 256 if memory allows
    "lr":              1e-4,
    "lr_patience":     5,                         # reduce LR after N stagnant epochs
    "early_stop":      15,                        # stop if val F1 doesn't improve
    "accum_steps":     4,                         # gradient accumulation (effective bs=32)
    "val_split":       0.15,
    "test_split":      0.10,
    "num_workers":     2,                         # sweet spot for Mac

    # Class weights — litter is rare so weight it heavily
    "class_weights":   [0.1, 10.0, 1.0],

    # QC
    "vis_every":       5,                         # visualise preds every N epochs
    "wandb_project":   "sentinel2-litter-chicago",
    "wandb_enabled":   True,                      # set False to skip wandb

    # Reproducibility
    "seed":            42,
}

# ───────────────────────────────────────────────────────────────────────────
# 1. SETUP
# ───────────────────────────────────────────────────────────────────────────
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # fallback unsupported MPS ops to CPU

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
os.makedirs(CONFIG["output_dir"], exist_ok=True)

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple Silicon MPS GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("⚠️  Using CPU — training will be slow")


# ───────────────────────────────────────────────────────────────────────────
# 2. DATASET
# ───────────────────────────────────────────────────────────────────────────
def compute_fdi(bands):
    """
    Floating Debris Index from Sentinel-2 bands.
    bands shape: (C, H, W) — expects B6=idx4, B8=idx3, B11=idx5 (0-indexed).
    Adjust indices to match your band ordering.
    """
    b6  = bands[4]   # Red-edge (~740nm)
    b8  = bands[3]   # NIR (~833nm)
    b11 = bands[5]   # SWIR (~1610nm)
    fdi = b8 - (b6 + (b11 - b6) * ((833 - 740) / (1610 - 740)) * 10)
    return fdi[np.newaxis, :]  # (1, H, W)


class LitterDataset(Dataset):
    """
    Expects:
      data_dir/  — .npy files of shape (C, H, W), float32, range ~[0, 1]
      label_dir/ — .npy files of shape (H, W), int64, values {0, 1, 2}
    Filenames must match between the two folders.
    """
    def __init__(self, data_dir, label_dir, augment=False):
        self.data_dir  = data_dir
        self.label_dir = label_dir
        self.augment   = augment
        self.files     = sorted(os.listdir(data_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname  = self.files[idx]
        image  = np.load(os.path.join(self.data_dir,  fname)).astype(np.float32)
        label  = np.load(os.path.join(self.label_dir, fname)).astype(np.int64)

        # Append FDI as extra channel
        fdi   = compute_fdi(image)
        image = np.concatenate([image, fdi], axis=0)  # (7, H, W)

        # Augmentation (random flips)
        if self.augment:
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2).copy()
                label = np.flip(label, axis=1).copy()
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()
                label = np.flip(label, axis=0).copy()
            if np.random.rand() > 0.5:
                k = np.random.randint(1, 4)
                image = np.rot90(image, k, axes=(1, 2)).copy()
                label = np.rot90(label, k, axes=(0, 1)).copy()

        return torch.from_numpy(image), torch.from_numpy(label)


def build_dataloaders(cfg):
    full_ds = LitterDataset(cfg["data_dir"], cfg["label_dir"], augment=False)
    n       = len(full_ds)
    n_test  = int(n * cfg["test_split"])
    n_val   = int(n * cfg["val_split"])
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg["seed"])
    )
    # Enable augmentation only on train split
    train_ds.dataset.augment = True

    print(f"Dataset splits — train: {n_train} | val: {n_val} | test: {n_test}")

    kwargs = dict(
        batch_size  = cfg["batch_size"],
        num_workers = cfg["num_workers"],
        pin_memory  = False,   # pin_memory=True can cause MPS issues
    )
    return (
        DataLoader(train_ds, shuffle=True,  **kwargs),
        DataLoader(val_ds,   shuffle=False, **kwargs),
        DataLoader(test_ds,  shuffle=False, **kwargs),
    )


# ───────────────────────────────────────────────────────────────────────────
# 3. MODEL
# ───────────────────────────────────────────────────────────────────────────
def build_model(cfg):
    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = None,              # no RGB pretraining — custom bands
        in_channels     = cfg["in_channels"],
        classes         = cfg["num_classes"],
        activation      = None,
    )
    return model.to(device)


# ───────────────────────────────────────────────────────────────────────────
# 4. LOSS
# ───────────────────────────────────────────────────────────────────────────
def build_loss(cfg):
    weights = torch.tensor(cfg["class_weights"], dtype=torch.float32).to(device)
    ce_loss   = nn.CrossEntropyLoss(weight=weights)
    dice_loss = smp.losses.DiceLoss(mode="multiclass")

    def combined_loss(pred, target):
        return 0.5 * ce_loss(pred, target) + 0.5 * dice_loss(pred, target)

    return combined_loss


# ───────────────────────────────────────────────────────────────────────────
# 5. METRICS
# ───────────────────────────────────────────────────────────────────────────
class LitterMetrics:
    def __init__(self, num_classes, device):
        self.iou       = JaccardIndex(task="multiclass", num_classes=num_classes).to(device)
        self.f1        = F1Score(task="multiclass",      num_classes=num_classes, average="none").to(device)
        self.precision = Precision(task="multiclass",    num_classes=num_classes, average="none").to(device)
        self.recall    = Recall(task="multiclass",       num_classes=num_classes, average="none").to(device)

    def update(self, preds, targets):
        preds_cls = torch.argmax(preds, dim=1)
        self.iou.update(preds_cls, targets)
        self.f1.update(preds_cls, targets)
        self.precision.update(preds_cls, targets)
        self.recall.update(preds_cls, targets)

    def compute(self):
        f1_per_class  = self.f1.compute()
        pr_per_class  = self.precision.compute()
        re_per_class  = self.recall.compute()
        return {
            "mean_iou":         self.iou.compute().item(),
            "debris_f1":        f1_per_class[1].item(),    # class 1 = litter
            "debris_precision": pr_per_class[1].item(),
            "debris_recall":    re_per_class[1].item(),
            "f1_per_class":     f1_per_class.tolist(),
        }

    def reset(self):
        self.iou.reset()
        self.f1.reset()
        self.precision.reset()
        self.recall.reset()


# ───────────────────────────────────────────────────────────────────────────
# 6. CHECKPOINTING
# ───────────────────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, metrics, path):
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "metrics":    metrics,
    }, path)
    print(f"  💾 Checkpoint saved → {path}")


def load_checkpoint(model, optimizer, path):
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        print(f"  ▶️  Resumed from epoch {ckpt['epoch']}")
        return ckpt["epoch"], ckpt["metrics"]
    return 0, {}


# ───────────────────────────────────────────────────────────────────────────
# 7. VISUALISATION
# ───────────────────────────────────────────────────────────────────────────
def visualize_predictions(model, loader, device, epoch, out_dir, n=4):
    model.eval()
    fig, axes = plt.subplots(n, 4, figsize=(18, n * 4.5))
    col_titles = ["RGB Composite", "False Color (NIR)", "Ground Truth", "Prediction"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12, fontweight="bold")

    cmap = plt.get_cmap("tab10")

    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            if i >= n:
                break
            preds = torch.argmax(model(images.to(device)), dim=1)
            img   = images[0].cpu().numpy()
            true  = masks[0].cpu().numpy()
            pred  = preds[0].cpu().numpy()

            # RGB (bands 2,1,0 → R,G,B)
            rgb = np.stack([img[2], img[1], img[0]], axis=-1)
            rgb = np.clip(rgb / rgb.max(), 0, 1)
            axes[i, 0].imshow(rgb)

            # False color NIR (bands 3,2,1)
            fc = np.stack([img[3], img[2], img[1]], axis=-1)
            fc = np.clip(fc / fc.max(), 0, 1)
            axes[i, 1].imshow(fc)

            # Ground truth and prediction
            axes[i, 2].imshow(true, cmap="tab10", vmin=0, vmax=2, interpolation="nearest")
            axes[i, 3].imshow(pred, cmap="tab10", vmin=0, vmax=2, interpolation="nearest")

            for ax in axes[i]:
                ax.axis("off")

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=cmap(0), label="Background"),
        plt.Rectangle((0, 0), 1, 1, color=cmap(1), label="Litter ⚠️"),
        plt.Rectangle((0, 0), 1, 1, color=cmap(2), label="Vegetation/Water"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11, framealpha=0.9)
    plt.suptitle(f"Predictions — Epoch {epoch}", fontsize=14, y=1.01)
    plt.tight_layout()

    path = os.path.join(out_dir, f"predictions_epoch_{epoch:03d}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  🖼  Predictions saved → {path}")
    return path


def plot_confusion_matrix(model, loader, device, epoch, out_dir):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, masks in loader:
            preds = torch.argmax(model(images.to(device)), dim=1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.numpy().flatten())

    cm   = confusion_matrix(all_targets, all_preds, normalize="true")
    disp = ConfusionMatrixDisplay(cm, display_labels=["Background", "Litter", "Other"])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix (normalised) — Epoch {epoch}")

    path = os.path.join(out_dir, f"confusion_matrix_epoch_{epoch:03d}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  📊 Confusion matrix saved → {path}")
    return path


def plot_learning_curves(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train", color="royalblue")
    axes[0].plot(epochs, history["val_loss"],   label="Val",   color="tomato")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(epochs, history["debris_f1"],        color="darkorange", label="Debris F1")
    axes[1].plot(epochs, history["debris_precision"],  color="mediumseagreen", linestyle="--", label="Precision")
    axes[1].plot(epochs, history["debris_recall"],     color="mediumpurple",   linestyle=":", label="Recall")
    axes[1].set_title("Debris Detection Quality")
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, history["mean_iou"], color="steelblue", label="Mean IoU")
    axes[2].set_title("Mean IoU"); axes[2].set_ylim(0, 1)
    axes[2].legend(); axes[2].set_xlabel("Epoch")

    plt.suptitle("Training Learning Curves", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, "learning_curves.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  📈 Learning curves saved → {path}")


# ───────────────────────────────────────────────────────────────────────────
# 8. SANITY CHECKS (run before full training)
# ───────────────────────────────────────────────────────────────────────────
def run_sanity_checks(model, loader, criterion, cfg):
    print("\n" + "="*60)
    print("🔍 PRE-TRAINING SANITY CHECKS")
    print("="*60)

    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)

    # 1. Input range
    print(f"\n1. Input tensor range: [{images.min():.4f}, {images.max():.4f}]")
    assert images.max() <= 5.0, "⚠️  Values seem very large — check normalisation!"
    print("   ✅ Input range looks reasonable")

    # 2. Class distribution
    all_labels = []
    for _, m in loader:
        all_labels.extend(m.numpy().flatten())
    unique, counts = np.unique(all_labels, return_counts=True)
    print("\n2. Class distribution in dataset:")
    cls_names = {0: "Background", 1: "Litter ⚠️", 2: "Other"}
    for cls, cnt in zip(unique, counts):
        pct = cnt / len(all_labels) * 100
        print(f"   Class {cls} ({cls_names.get(cls, '?')}): {cnt:,} px  ({pct:.2f}%)")
    if 1 not in unique:
        print("   ❌ NO LITTER CLASS FOUND — check your labels!")
    else:
        litter_pct = counts[list(unique).index(1)] / len(all_labels) * 100
        if litter_pct < 0.5:
            print(f"   ⚠️  Litter is only {litter_pct:.2f}% of pixels — high class imbalance, weights are important")

    # 3. Forward pass shape
    model.train()
    with torch.no_grad():
        out = model(images)
    print(f"\n3. Output shape: {out.shape}  (expected: [{cfg['batch_size']}, {cfg['num_classes']}, H, W])")
    assert out.shape[1] == cfg["num_classes"], "❌ Output channels don't match num_classes!"
    print("   ✅ Output shape correct")

    # 4. Single-batch overfit test
    print("\n4. Single-batch overfit test (loss should drop fast):")
    test_model  = build_model(cfg)
    test_optim  = torch.optim.Adam(test_model.parameters(), lr=1e-3)
    start_loss  = None
    for step in range(30):
        pred = test_model(images)
        loss = criterion(pred, masks)
        test_optim.zero_grad()
        loss.backward()
        test_optim.step()
        if step == 0:
            start_loss = loss.item()
        if step % 9 == 0:
            print(f"   Step {step+1:2d}: loss = {loss.item():.4f}")
    if loss.item() < start_loss * 0.5:
        print("   ✅ Model can overfit a single batch — architecture OK")
    else:
        print("   ⚠️  Loss didn't drop much — check learning rate or loss function")
    del test_model, test_optim

    print("\n" + "="*60 + "\n")


# ───────────────────────────────────────────────────────────────────────────
# 9. TRAINING LOOP
# ───────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, metrics, cfg, epoch):
    model.train()
    metrics.reset()
    total_loss  = 0.0
    optimizer.zero_grad()

    for step, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        preds  = model(images)
        loss   = criterion(preds, masks) / cfg["accum_steps"]
        loss.backward()

        if (step + 1) % cfg["accum_steps"] == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * cfg["accum_steps"]
        metrics.update(preds.detach(), masks)

    return total_loss / len(loader), metrics.compute()


@torch.no_grad()
def validate(model, loader, criterion, metrics):
    model.eval()
    metrics.reset()
    total_loss = 0.0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        preds       = model(images)
        loss        = criterion(preds, masks)
        total_loss += loss.item()
        metrics.update(preds, masks)

    return total_loss / len(loader), metrics.compute()


# ───────────────────────────────────────────────────────────────────────────
# 10. MAIN
# ───────────────────────────────────────────────────────────────────────────
def main():
    cfg = CONFIG

    # W&B init
    use_wandb = cfg["wandb_enabled"] and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=cfg["wandb_project"], config=cfg)
        print("📡 W&B logging enabled — watch your run at wandb.ai")
    else:
        print("📁 Logging locally only (no W&B)")

    # Build everything
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    model     = build_model(cfg)
    criterion = build_loss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=cfg["lr_patience"], verbose=True
    )
    train_metrics = LitterMetrics(cfg["num_classes"], device)
    val_metrics   = LitterMetrics(cfg["num_classes"], device)

    ckpt_path  = os.path.join(cfg["output_dir"], "best_model.pth")
    last_path  = os.path.join(cfg["output_dir"], "last_model.pth")
    start_epoch, _ = load_checkpoint(model, optimizer, last_path)

    # Sanity checks before first run
    if start_epoch == 0:
        run_sanity_checks(model, train_loader, criterion, cfg)

    # History
    history = {k: [] for k in [
        "train_loss", "val_loss", "debris_f1",
        "debris_precision", "debris_recall", "mean_iou"
    ]}

    best_f1      = 0.0
    no_improve   = 0
    total_start  = time.time()

    print("🚀 Starting training — plug in your Mac!\n")

    for epoch in range(start_epoch + 1, cfg["epochs"] + 1):
        t0 = time.time()

        train_loss, train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, train_metrics, cfg, epoch
        )
        val_loss, val_m = validate(model, val_loader, criterion, val_metrics)
        scheduler.step(val_m["debris_f1"])

        elapsed = time.time() - t0
        remaining_epochs = cfg["epochs"] - epoch
        eta_mins = remaining_epochs * elapsed / 60

        # ── Console summary ────────────────────────────────────────────
        print(
            f"Epoch {epoch:3d}/{cfg['epochs']} | "
            f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
            f"Debris F1: {val_m['debris_f1']:.4f} | "
            f"IoU: {val_m['mean_iou']:.4f} | "
            f"{elapsed:.0f}s/epoch | ETA: {eta_mins:.0f}min"
        )

        # ── History ────────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["debris_f1"].append(val_m["debris_f1"])
        history["debris_precision"].append(val_m["debris_precision"])
        history["debris_recall"].append(val_m["debris_recall"])
        history["mean_iou"].append(val_m["mean_iou"])

        # ── W&B logging ────────────────────────────────────────────────
        if use_wandb:
            log_dict = {
                "epoch":              epoch,
                "train/loss":         train_loss,
                "val/loss":           val_loss,
                "val/debris_f1":      val_m["debris_f1"],
                "val/debris_precision": val_m["debris_precision"],
                "val/debris_recall":  val_m["debris_recall"],
                "val/mean_iou":       val_m["mean_iou"],
                "lr":                 optimizer.param_groups[0]["lr"],
            }

        # ── Visualisations ─────────────────────────────────────────────
        if epoch % cfg["vis_every"] == 0 or epoch == 1:
            pred_path = visualize_predictions(
                model, val_loader, device, epoch, cfg["output_dir"]
            )
            cm_path = plot_confusion_matrix(
                model, val_loader, device, epoch, cfg["output_dir"]
            )
            if use_wandb:
                log_dict["predictions"]      = wandb.Image(pred_path)
                log_dict["confusion_matrix"] = wandb.Image(cm_path)

        if use_wandb:
            wandb.log(log_dict)

        # ── Checkpointing ──────────────────────────────────────────────
        save_checkpoint(model, optimizer, epoch, val_m, last_path)

        if val_m["debris_f1"] > best_f1:
            best_f1   = val_m["debris_f1"]
            no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_m, ckpt_path)
            print(f"  ⭐ New best Debris F1: {best_f1:.4f}")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{cfg['early_stop']} epochs")

        # ── Early stopping ─────────────────────────────────────────────
        if no_improve >= cfg["early_stop"]:
            print(f"\n⏹  Early stopping at epoch {epoch} — best F1: {best_f1:.4f}")
            break

    # ── Final plots ────────────────────────────────────────────────────────
    plot_learning_curves(history, cfg["output_dir"])
    total_mins = (time.time() - total_start) / 60
    print(f"\n✅ Training complete in {total_mins:.1f} minutes")
    print(f"   Best Debris F1: {best_f1:.4f}")
    print(f"   Best model saved → {ckpt_path}")

    # ── Final test evaluation ──────────────────────────────────────────────
    print("\n🧪 Final evaluation on held-out test set...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_m = validate(model, test_loader, criterion, val_metrics)

    print("\n" + "="*50)
    print("TEST SET RESULTS")
    print("="*50)
    print(f"  Loss:              {test_loss:.4f}")
    print(f"  Mean IoU:          {test_m['mean_iou']:.4f}")
    print(f"  Debris F1:         {test_m['debris_f1']:.4f}")
    print(f"  Debris Precision:  {test_m['debris_precision']:.4f}")
    print(f"  Debris Recall:     {test_m['debris_recall']:.4f}")
    print("="*50)

    if use_wandb:
        wandb.log({
            "test/loss":              test_loss,
            "test/mean_iou":          test_m["mean_iou"],
            "test/debris_f1":         test_m["debris_f1"],
            "test/debris_precision":  test_m["debris_precision"],
            "test/debris_recall":     test_m["debris_recall"],
        })
        wandb.finish()


if __name__ == "__main__":
    main()
