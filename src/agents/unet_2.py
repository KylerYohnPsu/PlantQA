import time
import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- model
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Same layout as the Keras UNet. base=32 -> 32..512 filters, base=64 -> 64..1024."""

    def __init__(self, in_ch=3, base=32):
        super().__init__()
        f = [base * 2 ** i for i in range(5)]
        self.enc = nn.ModuleList(
            [ConvBlock(in_ch, f[0])] + [ConvBlock(f[i], f[i + 1]) for i in range(3)]
        )
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(f[3], f[4])
        self.up = nn.ModuleList(
            [nn.ConvTranspose2d(f[i + 1], f[i], 2, stride=2) for i in reversed(range(4))]
        )
        self.dec = nn.ModuleList([ConvBlock(f[i] * 2, f[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(f[0], 1, 1)  # outputs logits (no sigmoid)

    def forward(self, x):
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)


# ---------------------------------------------------------------- helpers
def to_device(x, y):
    """DataLoader gives (B, H, W, C); PyTorch convs want (B, C, H, W)."""
    x = x.to(DEVICE, non_blocking=True).permute(0, 3, 1, 2)
    x = x.contiguous(memory_format=torch.channels_last)
    y = y.to(DEVICE, non_blocking=True).permute(0, 3, 1, 2).contiguous()
    return x, y


def loss_fn(logits, y):
    """Focal loss + Dice loss, computed in float32 for stability."""
    logits = logits.float()
    focal = sigmoid_focal_loss(logits, y, alpha=0.25, gamma=2.0, reduction="mean")
    p = torch.sigmoid(logits)
    inter = (p * y).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
    dice = 1 - ((2 * inter + 1) / (denom + 1)).mean()
    return focal + dice


# ---------------------------------------------------------------- benchmark
def benchmark(model, loader, steps=10):
    """Times training steps on one reused batch. Should be well under 1s/step on a 5070."""
    model.to(DEVICE).train()
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    x, y = to_device(*next(iter(loader)))

    for i in range(steps + 2):
        if i == 2:  # first steps include cuDNN setup
            torch.cuda.synchronize()
            start = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    per_step = (time.time() - start) / steps
    print(f"{per_step:.3f}s per step  ->  ~{per_step * len(loader) / 60:.1f} min per epoch")
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


# ---------------------------------------------------------------- train / eval
@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    model.eval()
    total_loss, inter, union = 0.0, 0.0, 0.0
    for x, y in loader:
        x, y = to_device(x, y)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        total_loss += loss_fn(logits, y).item()
        pred = (torch.sigmoid(logits.float()) > threshold).float()
        inter += (pred * y).sum().item()
        union += ((pred + y) > 0).float().sum().item()
    iou = inter / union if union > 0 else 1.0
    return {"loss": total_loss / len(loader), "iou": iou}


def fit(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=4,
        save_path="best_unet.pt", log_every=50):
    model.to(DEVICE)
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda")
    best_iou, bad_epochs = -1.0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        start, running = time.time(), 0.0
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = to_device(x, y)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            if step % log_every == 0:
                elapsed = time.time() - start
                print(f"  epoch {epoch} step {step}/{len(train_loader)}  "
                      f"loss {running / step:.4f}  {elapsed / step:.3f}s/step")

        val = evaluate(model, val_loader)
        print(f"Epoch {epoch}: train_loss {running / len(train_loader):.4f}  "
              f"val_loss {val['loss']:.4f}  val_iou {val['iou']:.4f}  "
              f"({(time.time() - start) / 60:.1f} min)")

        if val["iou"] > best_iou:
            best_iou, bad_epochs = val["iou"], 0
            torch.save(model.state_dict(), save_path)
            print(f"  saved new best model (val_iou {best_iou:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping. Best val_iou: {best_iou:.4f}")
                break

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    return model