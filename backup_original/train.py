from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from synthetic_data import make_synth_sample
from model import UNetSmall, BCEDiceLoss

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

class SynthCurveDataset(Dataset):
    def __init__(self, length=20000, train_size=512):
        self.length = length
        self.train_size = train_size
        self.img_tf = transforms.Compose([
            transforms.Resize((train_size, train_size)),
            transforms.ToTensor(),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((train_size, train_size), interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def __len__(self): return self.length

    def __getitem__(self, idx):
        pil_img, mask_bin = make_synth_sample(seed=None)
        # resize image
        img = self.img_tf(pil_img)  # [3,H,W] float 0..1

        # mask -> PIL -> resize nearest -> tensor
        mask_pil = Image.fromarray((mask_bin*255).astype(np.uint8))
        mask_pil = self.mask_tf(mask_pil)
        mask = torch.from_numpy((np.asarray(mask_pil) > 127).astype(np.uint8)).unsqueeze(0)  # [1,H,W]

        return img, mask

def train_unet(
    device="cuda" if torch.cuda.is_available() else "cpu",
    train_size=512,
    steps=30000,
    batch_size=6,
    lr=2e-4,
    save_path="curve_unet.pt"
):
    ds = SynthCurveDataset(length=steps*batch_size, train_size=train_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = UNetSmall(in_ch=3, base=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = BCEDiceLoss(bce_weight=0.4)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda")))

    model.train()
    pbar = tqdm(total=steps)
    it = iter(dl)

    for step in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl)
            x, y = next(it)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.startswith("cuda"))):
            logits = model(x)
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if (step+1) % 500 == 0:
            torch.save(model.state_dict(), save_path)

        pbar.set_description(f"loss={loss.item():.4f}")
        pbar.update(1)

    torch.save(model.state_dict(), save_path)
    return model
