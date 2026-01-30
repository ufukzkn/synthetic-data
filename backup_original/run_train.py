import torch
from torch.utils.data import DataLoader
from model import UNetSmall, BCEDiceLoss
from synthetic_data import make_synth_sample
from fixed_dataset import FixedSynthDataset

def overfit_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    ds = FixedSynthDataset(make_synth_sample, n=32, W=512, H=512, seed=0)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)

    model = UNetSmall(in_ch=3, base=32).to(device)
    loss_fn = BCEDiceLoss(bce_weight=0.4)  # şimdilik bu kalsın
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    model.train()
    for step in range(1, 801):  # 800 step yeter
        x, y = next(iter(dl))
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(f"[overfit] step={step} loss={loss.item():.4f}")

        if step % 200 == 0:
            torch.save(model.state_dict(), "curve_unet_overfit.pt")

    print("Saved curve_unet_overfit.pt")

if __name__ == "__main__":
    overfit_test()
