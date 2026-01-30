# fixed_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

class FixedSynthDataset(Dataset):
    """
    Pre-generates N synthetic (image, mask) pairs in memory with fixed seeds.
    Used for overfit/sanity tests so we can SEE if the model can learn at all.
    """
    def __init__(self, make_synth_sample, n=32, W=512, H=512, seed=0):
        self.samples = []
        rng = np.random.RandomState(seed)
        for _ in range(n):
            s = int(rng.randint(0, 10_000_000))
            img, mask = make_synth_sample(W=W, H=H, seed=s)  # img: PIL, mask: 0/1
            self.samples.append((img, mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, mask = self.samples[idx]
        x = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0   # (3,H,W)
        y = torch.from_numpy(mask[None, ...]).float()                             # (1,H,W)
        return x, y
