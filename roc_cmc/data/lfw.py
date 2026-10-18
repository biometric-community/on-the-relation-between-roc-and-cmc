"""LFW loader for face stand-in experiment (FRGC-V2 blocked)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def resolve_lfw_root(cfg: dict) -> Path:
    root = Path(cfg["data"]["lfw_root"])
    if not root.is_absolute():
        proj = Path(__file__).resolve().parents[2]
        cand = (proj / root).resolve()
        root = cand if cand.exists() else root
    for sub in (root / "lfw", root / "extracted" / "lfw", root):
        if sub.is_dir() and any(sub.iterdir()):
            # identity folders?
            if any(p.is_dir() for p in sub.iterdir()):
                return sub
    raise FileNotFoundError(f"LFW not found under {cfg['data']['lfw_root']}")


def list_identities(root: Path, min_images: int = 20) -> Dict[str, List[Path]]:
    ids = {}
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        imgs = sorted(list(p.glob("*.jpg")) + list(p.glob("*.png")))
        if len(imgs) >= min_images:
            ids[p.name] = imgs
    return ids


class LFWIdentityDataset(Dataset):
    def __init__(self, id_map: Dict[str, List[Path]], transform=None, max_per_id: int | None = None):
        self.samples: List[Tuple[Path, int]] = []
        self.names = sorted(id_map.keys())
        self.name2lab = {n: i for i, n in enumerate(self.names)}
        for name, paths in id_map.items():
            paths = paths if max_per_id is None else paths[:max_per_id]
            lab = self.name2lab[name]
            for p in paths:
                self.samples.append((p, lab))
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, lab = self.samples[i]
        img = Image.open(path).convert("RGB")
        return {"image": self.transform(img), "label": lab, "path": str(path)}
