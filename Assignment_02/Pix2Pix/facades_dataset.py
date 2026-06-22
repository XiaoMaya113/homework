from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FacadesDataset(Dataset):
    def __init__(self, root, split="train", image_size=256):
        self.root = Path(root) / split
        self.paths = sorted(list(self.root.glob("*.jpg")) + list(self.root.glob("*.png")))
        if not self.paths:
            raise FileNotFoundError(f"No images found in {self.root}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        w, h = image.size
        photo = image.crop((0, 0, w // 2, h))
        label = image.crop((w // 2, 0, w, h))
        source = self.transform(label)
        target = self.transform(photo)
        return {"source": source, "target": target, "path": str(self.paths[index])}
