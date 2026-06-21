import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--views", default="0,12,25,37,49")
    parser.add_argument("--output_dir", default="outputs/overlays")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points2d = np.load(data_dir / "points2d.npz")
    sample = points2d[points2d.files[0]]
    colors = cv2.applyColorMap(np.linspace(0, 255, len(sample), dtype=np.uint8), cv2.COLORMAP_TURBO)[:, 0, :]

    for view_id in [int(v.strip()) for v in args.views.split(",") if v.strip()]:
        key = f"view_{view_id:03d}"
        image = cv2.imread(str(data_dir / "images" / f"{key}.png"))
        if image is None:
            raise FileNotFoundError(data_dir / "images" / f"{key}.png")
        obs = points2d[key]
        for idx, (x, y, visible) in enumerate(obs):
            if visible > 0.5:
                cv2.circle(image, (int(round(x)), int(round(y))), 2, tuple(int(c) for c in colors[idx]), -1)
        cv2.imwrite(str(out_dir / f"{key}_overlay.png"), image)
        print(out_dir / f"{key}_overlay.png")


if __name__ == "__main__":
    main()
