import argparse
from pathlib import Path

import cv2
import numpy as np

from data_utils import ColmapDataset


def project(points, K, R, t):
    cam = points @ R.T + t[None, :]
    z = cam[:, 2]
    valid = z > 1e-4
    uv = cam[:, :2] / z[:, None].clip(min=1e-4)
    uv = uv @ K[:2, :2].T + K[:2, 2]
    return uv, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_views", type=int, default=12)
    parser.add_argument("--resize", type=int, default=None)
    args = parser.parse_args()

    dataset = ColmapDataset(args.data_dir, resize=args.resize)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.data_dir) / "projection_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    points = dataset.points3D_xyz.numpy()
    colors = dataset.points3D_rgb.numpy().clip(0, 255).astype(np.uint8)

    for i in range(min(args.max_views, len(dataset))):
        item = dataset[i]
        image = (item["image"].numpy() * 255).clip(0, 255).astype(np.uint8)
        overlay = image.copy()
        K = item["K"].numpy()
        R = item["R"].numpy()
        t = item["t"].numpy()
        uv, valid = project(points, K, R, t)
        h, w = image.shape[:2]
        for point, color, ok in zip(uv, colors, valid):
            x, y = int(round(point[0])), int(round(point[1]))
            if ok and 0 <= x < w and 0 <= y < h:
                cv2.circle(overlay, (x, y), 2, tuple(int(c) for c in color[::-1]), -1)
        pair = np.concatenate([image, overlay], axis=1)
        out = out_dir / f"{Path(item['image_path']).stem}_projection.png"
        cv2.imwrite(str(out), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))
        print(out)


if __name__ == "__main__":
    main()
