import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import ColmapDataset
from gaussian_model import GaussianModel
from gaussian_renderer import GaussianRenderer


def save_debug(path, target, rendered):
    target_np = (target.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    rendered_np = (rendered.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    image = np.concatenate([target_np, rendered_np], axis=1)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset = ColmapDataset(args.colmap_dir, resize=args.resize, max_points=args.max_points)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.workers)
    sample = dataset[0]["image"]
    height, width = sample.shape[:2]

    model = GaussianModel(dataset.points3D_xyz, dataset.points3D_rgb).to(device)
    renderer = GaussianRenderer(height, width, background=args.background).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": [model.positions], "lr": args.lr_xyz},
            {"params": [model.colors], "lr": args.lr_color},
            {"params": [model.opacities], "lr": args.lr_opacity},
            {"params": [model.scales], "lr": args.lr_scale},
            {"params": [model.rotations], "lr": args.lr_rotation},
        ],
        eps=1e-15,
    )
    checkpoint_dir = Path(args.checkpoint_dir)
    debug_dir = checkpoint_dir / "debug"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    fixed = dataset[0]
    fixed_batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in fixed.items()}

    for epoch in range(1, args.num_epochs + 1):
        total = 0.0
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for batch in pbar:
            gt = batch["image"].squeeze(0).to(device)
            K = batch["K"].squeeze(0).to(device)
            R = batch["R"].squeeze(0).to(device)
            t = batch["t"].squeeze(0).to(device)
            params = model()
            rendered = renderer(
                params["positions"],
                params["covariance"],
                params["colors"],
                params["opacities"],
                K,
                R,
                t,
            )
            loss_l1 = (rendered - gt).abs().mean()
            loss_opacity = 1e-4 * params["opacities"].mean()
            loss_scale = 1e-4 * params["scales"].mean()
            loss = loss_l1 + loss_opacity + loss_scale
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += loss_l1.item()
            pbar.set_postfix(l1=f"{loss_l1.item():.4f}")

        if epoch % args.debug_every == 0:
            with torch.no_grad():
                fixed_gt = fixed_batch["image"].squeeze(0).to(device)
                params = model()
                fixed_render = renderer(
                    params["positions"],
                    params["covariance"],
                    params["colors"],
                    params["opacities"],
                    fixed_batch["K"].squeeze(0).to(device),
                    fixed_batch["R"].squeeze(0).to(device),
                    fixed_batch["t"].squeeze(0).to(device),
                )
            save_debug(debug_dir / f"epoch_{epoch:04d}.png", fixed_gt, fixed_render)

        if epoch % args.save_every == 0 or epoch == args.num_epochs:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                checkpoint_dir / f"checkpoint_{epoch:06d}.pt",
            )
        print(f"epoch {epoch} mean_l1={total / max(len(loader), 1):.5f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_dir", required=True)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--max_points", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr_xyz", type=float, default=1.6e-5)
    parser.add_argument("--lr_color", type=float, default=2.5e-2)
    parser.add_argument("--lr_opacity", type=float, default=5e-2)
    parser.add_argument("--lr_scale", type=float, default=5e-3)
    parser.add_argument("--lr_rotation", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--background", type=float, default=0.0)
    parser.add_argument("--debug_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
