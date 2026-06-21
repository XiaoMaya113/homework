import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def euler_xyz_to_matrix(angles):
    rx, ry, rz = angles.unbind(-1)
    sx, cx = torch.sin(rx), torch.cos(rx)
    sy, cy = torch.sin(ry), torch.cos(ry)
    sz, cz = torch.sin(rz), torch.cos(rz)

    zeros = torch.zeros_like(rx)
    ones = torch.ones_like(rx)
    rx_mat = torch.stack(
        [ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx], dim=-1
    ).reshape(*angles.shape[:-1], 3, 3)
    ry_mat = torch.stack(
        [cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy], dim=-1
    ).reshape(*angles.shape[:-1], 3, 3)
    rz_mat = torch.stack(
        [cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones], dim=-1
    ).reshape(*angles.shape[:-1], 3, 3)
    return rz_mat @ ry_mat @ rx_mat


def load_observations(data_dir, max_points=None, seed=7):
    data_dir = Path(data_dir)
    points2d = np.load(data_dir / "points2d.npz")
    keys = sorted(points2d.files)
    first = points2d[keys[0]]
    n_points = first.shape[0]
    if max_points is None or max_points <= 0 or max_points >= n_points:
        indices = np.arange(n_points)
    else:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(n_points, size=max_points, replace=False))

    obs = []
    vis = []
    for key in keys:
        view = points2d[key][indices]
        obs.append(view[:, :2].astype(np.float32))
        vis.append(view[:, 2].astype(np.float32))
    colors = np.load(data_dir / "points3d_colors.npy")[indices].astype(np.float32)
    if colors.max() > 1.0:
        colors /= 255.0
    return np.stack(obs), np.stack(vis), colors, indices


def project(points3d, angles, translations, focal, image_size):
    rotations = euler_xyz_to_matrix(angles)
    camera_points = torch.einsum("vij,nj->vni", rotations, points3d) + translations[:, None, :]
    x, y, z = camera_points.unbind(-1)
    z_safe = torch.where(z.abs() < 1e-5, z.sign() * 1e-5 - (z == 0).float() * 1e-5, z)
    center = image_size * 0.5
    u = -focal * x / z_safe + center
    v = focal * y / z_safe + center
    return torch.stack([u, v], dim=-1), z


def reprojection_loss(pred, obs, vis, points3d, translations, z_cam):
    residual = (pred - obs).pow(2).sum(dim=-1)
    data_loss = (residual * vis).sum() / vis.sum().clamp_min(1.0)
    center_loss = points3d.mean(dim=0).pow(2).sum()
    scale_loss = torch.relu(points3d.norm(dim=-1).mean() - 2.5).pow(2)
    front_loss = torch.relu(z_cam + 0.02).mean()
    translation_loss = translations[:, :2].pow(2).mean()
    total = data_loss + 1e-3 * center_loss + 5e-4 * scale_loss + 1e-2 * front_loss + 1e-4 * translation_loss
    return total, data_loss


def save_obj(path, points, colors):
    with open(path, "w", encoding="utf-8") as f:
        for point, color in zip(points, colors):
            x, y, z = point
            r, g, b = color
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")


def run(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    obs_np, vis_np, colors, selected = load_observations(args.data_dir, args.max_points, args.seed)
    n_views, n_points = vis_np.shape
    obs = torch.from_numpy(obs_np).to(device)
    vis = torch.from_numpy(vis_np).to(device)

    focal0 = args.image_size / (2.0 * math.tan(math.radians(args.fov) * 0.5))
    log_focal = torch.nn.Parameter(torch.tensor(math.log(focal0), dtype=torch.float32, device=device))
    angles = torch.nn.Parameter(torch.zeros(n_views, 3, dtype=torch.float32, device=device))
    translations = torch.nn.Parameter(torch.zeros(n_views, 3, dtype=torch.float32, device=device))
    translations.data[:, 2] = -args.distance
    points3d = torch.nn.Parameter(torch.randn(n_points, 3, dtype=torch.float32, device=device) * args.point_noise)

    optimizer = torch.optim.Adam(
        [
            {"params": [points3d], "lr": args.lr_points},
            {"params": [angles], "lr": args.lr_pose},
            {"params": [translations], "lr": args.lr_pose},
            {"params": [log_focal], "lr": args.lr_focal},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.iterations, 1))
    history = []
    best = {"loss": float("inf")}

    for step in range(args.iterations):
        focal = torch.exp(log_focal)
        pred, z_cam = project(points3d, angles, translations, focal, args.image_size)
        loss, data = reprojection_loss(pred, obs, vis, points3d, translations, z_cam)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([points3d, angles, translations, log_focal], args.grad_clip)
        optimizer.step()
        scheduler.step()
        history.append(float(data.detach().cpu()))

        if loss.item() < best["loss"]:
            best = {
                "loss": loss.item(),
                "data_loss": data.item(),
                "points3d": points3d.detach().cpu().numpy().copy(),
                "angles": angles.detach().cpu().numpy().copy(),
                "translations": translations.detach().cpu().numpy().copy(),
                "focal": float(torch.exp(log_focal).detach().cpu()),
            }
        if step % args.log_every == 0 or step == args.iterations - 1:
            focal_value = float(torch.exp(log_focal).detach().cpu())
            print(f"{step:05d} loss={loss.item():.4f} reproj_mse={data.item():.4f} f={focal_value:.2f}")

    plt.figure(figsize=(8, 4))
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Visible reprojection MSE")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close()

    save_obj(output_dir / "reconstruction.obj", best["points3d"], colors)
    np.savez(
        output_dir / "camera_params.npz",
        focal=best["focal"],
        euler_angles=best["angles"],
        translations=best["translations"],
        selected_point_indices=selected,
    )
    print(f"best reprojection RMSE: {math.sqrt(best['data_loss']):.4f}px")
    print(f"saved outputs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--max_points", type=int, default=3000)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--distance", type=float, default=2.5)
    parser.add_argument("--point_noise", type=float, default=0.08)
    parser.add_argument("--lr_points", type=float, default=0.05)
    parser.add_argument("--lr_pose", type=float, default=0.005)
    parser.add_argument("--lr_focal", type=float, default=0.005)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
