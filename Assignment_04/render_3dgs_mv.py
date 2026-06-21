import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from data_utils import ColmapDataset
from gaussian_model import GaussianModel
from gaussian_renderer import GaussianRenderer


def look_at(eye, target, up):
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward], axis=0)
    translation = -rotation @ eye
    return rotation.astype(np.float32), translation.astype(np.float32)


def camera_centers(dataset):
    centers = []
    for item in dataset.items:
        R = item["R"]
        t = item["t"]
        centers.append(-R.T @ t)
    return np.stack(centers)


def orbit_path(dataset, frames):
    centers = camera_centers(dataset)
    scene_center = dataset.points3D_xyz.numpy().mean(axis=0)
    up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    radius = np.linalg.norm(centers - scene_center[None], axis=1).mean()
    height = (centers[:, 1] - scene_center[1]).mean()
    path_R, path_t = [], []
    for i in range(frames):
        theta = 2.0 * np.pi * i / frames
        eye = scene_center + np.array([radius * np.cos(theta), height, radius * np.sin(theta)], dtype=np.float32)
        R, t = look_at(eye, scene_center, up)
        path_R.append(R)
        path_t.append(t)
    return np.stack(path_R), np.stack(path_t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--num_frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--max_points", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset = ColmapDataset(args.colmap_dir, resize=args.resize, max_points=args.max_points)
    sample = dataset[0]
    h, w = sample["image"].shape[:2]
    K = sample["K"].to(device)
    model = GaussianModel(dataset.points3D_xyz, dataset.points3D_rgb).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("model", checkpoint.get("model_state_dict")))
    model.eval()
    renderer = GaussianRenderer(h, w).to(device)
    R_path, t_path = orbit_path(dataset, args.num_frames)
    output = Path(args.output) if args.output else Path(args.colmap_dir) / "render_mv.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    with torch.no_grad():
        params = model()
        for i in tqdm(range(args.num_frames), desc="render"):
            frame = renderer(
                params["positions"],
                params["covariance"],
                params["colors"],
                params["opacities"],
                K,
                torch.from_numpy(R_path[i]).to(device),
                torch.from_numpy(t_path[i]).to(device),
            )
            frame_np = (frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            writer.write(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
    writer.release()
    print(output)


if __name__ == "__main__":
    main()
