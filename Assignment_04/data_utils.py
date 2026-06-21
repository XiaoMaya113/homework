import os
import struct
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def qvec_to_rotmat(qvec):
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float32,
    )


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(fid, 24, "iiQQ")
            num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}.get(model_id, 4)
            params = np.array(read_next_bytes(fid, 8 * num_params, "d" * num_params), dtype=np.float32)
            cameras[camera_id] = {"model_id": model_id, "width": width, "height": height, "params": params}
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(fid, 4, "i")[0]
            qvec = np.array(read_next_bytes(fid, 32, "dddd"), dtype=np.float32)
            tvec = np.array(read_next_bytes(fid, 24, "ddd"), dtype=np.float32)
            camera_id = read_next_bytes(fid, 4, "i")[0]
            name = b""
            while True:
                char = fid.read(1)
                if char == b"\x00":
                    break
                name += char
            num_points2d = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(num_points2d * 24, os.SEEK_CUR)
            images[image_id] = {"qvec": qvec, "tvec": tvec, "camera_id": camera_id, "name": name.decode("utf-8")}
    return images


def read_points3d_binary(path):
    xyz, rgb = [], []
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            read_next_bytes(fid, 8, "Q")
            point_xyz = read_next_bytes(fid, 24, "ddd")
            point_rgb = read_next_bytes(fid, 3, "BBB")
            read_next_bytes(fid, 8, "d")
            track_length = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(track_length * 8, os.SEEK_CUR)
            xyz.append(point_xyz)
            rgb.append(point_rgb)
    return np.asarray(xyz, dtype=np.float32), np.asarray(rgb, dtype=np.float32)


def parse_camera_matrix(camera):
    width, height = camera["width"], camera["height"]
    params = camera["params"]
    model_id = camera.get("model_id")
    if model_id in (0, 2, 3) or len(params) == 3:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:
        fx, fy, cx, cy = params[:4]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32), int(height), int(width)


class ColmapDataset(Dataset):
    def __init__(self, colmap_dir, resize=None, max_points=None):
        self.root = Path(colmap_dir)
        sparse = self.root / "sparse" / "0"
        cameras = read_cameras_binary(sparse / "cameras.bin")
        images = read_images_binary(sparse / "images.bin")
        xyz, rgb = read_points3d_binary(sparse / "points3D.bin")
        if max_points is not None and max_points > 0 and max_points < len(xyz):
            idx = np.linspace(0, len(xyz) - 1, max_points).astype(np.int64)
            xyz, rgb = xyz[idx], rgb[idx]
        self.points3D_xyz = torch.from_numpy(xyz).float()
        self.points3D_rgb = torch.from_numpy(rgb).float()
        self.items = []
        for item in sorted(images.values(), key=lambda x: x["name"]):
            camera = cameras[item["camera_id"]]
            K, h, w = parse_camera_matrix(camera)
            R = qvec_to_rotmat(item["qvec"])
            t = item["tvec"].astype(np.float32)
            path = self.root / "images" / item["name"]
            if not path.exists():
                continue
            self.items.append({"path": path, "K": K, "R": R, "t": t, "height": h, "width": w})
        if not self.items:
            raise FileNotFoundError(f"No registered images found under {self.root}")
        self.resize = resize

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image = cv2.imread(str(item["path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(item["path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        K = item["K"].copy()
        if self.resize:
            old_h, old_w = image.shape[:2]
            image = cv2.resize(image, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
            K[0, :] *= self.resize / old_w
            K[1, :] *= self.resize / old_h
        image = torch.from_numpy(image.astype(np.float32) / 255.0)
        return {
            "image": image,
            "K": torch.from_numpy(K).float(),
            "R": torch.from_numpy(item["R"]).float(),
            "t": torch.from_numpy(item["t"]).float(),
            "image_path": str(item["path"]),
        }
