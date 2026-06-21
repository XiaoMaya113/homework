from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GaussianParameters:
    positions: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor
    covariance: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor


def quaternion_to_matrix(quaternions):
    q = F.normalize(quaternions, dim=-1, eps=1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


class GaussianModel(nn.Module):
    def __init__(self, points3D_xyz, points3D_rgb, init_opacity=0.4):
        super().__init__()
        xyz = torch.as_tensor(points3D_xyz, dtype=torch.float32)
        rgb = torch.as_tensor(points3D_rgb, dtype=torch.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        self.n_points = xyz.shape[0]
        self.positions = nn.Parameter(xyz)
        self.rotations = nn.Parameter(self._identity_quaternions(self.n_points))
        self.scales = nn.Parameter(torch.log(self._initial_scales(xyz)).repeat(1, 3))
        self.colors = nn.Parameter(torch.logit(rgb.clamp(1e-3, 1.0 - 1e-3)))
        opacity = torch.full((self.n_points, 1), float(init_opacity)).clamp(1e-3, 1.0 - 1e-3)
        self.opacities = nn.Parameter(torch.logit(opacity))

    @staticmethod
    def _identity_quaternions(count):
        q = torch.zeros(count, 4, dtype=torch.float32)
        q[:, 0] = 1.0
        return q

    @staticmethod
    def _initial_scales(points):
        n = points.shape[0]
        if n <= 1:
            return torch.full((n, 1), 0.01, dtype=torch.float32)
        with torch.no_grad():
            k = min(8, n - 1)
            distances = torch.cdist(points, points)
            distances.fill_diagonal_(float("inf"))
            nn_dist = torch.topk(distances, k=k, largest=False).values.mean(dim=1, keepdim=True)
            median = nn_dist.median().clamp_min(1e-4)
            return nn_dist.clamp(0.25 * median, 4.0 * median).clamp_min(1e-4)

    def compute_covariance(self):
        rotations = quaternion_to_matrix(self.rotations)
        scales = torch.exp(self.scales).clamp_min(1e-6)
        scale_matrix = torch.diag_embed(scales)
        transform = rotations @ scale_matrix
        return transform @ transform.transpose(1, 2)

    def get_gaussian_params(self):
        return GaussianParameters(
            positions=self.positions,
            colors=torch.sigmoid(self.colors),
            opacities=torch.sigmoid(self.opacities),
            covariance=self.compute_covariance(),
            rotations=F.normalize(self.rotations, dim=-1, eps=1e-8),
            scales=torch.exp(self.scales),
        )

    def forward(self):
        params = self.get_gaussian_params()
        return {
            "positions": params.positions,
            "colors": params.colors,
            "opacities": params.opacities,
            "covariance": params.covariance,
            "rotations": params.rotations,
            "scales": params.scales,
        }
