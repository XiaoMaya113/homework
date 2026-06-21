from typing import Tuple

import torch
import torch.nn as nn


class GaussianRenderer(nn.Module):
    def __init__(self, image_height, image_width, background=0.0):
        super().__init__()
        self.H = int(image_height)
        self.W = int(image_width)
        y, x = torch.meshgrid(
            torch.arange(self.H, dtype=torch.float32),
            torch.arange(self.W, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("pixels", torch.stack([x, y], dim=-1))
        self.background = float(background)

    def compute_projection(
        self,
        means3D: torch.Tensor,
        covs3d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = means3D.shape[0]
        t = t.reshape(3)
        cam = means3D @ R.transpose(0, 1) + t[None, :]
        x, y, z = cam.unbind(-1)
        z_safe = z.clamp_min(1e-4)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        means2D = torch.stack([fx * x / z_safe + cx, fy * y / z_safe + cy], dim=-1)

        jacobian = torch.zeros(n, 2, 3, device=means3D.device, dtype=means3D.dtype)
        jacobian[:, 0, 0] = fx / z_safe
        jacobian[:, 0, 2] = -fx * x / (z_safe * z_safe)
        jacobian[:, 1, 1] = fy / z_safe
        jacobian[:, 1, 2] = -fy * y / (z_safe * z_safe)

        r_batch = R.unsqueeze(0).expand(n, -1, -1)
        cov_cam = r_batch @ covs3d @ r_batch.transpose(1, 2)
        cov2d = jacobian @ cov_cam @ jacobian.transpose(1, 2)
        cov2d = cov2d + torch.eye(2, device=means3D.device, dtype=means3D.dtype)[None] * 0.3
        return means2D, cov2d, z

    def compute_gaussian_values(self, means2D, covs2D, pixels):
        n = means2D.shape[0]
        offsets = pixels[None, :, :, :] - means2D[:, None, None, :]
        eye = torch.eye(2, device=covs2D.device, dtype=covs2D.dtype)[None]
        covs2D = covs2D + eye * 1e-4
        det = torch.linalg.det(covs2D).clamp_min(1e-8)
        inv = torch.linalg.inv(covs2D)
        exponent = torch.einsum("nhwi,nij,nhwj->nhw", offsets, inv, offsets)
        gaussian = torch.exp(-0.5 * exponent) / (2.0 * torch.pi * torch.sqrt(det))[:, None, None]
        return gaussian.clamp_max(1.0)

    def forward(self, means3D, covs3d, colors, opacities, K, R, t):
        n = means3D.shape[0]
        means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t)
        in_front = depths > 1e-3
        on_screen = (
            (means2D[:, 0] > -self.W)
            & (means2D[:, 0] < 2 * self.W)
            & (means2D[:, 1] > -self.H)
            & (means2D[:, 1] < 2 * self.H)
        )
        valid = in_front & on_screen
        order = torch.argsort(depths, descending=False)
        means2D = means2D[order]
        covs2D = covs2D[order]
        colors = colors[order]
        opacities = opacities[order]
        valid = valid[order]

        values = self.compute_gaussian_values(means2D, covs2D, self.pixels)
        alpha = (opacities.view(n, 1, 1) * values * valid.view(n, 1, 1)).clamp(0.0, 0.98)
        trans = torch.cumprod(
            torch.cat([torch.ones_like(alpha[:1]), (1.0 - alpha).clamp_min(1e-6)], dim=0),
            dim=0,
        )[:-1]
        weights = alpha * trans
        image = (weights[..., None] * colors[:, None, None, :]).sum(dim=0)
        if self.background > 0:
            image = image + trans[-1, :, :, None] * self.background
        return image.clamp(0.0, 1.0)
