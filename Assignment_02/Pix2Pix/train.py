import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from FCN_network import PatchDiscriminator, UNetGenerator
from facades_dataset import FacadesDataset


def denormalize(x):
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def save_preview(source, fake, target, out_path):
    grid = torch.cat([denormalize(source[:4]), denormalize(fake[:4]), denormalize(target[:4])], dim=0)
    save_image(grid, out_path, nrow=min(4, source.shape[0]))


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset = FacadesDataset(args.data_root, "train", args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    generator = UNetGenerator(base_channels=args.base_channels).to(device)
    discriminator = PatchDiscriminator(base_channels=args.base_channels).to(device)
    gan_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    out_dir = Path(args.output_dir)
    (out_dir / "previews").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for batch in pbar:
            source = batch["source"].to(device)
            target = batch["target"].to(device)

            with torch.no_grad():
                fake_detached = generator(source).detach()
            real_logits = discriminator(source, target)
            fake_logits = discriminator(source, fake_detached)
            d_loss = 0.5 * (
                gan_loss(real_logits, torch.ones_like(real_logits))
                + gan_loss(fake_logits, torch.zeros_like(fake_logits))
            )
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            fake = generator(source)
            fake_logits = discriminator(source, fake)
            g_adv = gan_loss(fake_logits, torch.ones_like(fake_logits))
            g_l1 = l1_loss(fake, target) * args.lambda_l1
            g_loss = g_adv + g_l1
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            pbar.set_postfix(d=f"{d_loss.item():.3f}", g=f"{g_loss.item():.3f}", l1=f"{g_l1.item():.3f}")

        if epoch % args.preview_every == 0:
            save_preview(source, fake, target, out_dir / "previews" / f"epoch_{epoch:04d}.png")
        if epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                },
                out_dir / "checkpoints" / f"pix2pix_{epoch:04d}.pt",
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="datasets/facades")
    parser.add_argument("--output_dir", default="runs/facades")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--lambda_l1", type=float, default=100.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--preview_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
