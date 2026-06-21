import argparse
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def initialize_polygon():
    return {"points": [], "closed": False}


def draw_polygon_overlay(image, points, closed=False, fill=None):
    if image is None:
        return None
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    if len(points) > 1:
        draw.line(points + ([points[0]] if closed else []), fill="red", width=2)
    if closed and len(points) >= 3:
        draw.polygon(points, outline="red", fill=fill)
    for x, y in points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="blue")
    return canvas


def add_point(img_original, polygon_state, evt: gr.SelectData):
    if img_original is None or polygon_state["closed"]:
        return img_original, polygon_state
    x, y = evt.index
    polygon_state["points"].append((int(x), int(y)))
    return draw_polygon_overlay(img_original, polygon_state["points"]), polygon_state


def close_polygon(img_original, polygon_state):
    if img_original is None:
        return None, polygon_state
    if len(polygon_state["points"]) >= 3:
        polygon_state["closed"] = True
    return draw_polygon_overlay(img_original, polygon_state["points"], polygon_state["closed"]), polygon_state


def update_background(background_image_original, polygon_state, dx, dy):
    if background_image_original is None:
        return None
    if not polygon_state["closed"]:
        return background_image_original
    shifted = [(x + int(dx), y + int(dy)) for x, y in polygon_state["points"]]
    return draw_polygon_overlay(background_image_original, shifted, closed=True)


def create_mask_from_points(points, img_h, img_w):
    mask_image = Image.new("L", (img_w, img_h), 0)
    pts = [(int(x), int(y)) for x, y in np.asarray(points).reshape(-1, 2)]
    if len(pts) >= 3:
        ImageDraw.Draw(mask_image).polygon(pts, fill=255, outline=255)
    return np.asarray(mask_image, dtype=np.uint8)


def image_to_tensor(image, device):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def mask_to_tensor(mask, device):
    return torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)


def laplacian(image):
    channels = image.shape[1]
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3)
    return F.conv2d(image, kernel.repeat(channels, 1, 1, 1), padding=1, groups=channels)


def erode_mask(mask):
    keep = F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)
    return 1.0 - keep


def cal_laplacian_loss(foreground_img, foreground_mask, blended_img, background_mask):
    fg_lap = laplacian(foreground_img)
    out_lap = laplacian(blended_img)
    fg_region = foreground_mask.expand_as(fg_lap)
    bg_region = background_mask.expand_as(out_lap)
    if fg_region.sum() < 1 or bg_region.sum() < 1:
        return torch.zeros((), device=foreground_img.device, dtype=foreground_img.dtype)
    fg_vals = fg_lap[fg_region > 0.5]
    out_vals = out_lap[bg_region > 0.5]
    count = min(fg_vals.numel(), out_vals.numel())
    return F.mse_loss(out_vals[:count], fg_vals[:count])


def paste_foreground_into_background(fg, fg_mask, bg, bg_mask):
    fg_pixels = fg[fg_mask.bool().expand_as(fg)]
    result = bg.clone()
    target = bg_mask.bool().expand_as(bg)
    count = min(target.sum().item(), fg_pixels.numel())
    result[target] = fg_pixels[:count]
    return result


def poisson_blend(
    foreground_image_original,
    background_image_original,
    dx,
    dy,
    polygon_state,
    steps=1200,
    lr=0.03,
):
    if (
        foreground_image_original is None
        or background_image_original is None
        or not polygon_state["closed"]
        or len(polygon_state["points"]) < 3
    ):
        return background_image_original

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fg = image_to_tensor(foreground_image_original, device)
    bg = image_to_tensor(background_image_original, device)

    fg_pts = np.asarray(polygon_state["points"], dtype=np.int64)
    bg_pts = fg_pts + np.array([int(dx), int(dy)], dtype=np.int64)
    fg_mask = create_mask_from_points(fg_pts, fg.shape[-2], fg.shape[-1])
    bg_mask = create_mask_from_points(bg_pts, bg.shape[-2], bg.shape[-1])
    fg_mask_t = mask_to_tensor(fg_mask, device)
    bg_mask_t = mask_to_tensor(bg_mask, device)

    if bg_mask_t.sum() < 1:
        return background_image_original

    variable = paste_foreground_into_background(fg, fg_mask_t, bg, bg_mask_t).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([variable], lr=lr)
    inner = erode_mask(bg_mask_t)
    boundary = (bg_mask_t - inner).clamp(0.0, 1.0)

    for step in range(int(steps)):
        candidate = variable * bg_mask_t + bg.detach() * (1.0 - bg_mask_t)
        region_loss = cal_laplacian_loss(fg, fg_mask_t, candidate, bg_mask_t)
        boundary_loss = ((candidate - bg) * boundary).pow(2).mean()
        loss = region_loss + 50.0 * boundary_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            variable.clamp_(0.0, 1.0)
        if step == int(steps * 0.7):
            optimizer.param_groups[0]["lr"] *= 0.2

    result = (variable * bg_mask_t + bg * (1.0 - bg_mask_t)).detach()
    array = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (array.clip(0.0, 1.0) * 255).astype(np.uint8)


def blending(foreground_image_original, background_image_original, dx, dy, polygon_state):
    return poisson_blend(foreground_image_original, background_image_original, dx, dy, polygon_state)


def close_polygon_and_reset_dx(img_original, polygon_state, dx, dy, background_image_original):
    marked, state = close_polygon(img_original, polygon_state)
    background = update_background(background_image_original, state, 0, dy)
    return marked, state, background, gr.update(value=0)


def make_demo_images():
    fg = Image.new("RGB", (220, 180), (245, 247, 250))
    draw = ImageDraw.Draw(fg)
    draw.ellipse((45, 35, 175, 155), fill=(220, 80, 70), outline=(80, 40, 40), width=3)
    draw.line((75, 100, 145, 100), fill=(255, 240, 180), width=10)
    bg = Image.new("RGB", (300, 220), (90, 145, 170))
    draw = ImageDraw.Draw(bg)
    draw.rectangle((0, 140, 300, 220), fill=(60, 120, 100))
    draw.ellipse((190, 25, 270, 105), fill=(245, 215, 90))
    return fg, bg


def run_demo(steps):
    out_dir = Path("pics")
    out_dir.mkdir(exist_ok=True)
    fg, bg = make_demo_images()
    state = {"points": [(55, 45), (170, 45), (165, 150), (50, 145)], "closed": True}
    result = poisson_blend(fg, bg, 40, 20, state, steps=steps)
    Image.fromarray(result).save(out_dir / "poisson_demo.png")
    print(out_dir / "poisson_demo.png")


def build_app():
    with gr.Blocks(title="Poisson Image Blending") as demo:
        polygon_state = gr.State(initialize_polygon())
        background_original = gr.State(None)
        gr.Markdown("# Poisson Image Blending")
        with gr.Row():
            with gr.Column():
                fg = gr.Image(type="pil", label="Foreground", interactive=True)
                fg_marked = gr.Image(type="pil", label="Select Polygon", interactive=True)
                close = gr.Button("Close Polygon")
            with gr.Column():
                bg = gr.Image(type="pil", label="Background", interactive=True)
                bg_marked = gr.Image(type="pil", label="Target Position")
        with gr.Row():
            dx = gr.Slider(-500, 500, value=0, step=1, label="X Offset")
            dy = gr.Slider(-500, 500, value=0, step=1, label="Y Offset")
            run = gr.Button("Blend")
        output = gr.Image(label="Result")

        fg.change(lambda image: image, fg, fg_marked)
        fg_marked.select(add_point, [fg, polygon_state], [fg_marked, polygon_state])
        bg.change(lambda image: image, bg, background_original)
        close.click(close_polygon_and_reset_dx, [fg, polygon_state, dx, dy, background_original], [fg_marked, polygon_state, bg_marked, dx])
        dx.change(update_background, [background_original, polygon_state, dx, dy], bg_marked)
        dy.change(update_background, [background_original, polygon_state, dx, dy], bg_marked)
        run.click(blending, [fg, background_original, dx, dy, polygon_state], output)
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    if args.demo:
        run_demo(args.steps)
    else:
        build_app().launch()


if __name__ == "__main__":
    main()
