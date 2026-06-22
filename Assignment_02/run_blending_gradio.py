import argparse
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps


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


def align_foreground_to_background(foreground_image, background_image, points, dx, dy):
    fg = np.asarray(foreground_image.convert("RGB"), dtype=np.uint8)
    bg = np.asarray(background_image.convert("RGB"), dtype=np.uint8)
    fg_h, fg_w = fg.shape[:2]
    bg_h, bg_w = bg.shape[:2]
    offset = np.array([int(dx), int(dy)], dtype=np.int64)
    fg_pts = np.asarray(points, dtype=np.int64)
    bg_pts = fg_pts + offset

    fg_mask = create_mask_from_points(fg_pts, fg_h, fg_w) > 0
    bg_mask = create_mask_from_points(bg_pts, bg_h, bg_w) > 0
    yy, xx = np.mgrid[0:bg_h, 0:bg_w]
    sx = xx - offset[0]
    sy = yy - offset[1]
    in_source = (sx >= 0) & (sx < fg_w) & (sy >= 0) & (sy < fg_h)
    valid = bg_mask & in_source
    source_inside = np.zeros_like(valid)
    source_inside[valid] = fg_mask[sy[valid], sx[valid]]
    valid &= source_inside

    source_aligned = bg.copy()
    initial = bg.copy()
    source_aligned[valid] = fg[sy[valid], sx[valid]]
    initial[valid] = fg[sy[valid], sx[valid]]
    return source_aligned, initial, (valid.astype(np.uint8) * 255)


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
    source_aligned, initial, bg_mask = align_foreground_to_background(
        foreground_image_original,
        background_image_original,
        polygon_state["points"],
        dx,
        dy,
    )
    bg = image_to_tensor(background_image_original, device)
    source = image_to_tensor(Image.fromarray(source_aligned), device)
    bg_mask_t = mask_to_tensor(bg_mask, device)

    if bg_mask_t.sum() < 1:
        return background_image_original

    variable = image_to_tensor(Image.fromarray(initial), device).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([variable], lr=lr)
    inner = erode_mask(bg_mask_t)
    if inner.sum() < 1:
        inner = bg_mask_t
    boundary = (bg_mask_t - inner).clamp(0.0, 1.0)
    source_lap = laplacian(source).detach()
    denom = bg_mask_t.sum().clamp_min(1.0) * 3.0

    for step in range(int(steps)):
        candidate = variable * bg_mask_t + bg.detach() * (1.0 - bg_mask_t)
        region_loss = ((laplacian(candidate) - source_lap) * inner).pow(2).sum() / denom
        color_loss = ((candidate - source) * inner).pow(2).sum() / denom
        boundary_loss = ((candidate - bg) * boundary).pow(2).sum() / denom
        loss = region_loss + 0.08 * color_loss + 30.0 * boundary_loss
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
    fg = Image.new("RGB", (420, 280), (116, 167, 182))
    draw = ImageDraw.Draw(fg)
    for y in range(280):
        color = (95 + y // 12, 148 + y // 18, 166 + y // 20)
        draw.line((0, y, 420, y), fill=color)
    for x in range(-40, 450, 55):
        draw.arc((x, 54, x + 80, 100), 0, 180, fill=(215, 232, 236), width=2)
        draw.arc((x, 178, x + 95, 224), 0, 180, fill=(202, 224, 230), width=2)
    draw.ellipse((115, 92, 295, 190), fill=(235, 108, 70), outline=(113, 58, 48), width=3)
    draw.polygon([(292, 140), (358, 90), (342, 140), (360, 192)], fill=(231, 92, 64), outline=(113, 58, 48))
    draw.polygon([(184, 95), (222, 42), (228, 107)], fill=(242, 143, 82), outline=(113, 58, 48))
    draw.polygon([(188, 190), (228, 235), (230, 176)], fill=(242, 143, 82), outline=(113, 58, 48))
    draw.ellipse((142, 120, 164, 142), fill=(245, 245, 245))
    draw.ellipse((151, 128, 159, 136), fill=(25, 25, 25))
    draw.arc((130, 142, 184, 176), 10, 155, fill=(91, 45, 42), width=3)

    bg = Image.new("RGB", (560, 360), (139, 192, 235))
    draw = ImageDraw.Draw(bg)
    for y in range(360):
        if y < 130:
            color = (130 + y // 10, 188 + y // 15, 236)
        elif y < 285:
            color = (64, 155 + (y - 130) // 8, 183 + (y - 130) // 14)
        else:
            color = (226, 203 - (y - 285) // 9, 142)
        draw.line((0, y, 560, y), fill=color)
    draw.ellipse((56, 42, 170, 82), fill=(242, 248, 252))
    draw.ellipse((122, 34, 246, 80), fill=(242, 248, 252))
    draw.rectangle((0, 130, 560, 136), fill=(238, 246, 249))
    for x in range(0, 580, 42):
        draw.arc((x, 210, x + 82, 260), 0, 180, fill=(226, 242, 244), width=2)
    draw.rectangle((0, 285, 560, 360), fill=(225, 200, 139))
    draw.arc((40, 292, 520, 410), 190, 350, fill=(244, 236, 207), width=7)
    return fg, bg


def make_demo_polygon():
    return [
        (108, 145),
        (126, 114),
        (176, 88),
        (215, 54),
        (230, 104),
        (276, 108),
        (360, 88),
        (344, 140),
        (362, 192),
        (296, 169),
        (232, 178),
        (228, 236),
        (190, 195),
        (145, 184),
        (116, 164),
    ]


def _panel(draw, sheet, title, image, box, note):
    x, y, w, h = box
    draw.text((x, y), title, fill=(38, 44, 52))
    frame = Image.new("RGB", (w, h), (248, 250, 252))
    content = ImageOps.contain(image, (w, h))
    frame.paste(content, ((w - content.width) // 2, (h - content.height) // 2))
    sheet.paste(frame, (x, y + 28))
    draw.rectangle((x, y + 28, x + w, y + 28 + h), outline=(218, 224, 232), width=1)
    draw.text((x, y + 42 + h), note, fill=(76, 82, 90))


def make_poisson_result_pages(fg, bg, polygon, dx, dy, result):
    fg_marked = draw_polygon_overlay(fg, polygon, closed=True)
    bg_marked = update_background(bg, {"points": polygon, "closed": True}, dx, dy)
    result_image = Image.fromarray(result) if isinstance(result, np.ndarray) else result

    page1 = Image.new("RGB", (980, 940), (255, 255, 255))
    draw1 = ImageDraw.Draw(page1)
    draw1.text((350, 24), "Poisson Image Blending", fill=(36, 40, 46))
    draw1.text((76, 70), "Select a foreground polygon, then translate that polygon onto the target background.", fill=(58, 64, 72))
    _panel(draw1, page1, "Foreground Image", fg, (70, 120, 390, 260), "Source image used for selecting the object region.")
    _panel(draw1, page1, "Background Image", bg, (520, 120, 390, 260), "Background where the selected object will be placed.")
    _panel(draw1, page1, "Foreground Image with Polygon", fg_marked, (70, 500, 390, 260), "The red polygon encloses the foreground object.")
    draw1.text((70, 835), "The selected polygon is intentionally tight around the object, so background pixels are not copied as a visible block.", fill=(76, 82, 90))

    page2 = Image.new("RGB", (980, 720), (255, 255, 255))
    draw2 = ImageDraw.Draw(page2)
    draw2.text((350, 24), "Poisson Image Blending", fill=(36, 40, 46))
    draw2.text((92, 70), "After choosing the target position, Poisson optimization preserves source gradients and matches the target boundary.", fill=(58, 64, 72))
    _panel(draw2, page2, "Target Position on Background", bg_marked, (70, 120, 390, 260), "Translated polygon on the background image.")
    _panel(draw2, page2, "Blended Result", result_image, (520, 120, 390, 260), "Final Poisson result at the selected location.")
    draw2.text((70, 470), f"Offset: dx={dx}, dy={dy}; polygon points={len(polygon)}; optimization steps match source Laplacian with boundary constraints.", fill=(76, 82, 90))

    sheet = Image.new("RGB", (1180, 1040), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((404, 24), "Poisson Image Blending", fill=(36, 40, 46))
    draw.text((178, 70), "Foreground selection, target placement, and final blended result generated by the same demo command.", fill=(58, 64, 72))
    _panel(draw, sheet, "Foreground Image", fg, (70, 118, 480, 300), "Source object before selecting the polygon.")
    _panel(draw, sheet, "Background Image", bg, (630, 118, 480, 300), "Background where the selected polygon will be placed.")
    _panel(draw, sheet, "Foreground Image with Polygon", fg_marked, (70, 512, 480, 300), "The red polygon defines the source region.")
    _panel(draw, sheet, "Blended Result", result_image, (630, 512, 480, 300), "Final Poisson result at the selected target location.")
    draw.text((70, 908), "Target overlay:", fill=(38, 44, 52))
    preview = ImageOps.contain(bg_marked, (250, 130))
    sheet.paste(preview, (250, 856))
    draw.text((630, 844), f"Offset: dx={dx}, dy={dy}; polygon points={len(polygon)}.", fill=(76, 82, 90))
    return page1, page2, sheet


def run_demo(steps):
    out_dir = Path("pics")
    out_dir.mkdir(exist_ok=True)
    fg, bg = make_demo_images()
    polygon = make_demo_polygon()
    dx, dy = 78, 58
    state = {"points": polygon, "closed": True}
    result = poisson_blend(fg, bg, dx, dy, state, steps=steps)
    page1, page2, sheet = make_poisson_result_pages(fg, bg, polygon, dx, dy, result)
    page1.save(out_dir / "poisson_result1.png")
    page2.save(out_dir / "poisson_result2.png")
    sheet.save(out_dir / "poisson_demo.png")
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
