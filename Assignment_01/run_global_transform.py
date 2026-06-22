import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image


WHITE = (255, 255, 255)


def _homogeneous(tx=0.0, ty=0.0, angle=0.0, scale=1.0, flip_x=False):
    theta = np.deg2rad(angle)
    c, s = np.cos(theta), np.sin(theta)
    flip = -1.0 if flip_x else 1.0
    return np.array(
        [
            [scale * flip * c, -scale * s, tx],
            [scale * flip * s, scale * c, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def center_transform(width, height, scale=1.0, angle=0.0, tx=0.0, ty=0.0, flip_x=False):
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    to_center = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], np.float32)
    back = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], np.float32)
    move = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], np.float32)
    local = _homogeneous(angle=angle, scale=scale, flip_x=flip_x)
    return move @ back @ local @ to_center


def pad_canvas(image, ratio=0.35):
    h, w = image.shape[:2]
    pad = max(16, int(min(h, w) * ratio))
    if image.ndim == 2:
        return cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    return cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=WHITE)


def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):
    if image is None:
        return None
    src = np.asarray(image).copy()
    if src.ndim == 2:
        border_value = 255
    else:
        border_value = WHITE
    src = pad_canvas(src)
    h, w = src.shape[:2]
    matrix = center_transform(
        width=w,
        height=h,
        scale=float(scale),
        angle=float(rotation),
        tx=float(translation_x),
        ty=float(translation_y),
        flip_x=bool(flip_horizontal),
    )
    return cv2.warpAffine(
        src,
        matrix[:2],
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def make_demo_image(size=360):
    image = np.full((size, size, 3), 245, np.uint8)
    cv2.rectangle(image, (70, 80), (290, 270), (60, 130, 220), 4)
    cv2.circle(image, (180, 175), 70, (230, 90, 70), -1)
    cv2.line(image, (70, 270), (290, 80), (40, 40, 40), 5, cv2.LINE_AA)
    cv2.putText(image, "DIP", (112, 193), cv2.FONT_HERSHEY_SIMPLEX, 1.4, WHITE, 3, cv2.LINE_AA)
    return image


def _fit_into(image, width, height, background=(255, 255, 255)):
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _draw_panel(canvas, x, y, w, h, title, content=None):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (225, 230, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (x + 12, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 90, 105), 1, cv2.LINE_AA)
    if content is None:
        cv2.putText(
            canvas,
            "Waiting for output",
            (x + w // 2 - 95, y + h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (145, 150, 160),
            1,
            cv2.LINE_AA,
        )
        return
    content = _fit_into(content, w - 24, h - 44)
    canvas[y + 34 : y + 34 + content.shape[0], x + 12 : x + 12 + content.shape[1]] = content


def _draw_slider(canvas, x, y, w, label, value, minimum, maximum, display=None):
    text = display if display is not None else value
    cv2.putText(canvas, f"{label}: {text}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (55, 65, 75), 1, cv2.LINE_AA)
    cv2.line(canvas, (x, y), (x + w, y), (195, 205, 215), 5, cv2.LINE_AA)
    ratio = (float(value) - minimum) / (maximum - minimum)
    ratio = max(0.0, min(1.0, ratio))
    knob_x = int(x + ratio * w)
    cv2.circle(canvas, (knob_x, y), 9, (55, 130, 220), -1, cv2.LINE_AA)


def _save_gif(frames, output, duration=90):
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(output, save_all=True, append_images=pil_frames[1:], duration=duration, loop=0, optimize=True)


def make_demo_gif(image):
    frames = []
    frame_count = 54
    for i in range(frame_count):
        t = i / (frame_count - 1)
        smooth = t * t * (3.0 - 2.0 * t)
        scale = 1.0 + (0.82 - 1.0) * smooth
        rotation = 28.0 * smooth
        tx = 45.0 * smooth
        ty = -18.0 * smooth
        flip = t > 0.78
        result = apply_transform(image, scale, rotation, tx, ty, flip)

        canvas = np.full((650, 1120, 3), (248, 249, 251), dtype=np.uint8)
        cv2.putText(canvas, "Basic Image Geometric Transformation Demo", (26, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 40, 52), 2, cv2.LINE_AA)
        _draw_panel(canvas, 24, 56, 520, 360, "Upload Image", image)
        _draw_panel(canvas, 576, 56, 520, 360, "Transformed Result", result)

        cv2.rectangle(canvas, (24, 444), (1096, 624), (235, 239, 244), -1)
        _draw_slider(canvas, 54, 490, 420, "Scale", scale, 0.1, 2.5, f"{scale:.2f}")
        _draw_slider(canvas, 54, 552, 420, "Rotation", rotation, -180, 180, f"{rotation:.0f} deg")
        _draw_slider(canvas, 610, 490, 420, "Translate X", tx, -400, 400, f"{tx:.0f}")
        _draw_slider(canvas, 610, 552, 420, "Translate Y", ty, -400, 400, f"{ty:.0f}")
        cv2.rectangle(canvas, (610, 586), (632, 608), (55, 130, 220) if flip else (255, 255, 255), -1)
        cv2.rectangle(canvas, (610, 586), (632, 608), (120, 130, 145), 1)
        cv2.putText(canvas, f"Flip Horizontal: {'ON' if flip else 'OFF'}", (642, 604), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 65, 75), 1, cv2.LINE_AA)
        frames.append(canvas)
    return frames


def run_demo():
    out_dir = Path("pics")
    out_dir.mkdir(exist_ok=True)
    image = make_demo_image()
    output = out_dir / "global_demo.gif"
    _save_gif(make_demo_gif(image), output)
    print(output)


def build_app():
    with gr.Blocks(title="Image Transform") as demo:
        gr.Markdown("# Image Transformation")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Input")
                scale = gr.Slider(0.1, 2.5, value=1.0, step=0.05, label="Scale")
                rotation = gr.Slider(-180, 180, value=0, step=1, label="Rotation")
                tx = gr.Slider(-400, 400, value=0, step=1, label="Translate X")
                ty = gr.Slider(-400, 400, value=0, step=1, label="Translate Y")
                flip = gr.Checkbox(label="Flip Horizontal")
            output = gr.Image(label="Output")
        controls = [image, scale, rotation, tx, ty, flip]
        for control in controls:
            control.change(apply_transform, controls, output)
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        run_demo()
    else:
        build_app().launch()


if __name__ == "__main__":
    main()
