import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np


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


def run_demo():
    out_dir = Path("pics")
    out_dir.mkdir(exist_ok=True)
    image = make_demo_image()
    result = apply_transform(image, 0.82, 28, 45, -18, True)
    cv2.imwrite(str(out_dir / "global_demo.png"), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(out_dir / "global_demo.png")


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
