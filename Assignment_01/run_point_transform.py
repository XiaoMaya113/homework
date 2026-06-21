import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np


source_points = []
target_points = []
current_image = None


def _as_points(points):
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("control points must have shape (N, 2)")
    return pts


def thin_plate_kernel(radius):
    safe = np.maximum(radius, 1e-12)
    out = radius * radius * np.log(safe)
    out[radius < 1e-12] = 0.0
    return out


def solve_tps_displacement(anchors, values, regularization=1e-4):
    anchors = _as_points(anchors)
    values = _as_points(values)
    n = len(anchors)
    pairwise = np.linalg.norm(anchors[:, None, :] - anchors[None, :, :], axis=-1)
    kernel = thin_plate_kernel(pairwise)
    kernel += np.eye(n) * regularization
    affine = np.concatenate([np.ones((n, 1)), anchors], axis=1)
    lhs = np.block(
        [
            [kernel, affine],
            [affine.T, np.zeros((3, 3), dtype=np.float64)],
        ]
    )
    rhs = np.concatenate([values, np.zeros((3, 2), dtype=np.float64)], axis=0)
    params = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return params[:n], params[n:]


def evaluate_tps(points, anchors, radial_weights, affine_weights, chunk=65536):
    points = _as_points(points)
    outputs = np.empty((len(points), 2), dtype=np.float64)
    for start in range(0, len(points), chunk):
        end = min(start + chunk, len(points))
        block = points[start:end]
        dist = np.linalg.norm(block[:, None, :] - anchors[None, :, :], axis=-1)
        basis = thin_plate_kernel(dist)
        affine = np.concatenate([np.ones((len(block), 1)), block], axis=1)
        outputs[start:end] = basis @ radial_weights + affine @ affine_weights
    return outputs


def rbf_warp(image, src_pts, dst_pts, regularization=1e-4):
    if image is None:
        return None
    src_pts = _as_points(src_pts)
    dst_pts = _as_points(dst_pts)
    if len(src_pts) != len(dst_pts) or len(src_pts) < 3:
        return np.asarray(image).copy()

    img = np.asarray(image).copy()
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    output_pixels = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)

    inverse_displacement = src_pts - dst_pts
    radial, affine = solve_tps_displacement(dst_pts, inverse_displacement, regularization)
    displacement = evaluate_tps(output_pixels, dst_pts, radial, affine)
    sample = output_pixels + displacement
    map_x = sample[:, 0].reshape(h, w).astype(np.float32)
    map_y = sample[:, 1].reshape(h, w).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def upload_image(image):
    global current_image
    source_points.clear()
    target_points.clear()
    current_image = np.asarray(image).copy() if image is not None else None
    return current_image


def mark_points(evt: gr.SelectData):
    if current_image is None:
        return None
    x, y = evt.index
    if len(source_points) == len(target_points):
        source_points.append([x, y])
    else:
        target_points.append([x, y])
    canvas = current_image.copy()
    for p in source_points:
        cv2.circle(canvas, tuple(map(int, p)), 5, (30, 90, 255), -1, cv2.LINE_AA)
    for p in target_points:
        cv2.circle(canvas, tuple(map(int, p)), 5, (255, 70, 60), -1, cv2.LINE_AA)
    for a, b in zip(source_points, target_points):
        cv2.arrowedLine(canvas, tuple(map(int, a)), tuple(map(int, b)), (30, 180, 80), 2, cv2.LINE_AA)
    return canvas


def run_warp(regularization):
    if current_image is None:
        return None
    return rbf_warp(
        current_image,
        np.asarray(source_points, dtype=np.float64),
        np.asarray(target_points, dtype=np.float64),
        regularization=float(regularization),
    )


def clear_points():
    source_points.clear()
    target_points.clear()
    return current_image


def make_demo_image(size=360):
    image = np.full((size, size, 3), 245, np.uint8)
    cv2.ellipse(image, (180, 188), (92, 124), 0, 0, 360, (90, 160, 230), -1)
    cv2.circle(image, (145, 150), 14, (20, 20, 20), -1)
    cv2.circle(image, (215, 150), 14, (20, 20, 20), -1)
    cv2.ellipse(image, (180, 210), (55, 28), 0, 10, 170, (30, 30, 30), 4)
    cv2.putText(image, "RBF", (118, 322), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (70, 70, 70), 2, cv2.LINE_AA)
    return image


def run_demo():
    out_dir = Path("pics")
    out_dir.mkdir(exist_ok=True)
    image = make_demo_image()
    src = np.array([[145, 150], [215, 150], [125, 230], [235, 230], [180, 75]], dtype=np.float64)
    dst = np.array([[125, 140], [235, 140], [140, 245], [220, 245], [180, 50]], dtype=np.float64)
    warped = rbf_warp(image, src, dst, regularization=1e-3)
    cv2.imwrite(str(out_dir / "rbf_demo.png"), cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
    print(out_dir / "rbf_demo.png")


def build_app():
    with gr.Blocks(title="RBF Image Warp") as demo:
        gr.Markdown("# Point Guided RBF Warping")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Upload")
                point_view = gr.Image(label="Click source then target", interactive=True)
                reg = gr.Slider(1e-6, 1e-1, value=1e-4, step=1e-4, label="Regularization")
                with gr.Row():
                    run = gr.Button("Warp")
                    reset = gr.Button("Clear")
            output = gr.Image(label="Warped")
        image.upload(upload_image, image, point_view)
        point_view.select(mark_points, None, point_view)
        run.click(run_warp, reg, output)
        reset.click(clear_points, None, point_view)
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
