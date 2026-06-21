#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data}"
IMAGE_DIR="$DATA_DIR/images"
WORK_DIR="$DATA_DIR/colmap"
SPARSE_DIR="$WORK_DIR/sparse"
DENSE_DIR="$WORK_DIR/dense"
DATABASE="$WORK_DIR/database.db"

mkdir -p "$SPARSE_DIR" "$DENSE_DIR"

colmap feature_extractor \
  --database_path "$DATABASE" \
  --image_path "$IMAGE_DIR" \
  --ImageReader.camera_model PINHOLE \
  --ImageReader.single_camera 1

colmap exhaustive_matcher \
  --database_path "$DATABASE"

colmap mapper \
  --database_path "$DATABASE" \
  --image_path "$IMAGE_DIR" \
  --output_path "$SPARSE_DIR"

colmap image_undistorter \
  --image_path "$IMAGE_DIR" \
  --input_path "$SPARSE_DIR/0" \
  --output_path "$DENSE_DIR" \
  --output_type COLMAP

colmap patch_match_stereo \
  --workspace_path "$DENSE_DIR" \
  --workspace_format COLMAP

colmap stereo_fusion \
  --workspace_path "$DENSE_DIR" \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path "$DENSE_DIR/fused.ply"

echo "Sparse model: $SPARSE_DIR/0"
echo "Dense point cloud: $DENSE_DIR/fused.ply"
