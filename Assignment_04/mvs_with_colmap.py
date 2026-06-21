import argparse
import os
import shutil
import subprocess
from pathlib import Path


def run_command(command):
    print(" ".join(str(part) for part in command))
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run a compact COLMAP SfM pipeline.")
    parser.add_argument("--data_dir", required=True, help="Scene directory containing images/")
    parser.add_argument("--colmap", default="colmap", help="COLMAP executable")
    parser.add_argument("--gpu", action="store_true", help="Enable SIFT GPU options")
    parser.add_argument("--pycolmap", action="store_true", help="Use pycolmap instead of the COLMAP executable")
    parser.add_argument("--force", action="store_true", help="Remove previous database and sparse output")
    args = parser.parse_args()

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    root = Path(args.data_dir)
    image_dir = root / "images"
    database = root / "database.db"
    sparse = root / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)

    if args.force:
        if database.exists():
            database.unlink()
        if sparse.exists():
            shutil.rmtree(sparse)
        sparse.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise FileNotFoundError(image_dir)

    if args.pycolmap or shutil.which(args.colmap) is None:
        import pycolmap

        print("Using pycolmap pipeline")
        pycolmap.extract_features(database, image_dir)
        pycolmap.match_exhaustive(database)
        reconstructions = pycolmap.incremental_mapping(database, image_dir, sparse)
        if not reconstructions:
            raise RuntimeError("pycolmap did not produce a reconstruction")
        print(f"Sparse models saved under {sparse}; model ids: {sorted(reconstructions.keys())}")
        return

    use_gpu = "1" if args.gpu else "0"
    run_command(
        [
            args.colmap,
            "feature_extractor",
            "--image_path",
            image_dir,
            "--database_path",
            database,
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            "PINHOLE",
            "--SiftExtraction.use_gpu",
            use_gpu,
        ]
    )
    run_command(
        [
            args.colmap,
            "exhaustive_matcher",
            "--database_path",
            database,
            "--SiftMatching.use_gpu",
            use_gpu,
        ]
    )
    run_command(
        [
            args.colmap,
            "mapper",
            "--image_path",
            image_dir,
            "--database_path",
            database,
            "--output_path",
            sparse,
        ]
    )
    print(f"Sparse model saved under {sparse}")


if __name__ == "__main__":
    main()
