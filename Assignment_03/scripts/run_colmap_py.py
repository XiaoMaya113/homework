import argparse
import shutil
from pathlib import Path

import pycolmap


def main():
    parser = argparse.ArgumentParser(description="Run sparse COLMAP reconstruction with pycolmap.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    image_dir = data_dir / "images"
    work_dir = data_dir / "colmap_py"
    database = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"

    if args.force and work_dir.exists():
        shutil.rmtree(work_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    pycolmap.extract_features(database, image_dir)
    pycolmap.match_exhaustive(database)
    reconstructions = pycolmap.incremental_mapping(database, image_dir, sparse_dir)
    if not reconstructions:
        raise RuntimeError("No sparse reconstruction was produced.")
    print(f"Sparse models saved to {sparse_dir}; model ids: {sorted(reconstructions.keys())}")


if __name__ == "__main__":
    main()
