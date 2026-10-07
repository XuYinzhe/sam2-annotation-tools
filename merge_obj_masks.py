import os
import re
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, label


def fill_holes_mask(mask: np.ndarray) -> np.ndarray:
    """Fill holes in a binary mask."""
    # mask is assumed to be 0/255 or 0/1
    binary = (mask > 0).astype(np.uint8)
    filled = binary_fill_holes(binary).astype(np.uint8)
    return filled * 255


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area."""
    if min_area <= 0:
        return mask
    binary = (mask > 0).astype(np.uint8)
    labeled, num = label(binary, structure=np.ones((3, 3)))
    component_sizes = np.bincount(labeled.ravel())
    # index 0 is background
    keep = component_sizes[1:] >= min_area
    keep_mask = np.zeros(len(keep) + 1, dtype=bool)
    keep_mask[1:] = keep
    filtered = keep_mask[labeled].astype(np.uint8) * 255
    return filtered


def process_masks(input_dir: str, output_dir: str, min_area: int = 0):
    """Fill holes per object mask, merge all object masks per frame, and filter small components."""
    os.makedirs(output_dir, exist_ok=True)

    # Group files by frame id
    frame_files = {}
    pattern = re.compile(r"^(\d+)_obj_(\d+)\.png$")
    for fname in sorted(os.listdir(input_dir)):
        match = pattern.match(fname)
        if not match:
            continue
        frame_id, obj_id = match.groups()
        frame_files.setdefault(frame_id, []).append((int(obj_id), fname))

    if not frame_files:
        print(f"No *_obj_*.png masks found in {input_dir}")
        return

    for frame_id in sorted(frame_files.keys()):
        merged = None
        for obj_id, fname in sorted(frame_files[frame_id], key=lambda x: x[0]):
            fpath = os.path.join(input_dir, fname)
            mask = np.array(Image.open(fpath).convert("L"))
            filled = fill_holes_mask(mask)
            if merged is None:
                merged = filled
            else:
                merged = np.maximum(merged, filled)

        if min_area > 0:
            merged = remove_small_components(merged, min_area)

        out_path = os.path.join(output_dir, f"{frame_id}.png")
        Image.fromarray(merged.astype(np.uint8)).save(out_path)
        print(f"Saved {out_path} ({len(frame_files[frame_id])} objects merged, min_area={min_area})")


def main():
    parser = argparse.ArgumentParser(description="Fill holes in per-object masks, merge them per frame, and optionally filter small connected components.")
    parser.add_argument("input_dir", type=str, help="Directory containing *_obj_*.png masks")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory. Default: <input_dir>-merged")
    parser.add_argument("--min_area", type=int, default=0, help="Minimum connected component area in pixels to keep. Default: 0 (no filtering)")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or f"{input_dir.rstrip(os.sep)}-merged"
    process_masks(input_dir, output_dir, min_area=args.min_area)


if __name__ == "__main__":
    main()
