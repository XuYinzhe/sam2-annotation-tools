import os
import sys
import argparse
import numpy as np
from PIL import Image


def normalize_mask_stack(masks, filename="<unknown>"):
    """Bring the loaded .npy array into shape (num_masks, H, W)."""
    masks = np.asarray(masks)

    if masks.ndim == 2:
        # (H, W) -> single mask
        return masks[np.newaxis, ...]

    if masks.ndim == 3:
        # Could be (num_masks, H, W) or (H, W, 1) or (1, H, W)
        if masks.shape[-1] == 1:
            # (H, W, 1) -> single mask
            return masks[..., 0][np.newaxis, ...]
        # Assume (num_masks, H, W)
        return masks

    if masks.ndim == 4:
        # Common SAM2 raw shape: (1, 1, H, W) or (B, 1, H, W)
        masks = np.squeeze(masks)
        if masks.ndim == 2:
            return masks[np.newaxis, ...]
        if masks.ndim == 3 and masks.shape[-1] == 1:
            return masks[..., 0][np.newaxis, ...]
        if masks.ndim == 3:
            return masks
        print(f"Warning: cannot normalize 4D array after squeezing for {filename}, shape {masks.shape}")
        return None

    print(f"Warning: unexpected mask array shape {masks.shape} for {filename}, skipping")
    return None


def convert_npy_to_binary(input_dir, output_dir=None):
    """Convert SAM2 .npy mask files to binary single-channel PNG images."""
    input_dir = os.path.abspath(input_dir)
    if not os.path.isdir(input_dir):
        print(f"Error: not a directory: {input_dir}")
        return 1

    if output_dir is None:
        output_dir = os.path.join(input_dir, "binary_masks")
    else:
        output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    npy_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".npy")])
    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        return 0

    saved_count = 0
    for npy_file in npy_files:
        npy_path = os.path.join(input_dir, npy_file)
        frame_name = os.path.splitext(npy_file)[0]
        try:
            raw_masks = np.load(npy_path)
        except Exception as e:
            print(f"Failed to load {npy_file}: {e}")
            continue

        masks = normalize_mask_stack(raw_masks, filename=npy_file)
        if masks is None:
            continue

        print(f"Processing {npy_file}: normalized shape {masks.shape}, raw shape {raw_masks.shape}")

        for obj_idx, mask in enumerate(masks):
            if mask.ndim != 2:
                print(f"  Skipping object {obj_idx}: unexpected slice shape {mask.shape}")
                continue
            binary_mask = (mask.astype(np.uint8)) * 255
            out_name = f"{frame_name}_obj_{obj_idx}.png"
            out_path = os.path.join(output_dir, out_name)
            Image.fromarray(binary_mask).save(out_path)
            saved_count += 1

    print(f"Done. Saved {saved_count} binary masks to {output_dir}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SAM2 .npy masks to binary PNG images.")
    parser.add_argument("input_dir", help="Directory containing .npy mask files")
    parser.add_argument("-o", "--output_dir", help="Output directory (default: input_dir/binary_masks)")
    args = parser.parse_args()
    sys.exit(convert_npy_to_binary(args.input_dir, args.output_dir))
