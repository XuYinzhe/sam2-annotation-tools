import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
from erp_rotation import rotate_erp_image


def _load_mask(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext == '.npy':
        arr = np.load(path)
        # Normalize to 0/1 binary
        arr = arr.astype(np.float32)
        if arr.max() > 1:
            arr = arr / 255.0
        # Squeeze trailing singleton channel dim
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr
    elif ext in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}:
        img = np.array(Image.open(path).convert('L'))
        return (img > 127).astype(np.float32)
    else:
        raise ValueError(f"Unsupported mask format: {ext}")


def _fill_holes(mask):
    """Fill all interior holes in a binary mask."""
    return ndimage.binary_fill_holes(mask).astype(mask.dtype)


def _keep_largest_component(mask):
    """Keep only the largest connected component in a binary mask."""
    binary = (mask > 0).astype(np.uint8)
    labeled, num_features = ndimage.label(binary)
    if num_features <= 1:
        return binary
    sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    return (labeled == largest_label).astype(np.uint8)


def _remove_small_components(mask, min_area):
    """Remove connected components smaller than min_area."""
    binary = (mask > 0).astype(np.uint8)
    labeled, num_features = ndimage.label(binary)
    if num_features <= 1:
        return binary
    sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    keep_labels = np.where(sizes >= min_area)[0] + 1
    return np.isin(labeled, keep_labels).astype(np.uint8)


def _rotate_single_mask(mask, target_lat, threshold, device, fill_holes=False,
                        keep_largest=False, min_area=None):
    """Rotate one (H, W) binary mask back to original ERP coordinates."""
    if fill_holes:
        mask = _fill_holes(mask)
    rotated = rotate_erp_image(mask, target_lat=target_lat, inverse=True,
                               mode='bilinear', padding_mode='border', device=device)
    binary = (rotated > threshold).astype(np.uint8)

    if keep_largest:
        binary = _keep_largest_component(binary)
    elif min_area is not None and min_area > 0:
        binary = _remove_small_components(binary, min_area)

    return binary * 255


def rotate_masks_back(input_dir, output_dir, pole='bottom', threshold=0.0,
                      fill_holes=False, keep_largest=False, min_area=None,
                      device='cpu', ext=None):
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    target_lat = np.pi / 2 if pole == 'bottom' else -np.pi / 2
    # Note: 360VOT convention uses y = sin(-lat), so the sign is reversed
    # relative to the standard ERP latitude. For 'bottom' pole we need +pi/2.

    if ext is None:
        exts = {'.npy', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    else:
        exts = {ext.lower() if ext.startswith('.') else '.' + ext.lower()}

    files = sorted([f for f in os.listdir(input_dir)
                    if os.path.splitext(f)[-1].lower() in exts])
    if not files:
        print(f"No supported masks found in {input_dir}")
        return 1

    for fname in tqdm(files, desc=f"Rotating masks back ({pole} pole)"):
        in_path = os.path.join(input_dir, fname)
        mask = _load_mask(in_path)

        if mask.ndim == 3 and mask.shape[0] <= 100:
            # Multi-object mask: (N, H, W)
            for obj_idx in range(mask.shape[0]):
                single = mask[obj_idx]  # (H, W)
                binary = _rotate_single_mask(single, target_lat, threshold, device,
                                              fill_holes=fill_holes,
                                              keep_largest=keep_largest,
                                              min_area=min_area)
                out_name = os.path.splitext(fname)[0] + f'_obj_{obj_idx}.png'
                Image.fromarray(binary).save(os.path.join(output_dir, out_name))
        else:
            binary = _rotate_single_mask(mask, target_lat, threshold, device,
                                         fill_holes=fill_holes,
                                         keep_largest=keep_largest,
                                         min_area=min_area)
            out_name = os.path.splitext(fname)[0] + '.png'
            Image.fromarray(binary).save(os.path.join(output_dir, out_name))

    print(f"Done. Rotated {len(files)} masks back to {output_dir}")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Rotate binary masks from centered ERP view back to original ERP.")
    parser.add_argument('input_dir', help='Directory containing binary masks in rotated ERP view')
    parser.add_argument('output_dir', help='Directory to save masks in original ERP view')
    parser.add_argument('--pole', choices=['bottom', 'top'], default='bottom',
                        help='Which pole was moved to the center during forward rotation')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='After bilinear interpolation, mask pixels with value > threshold become white. '
                             'Use 0.0 for the dilated hack (bilinear + >0), 0.5 for standard unbiased resampling.')
    parser.add_argument('--fill-holes', action='store_true',
                        help='Fill all interior holes in the rotated masks before rotating back.')
    parser.add_argument('--keep-largest', action='store_true',
                        help='Keep only the largest connected component in each mask.')
    parser.add_argument('--min-area', type=int, default=None,
                        help='Remove connected components with area smaller than this value.')
    parser.add_argument('--device', default='cpu', help='torch device: cpu or cuda')
    parser.add_argument('--ext', default=None, help='Only process files with this extension')
    args = parser.parse_args()
    sys.exit(rotate_masks_back(args.input_dir, args.output_dir,
                                pole=args.pole, threshold=args.threshold,
                                fill_holes=args.fill_holes,
                                keep_largest=args.keep_largest,
                                min_area=args.min_area,
                                device=args.device, ext=args.ext))
