import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage


def _rotation_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]], dtype=np.float32)


def _rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], dtype=np.float32)


def _rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]], dtype=np.float32)


def get_rotation_matrix(lon, lat, rotation=0):
    """
    Build the 3x3 rotation matrix used by 360VOT's rotate_zxy:
    R = Ry(lon) @ Rx(lat) @ Rz(rotation).
    Angles are in radians.
    """
    Rz = _rotation_z(rotation)
    Rx = _rotation_x(lat)
    Ry = _rotation_y(lon)
    return (Ry @ Rx @ Rz).astype(np.float32)


def _lonlat2xyz(lon, lat):
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(-lat)
    z = np.cos(lat) * np.cos(lon)
    return np.stack([x, y, z], axis=-1)


def _xyz2lonlat(x, y, z):
    lon = np.arctan2(x, z)
    lat = np.arcsin(-y)
    return lon, lat


def _prepare_tensor(img, device):
    """Convert numpy image/mask to torch tensor (1, C, H, W)."""
    img = np.asarray(img, dtype=np.float32)

    if img.ndim == 2:
        # (H, W) gray
        is_gray = True
        tensor = torch.from_numpy(img).to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif img.ndim == 3 and img.shape[0] == 1:
        # (1, H, W) single gray mask with leading dim
        is_gray = True
        tensor = torch.from_numpy(img).to(device).unsqueeze(0)  # (1, 1, H, W)
    elif img.ndim == 3 and img.shape[-1] == 1:
        # (H, W, 1)
        is_gray = True
        tensor = torch.from_numpy(img[..., 0]).to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif img.ndim == 3 and img.shape[-1] in (3, 4):
        # (H, W, C) image
        is_gray = False
        img = img[..., :3]  # keep RGB
        tensor = torch.from_numpy(img).to(device).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    else:
        raise ValueError(
            f"Unsupported image shape {img.shape}. "
            f"Expected (H, W), (1, H, W), (H, W, 1), or (H, W, 3/4)."
        )

    return tensor, is_gray


def _postprocess_tensor(tensor, is_gray):
    """Convert torch output back to numpy."""
    tensor = tensor.squeeze(0)  # remove batch dim
    if is_gray:
        return tensor.squeeze(0).cpu().numpy()  # remove channel dim, return (H, W)
    return tensor.permute(1, 2, 0).cpu().numpy()


def _compute_sample_grid(h, w, target_lat, inverse=False):
    """
    Compute the grid_sample grid for an ERP rotation that puts
    (lon=0, lat=target_lat) at the output center.
    """
    fx = w / (2.0 * np.pi)
    fy = -h / np.pi
    cx = w / 2.0
    cy = h / 2.0

    u_out = np.arange(w, dtype=np.float32) + 0.5
    v_out = np.arange(h, dtype=np.float32) + 0.5
    u_out, v_out = np.meshgrid(u_out, v_out, indexing='xy')

    lon_out = (u_out - cx) / fx
    lat_out = (v_out - cy) / fy

    directions = _lonlat2xyz(lon_out, lat_out)
    R = get_rotation_matrix(0, target_lat, 0)

    if inverse:
        # Rotated -> original: d_rotated = R^T @ d_original
        sample_dirs = directions @ R.T
    else:
        # Original -> rotated: d_original = R @ d_rotated
        sample_dirs = directions @ R

    x_r, y_r, z_r = sample_dirs[..., 0], sample_dirs[..., 1], sample_dirs[..., 2]
    lon_r, lat_r = _xyz2lonlat(x_r, y_r, z_r)

    u_in = lon_r * fx + cx - 0.5
    v_in = lat_r * fy + cy - 0.5

    # Normalize to [-1, 1] for grid_sample with align_corners=True
    u_norm = (u_in / (w - 1)) * 2.0 - 1.0
    v_norm = (v_in / (h - 1)) * 2.0 - 1.0

    grid = np.stack([u_norm, v_norm], axis=-1)
    return grid


def rotate_erp_image(img, target_lat, inverse=False, mode='bilinear',
                     padding_mode='border', device='cpu'):
    """
    Rotate an ERP image/mask so that (lon=0, lat=target_lat) is at the center.

    Args:
        img: numpy array, shape (H, W) or (H, W, C). Values can be uint8 or float.
        target_lat: latitude in radians of the point to center (e.g., -pi/2 for bottom pole).
        inverse: if True, rotate from centered view back to original ERP coordinates.
        mode: 'bilinear' or 'nearest' for grid_sample.
        padding_mode: grid_sample padding mode.
        device: 'cpu' or 'cuda'.

    Returns:
        numpy array of the same shape convention as input.
    """
    h, w = img.shape[:2]
    tensor, is_gray = _prepare_tensor(img, device)
    grid = _compute_sample_grid(h, w, target_lat, inverse=inverse)
    grid_t = torch.from_numpy(grid).to(device).unsqueeze(0)

    out = F.grid_sample(tensor, grid_t, mode=mode, padding_mode=padding_mode,
                        align_corners=True)
    return _postprocess_tensor(out, is_gray)


def pole_to_target_lat(pole):
    """Target latitude (radians) used to bring a pole to the ERP image center.

    Direction convention: 'bottom' -> +pi/2, 'top' -> -pi/2.
    """
    return np.pi / 2 if pole == 'bottom' else -np.pi / 2


def rotate_erp_image_fast(img, target_lat, inverse=False, mode='bilinear',
                          padding_mode='border', device='cpu', grid=None):
    """Same as rotate_erp_image, but accepts a precomputed grid.

    The sampling grid depends only on the image size and rotation, so a caller
    processing a sequence can compute it once and reuse it for every frame.
    """
    if grid is None:
        grid = _compute_sample_grid(img.shape[0], img.shape[1], target_lat, inverse=inverse)
    tensor, is_gray = _prepare_tensor(img, device)
    grid_t = torch.from_numpy(grid).to(device).unsqueeze(0)
    out = F.grid_sample(tensor, grid_t, mode=mode, padding_mode=padding_mode,
                        align_corners=True)
    return _postprocess_tensor(out, is_gray)


def list_image_files(directory):
    """Return the supported image filenames in a directory, sorted by name."""
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    files = [f for f in os.listdir(directory)
             if os.path.splitext(f)[-1].lower() in exts]
    files.sort()
    return files


def rotate_erp_frames_to_dir(input_dir, output_dir, target_lat, device='cpu',
                             progress=None, jpg_quality=95):
    """Rotate every image in input_dir and write 8-bit JPEGs to output_dir.

    Output files keep the source stem (e.g. 000000.png -> 000000.jpg) so that
    both the UI and SAM2 frame loaders (which sort by integer stem) agree on
    the frame order.

    Args:
        progress: optional callable(done, total) invoked once per frame.
        jpg_quality: JPEG quality for the written frames.

    Returns the list of written file names.
    """
    files = list_image_files(input_dir)
    if not files:
        raise ValueError(f"No supported images found in {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    first = np.array(Image.open(os.path.join(input_dir, files[0])).convert('RGB'))
    grid = _compute_sample_grid(first.shape[0], first.shape[1], target_lat, inverse=False)

    written = []
    total = len(files)
    for i, fname in enumerate(files):
        img = np.array(Image.open(os.path.join(input_dir, fname)).convert('RGB'))
        rotated = rotate_erp_image_fast(img, target_lat, inverse=False,
                                        mode='bilinear', padding_mode='border',
                                        device=device, grid=grid)
        rotated = np.clip(rotated, 0, 255).astype(np.uint8)
        out_name = os.path.splitext(fname)[0] + '.jpg'
        Image.fromarray(rotated).save(os.path.join(output_dir, out_name),
                                      quality=jpg_quality)
        written.append(out_name)
        if progress is not None:
            progress(i + 1, total)
    return written


def fill_mask_holes(mask):
    """Fill interior holes of a binary mask."""
    return ndimage.binary_fill_holes(mask > 0).astype(np.uint8)


def keep_largest_component(mask):
    """Keep only the largest connected component of a binary mask."""
    binary = (mask > 0).astype(np.uint8)
    labeled, num_features = ndimage.label(binary)
    if num_features <= 1:
        return binary
    sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    return (labeled == largest_label).astype(np.uint8)


def remove_small_components(mask, min_area):
    """Remove connected components smaller than min_area pixels."""
    binary = (mask > 0).astype(np.uint8)
    labeled, num_features = ndimage.label(binary)
    if num_features <= 1:
        return binary
    sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    keep_labels = np.where(sizes >= min_area)[0] + 1
    return np.isin(labeled, keep_labels).astype(np.uint8)


def rotate_mask_back_to_erp(mask, target_lat, device='cpu', grid=None,
                            threshold=0.0, fill_holes=True, keep_largest=False,
                            min_area=None):
    """Rotate one (H, W) mask from the pole-centered view back to ERP coords.

    Mirrors the offline pipeline: fill holes in the rotated view, inverse
    rotate (bilinear), binarize, then clean up connected components.
    Returns a 0/1 uint8 mask in original ERP resolution.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if fill_holes:
        binary = fill_mask_holes(binary)
    if binary.ndim == 3:
        raise ValueError(f"Expected a single (H, W) mask, got shape {binary.shape}")

    rotated = rotate_erp_image_fast(binary.astype(np.float32), target_lat,
                                    inverse=True, mode='bilinear',
                                    padding_mode='border', device=device, grid=grid)
    out = (rotated > threshold).astype(np.uint8)

    if keep_largest:
        out = keep_largest_component(out)
    elif min_area is not None and min_area > 0:
        out = remove_small_components(out, min_area)
    return out
