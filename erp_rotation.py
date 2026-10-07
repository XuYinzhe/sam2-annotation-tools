import numpy as np
import torch
import torch.nn.functional as F


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
