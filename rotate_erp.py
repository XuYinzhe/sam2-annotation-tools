import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from erp_rotation import rotate_erp_image


def rotate_erp_directory(input_dir, output_dir, pole='bottom', device='cpu', ext=None):
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    target_lat = np.pi / 2 if pole == 'bottom' else -np.pi / 2
    # Note: 360VOT convention uses y = sin(-lat), so the sign is reversed
    # relative to the standard ERP latitude. For 'bottom' pole we need +pi/2.

    if ext is None:
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    else:
        exts = {ext.lower() if ext.startswith('.') else '.' + ext.lower()}

    files = sorted([f for f in os.listdir(input_dir)
                    if os.path.splitext(f)[-1].lower() in exts])
    if not files:
        print(f"No supported images found in {input_dir}")
        return 1

    for fname in tqdm(files, desc=f"Rotating ERP ({pole} pole to center)"):
        in_path = os.path.join(input_dir, fname)
        out_name = os.path.splitext(fname)[0] + '.png'
        out_path = os.path.join(output_dir, out_name)

        img = np.array(Image.open(in_path).convert('RGB'))
        rotated = rotate_erp_image(img, target_lat=target_lat, inverse=False,
                                   mode='bilinear', padding_mode='border', device=device)
        rotated = np.clip(rotated, 0, 255).astype(np.uint8)
        Image.fromarray(rotated).save(out_path)

    print(f"Done. Rotated {len(files)} images to {output_dir}")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Rotate ERP images so the bottom/top pole is at the center.")
    parser.add_argument('input_dir', help='Directory containing ERP images')
    parser.add_argument('output_dir', help='Directory to save rotated images')
    parser.add_argument('--pole', choices=['bottom', 'top'], default='bottom',
                        help='Which pole to move to the center')
    parser.add_argument('--device', default='cuda', help='torch device: cpu or cuda')
    parser.add_argument('--ext', default=None, help='Only process files with this extension')
    args = parser.parse_args()
    sys.exit(rotate_erp_directory(args.input_dir, args.output_dir,
                                  pole=args.pole, device=args.device, ext=args.ext))
