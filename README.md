# SAM2 Video Annotation Tool

A powerful GUI application for video segmentation using Meta's [Segment Anything 2 (SAM2)](https://github.com/facebookresearch/sam2) model. This tool allows users to annotate video frames with points and bounding boxes, then automatically generate segmentation masks across the entire video sequence.

Here is a [demo video](./use_case.mp4).

<video controls>
  <source src="./use_case.mp4" type="video/mp4">
</video>

## About This Fork

This repository is a fork of
[YuLiu-LY/sam2-annotation-tools](https://github.com/YuLiu-LY/sam2-annotation-tools). The upstream
tool is kept as-is; this fork adds a 360° equirectangular (ERP) annotation workflow on top of it,
plus batch tools for rotating frames and masks and for post-processing the exported masks.

### What This Fork Adds

- **ERP pole rotation in the GUI** — annotate 360° panoramas with the bottom (nadir) pole rotated to
  the image center, where objects near the camera are no longer distorted, and have masks
  inverse-rotated back to the original ERP coordinates on save
- **Binary mask export** — per-object `{frame}_obj_{id}.png` masks in addition to the upstream
  `.npy` export
- **Propagation recovery controls** — **Only segment forward from current frame** and **Clear
  Current Object From Current Frame**, for when propagation drifts and an object has to be
  re-segmented from a given frame onward
- **Keyboard shortcuts** — `r` previous frame, `f` next frame, `a` add point, `s` segment current
  frame
- **Batch command-line tools** — `rotate_erp.py`, `rotate_masks_back.py`, `merge_obj_masks.py` and
  `convert_npy_to_binary.py`, sharing the rotation code in `erp_rotation.py`
- **Bounded GPU memory** — the previous SAM2 inference state is released before a new sequence is
  loaded, instead of both sets of encoded frames staying on the GPU

See [360° ERP Support](#360-erp-support) and [Command-Line Tools](#command-line-tools) for details.

## Features

### Core Functionality
- **Video Frame Loading**: Load image sequences from directories containing video frames
- **Interactive Annotation**: Add positive/negative points and bounding boxes to guide segmentation
- **Multi-Object Support**: Annotate multiple objects with different Object IDs
- **Real-time Segmentation**: Generate masks for individual frames or entire video sequences
- **Animation Preview**: Play through annotated frames to visualize results
- **Mask Export**: Save segmentation masks as numpy arrays, binary PNGs and visualization images
- **360° ERP Support**: Optionally load panoramas with the bottom pole rotated to the image center
  and rotate masks back to ERP coordinates on save
- **Keyboard Shortcuts**: `r`/`f` previous/next frame, `a` add point, `s` segment current frame

### User Interface Components

#### File Path Settings
- **Input Directory**: Select folder containing video frame images (JPG, PNG, JPEG)
- **Save Subfolder**: Specify output directory for saved masks (default: "masks")
- **Rotate bottom pole to equator (360 ERP)**: Checkbox; load 360° panoramas in a pole-centered
  view so objects near the camera can be annotated without ERP distortion, with masks rotated back
  to the original coordinates on save. See [360° ERP Support](#360-erp-support). Toggling it
  reloads the sequence and discards the current points and masks.

#### Frame Navigation
- **Previous/Next Frame**: Navigate through video frames (shortcuts: `r` / `f`)
- **Frame Jump**: Directly jump to specific frame numbers
- **Animation Controls**: 
  - Play/Stop animation through frames
  - Adjust interval to show every Nth frame

#### Annotation Tools
- **Object ID**: Set ID for different objects (starts from 0)
- **Point Modes**: 
  - Positive Points (green): Indicate object regions
  - Negative Points (red): Indicate background regions
- **Add Point**: Click on image to add annotation points (shortcut: `a`)
- **Add Bounding Box**: Click twice to define rectangular regions
- **Clear Annotations**: Remove all annotations from current frame
- **Reset State**: Clear all annotations and segmentation results

#### Segmentation
- **Segment Current Frame**: Generate mask for current frame only (shortcut: `s`)
- **Segment All Frames**: Propagate segmentation across entire video
- **Segment Current Object in All Frames**: Propagate only the active object ID across the video
- **Clear Current Object From Current Frame**: Delete the active object's masks and prompts on the
  current frame and all later frames (earlier frames are kept), useful when
  propagation drifts and the object must be re-segmented from a given frame onward
- **Only segment forward from current frame**: Checkbox; when checked, the two propagation buttons
  start at the current frame and skip earlier frames, leaving masks already computed for those
  frames untouched

#### Save Results
- **Save Masks**: Export segmentation results as numpy arrays and visualization images
- **Save Binary Masks**: Export every object mask as a binary PNG (`{frame}_obj_{id}.png`) into the
  save subfolder
- **Save Current Object Binary Masks**: Same, but only for the active Object ID

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- PyQt5
- SAM2 model checkpoint

### Setup
1. Install required dependencies:
```bash
cd segment-anything-2
pip install -e .
pip install PyQt5
pip install matplotlib
pip install Pillow
pip install scikit-image
pip install scipy
pip install tqdm
```

`scipy` is used by the mask cleanup routines and `tqdm` by the batch rotation scripts, so both
are needed for the [command-line tools](#command-line-tools).

2. Download SAM2 model checkpoint:
   - Place the checkpoint file in `./segment-anything-2/checkpoints/sam2.1_hiera_large.pt`
   - Or the application will prompt you to select the checkpoint file location

3. Ensure SAM2 library is properly installed in the `segment-anything-2/` directory

## Usage

### Starting the Application
```bash
python segment_ui.py
```

### Basic Workflow

1. **Load Video Frames**
   - Click "Browse..." to select directory containing video frame images
   - Supported formats: JPG, PNG, JPEG
   - Frames should be named in sequential order
   - For 360° panoramas, tick the ERP checkbox first (see [360° ERP Support](#360-erp-support))

2. **Navigate Frames**
   - Use "Previous Frame" / "Next Frame" buttons
   - Or enter frame number and click "Jump"
   - Use animation mode to preview video

3. **Add Annotations**
   - Set Object ID for the object you want to segment
   - Choose point mode (Positive/Negative)
   - Click "Add Point" then click on image to add annotation points
   - Or click "Add Bounding Box" and define rectangular region

4. **Generate Segmentation**
   - Click "Segment Current Frame" to process current frame
   - Click "Segment All Frames" to propagate across entire video
   - View results overlaid on frames

5. **Save Results**
   - Click "Save Masks" to export segmentation masks
   - Masks are saved as numpy arrays (.npy files), one per object
   - Use "Save Binary Masks" (or "Save Current Object Binary Masks") for per-object binary PNGs
   - Visualization images are also generated

### Advanced Features

#### Multi-Object Annotation
- Change Object ID to annotate different objects
- Each object maintains separate annotations and masks
- Objects are color-coded in visualization

#### Animation Mode
- Click "Animation" to start frame-by-frame playback
- Adjust "Interval" to control playback speed
- Useful for reviewing segmentation results

#### Frame Caching
- Application automatically caches frames for smooth navigation
- Cache size is limited to prevent memory issues

## 360° ERP Support

Equirectangular (ERP) panoramas are heavily stretched near the poles, so objects directly below
the camera — the operator, a stick, a cart — are hard to annotate accurately. This fork can load
the sequence with the bottom (nadir) pole rotated to the image center, and undoes that rotation
for every mask it exports.

### Annotating with pole rotation

1. Tick **Rotate bottom pole to equator (360 ERP)** in the File Path Settings group.
2. The sequence reloads: every frame is rotated into a sibling cache directory
   `<input_dir>.erp_rotated` (JPEGs keeping the original stems) and SAM2 is initialized on that
   cache. Progress is reported in the status bar.
3. Annotate and propagate as usual — points, boxes and masks all live in the pole-centered view,
   where pole-area objects appear undistorted.
4. On save, each mask is inverse-rotated back to the original ERP resolution and cleaned (holes
   filled, largest connected component kept), so the exported files line up with the raw frames.
   Visualization images are drawn over the original, un-rotated frames.

Notes:
- Toggling the checkbox reloads the sequence and discards the current points and masks; the UI
  asks for confirmation first.
- The cache is reused within a session, rebuilt when the source directory changes, and deleted
  when the application exits. Source frames are never modified.
- The GUI checkbox always rotates the bottom pole; the CLI tools can rotate either pole.

### CLI equivalents

The same rotation can be applied ahead of time with `rotate_erp.py` and undone with
`rotate_masks_back.py`. The GUI checkbox is a convenience wrapper that does both ends of it within
one annotation session.

## Command-Line Tools

These scripts share the rotation code in `erp_rotation.py` and work independently of the GUI.

### `rotate_erp.py`

Rotate every image of a directory so the chosen pole lands at the image center.

```bash
python rotate_erp.py input_dir output_dir --pole bottom --device cuda
```

- `input_dir` / `output_dir`: source images and destination for the rotated PNGs
- `--pole {bottom,top}`: which pole to move to the center (default `bottom`)
- `--device`: torch device, e.g. `cpu` or `cuda` (default `cuda`)
- `--ext`: only process files with this extension

Output keeps the source stem (`000123.jpg` → `000123.png`), so frame ordering is preserved.

### `rotate_masks_back.py`

Rotate binary masks from the pole-centered view back to the original ERP coordinates.

```bash
python rotate_masks_back.py input_dir output_dir --pole bottom \
    --fill-holes --keep-largest --device cuda
```

- Accepts `.npy` and image files; a multi-object `.npy` of shape `(N, H, W)` is written as
  `{name}_obj_{i}.png`, a single mask as `{name}.png`
- `--threshold`: pixels above this value become white after bilinear resampling. `0.0` is the
  dilated convention, `0.5` the unbiased one
- `--fill-holes` / `--keep-largest` / `--min-area`: mask cleanup applied after rotating back
- `--device`, `--ext`: as described above

### `merge_obj_masks.py`

Fill holes in each per-object mask, merge all objects of a frame into one mask, and optionally drop
small components.

```bash
python merge_obj_masks.py input_dir --output_dir merged --min_area 100
```

- Reads the `{frame}_obj_{id}.png` files produced by the annotation tool
- `--output_dir`: defaults to `<input_dir>-merged`
- `--min_area`: drop connected components smaller than this many pixels (default `0`, no filtering)

### `convert_npy_to_binary.py`

Convert the tool's `.npy` mask stacks into one binary PNG per object.

```bash
python convert_npy_to_binary.py masks_dir -o binary_masks
```

- Accepts the array shapes produced by SAM2 and the GUI; writes `{frame}_obj_{i}.png`
- `-o` / `--output_dir`: defaults to `<input_dir>/binary_masks`

### Batch examples

`erp_360roam.sh` and `back_360roam.sh` show how the two rotation scripts are driven over a whole
dataset. They contain absolute paths from the machine this fork was developed on — adjust them
before use.

## Output Format

### Saved Files
- **Masks**: `{frame_name}.npy` - one binary mask per object, stacked as (num_objects, H, W)
- **Binary masks**: `{frame_name}_obj_{id}.png` - one binary PNG per object, written next to the
  `.npy` files
- **Visualizations**: `{frame_name}.png` - overlaid masks on the corresponding frames

Both mask exports are written in original ERP coordinates when pole rotation is active.

### Directory Structure
```
output_directory/
├── masks/               # Numpy mask files and binary PNGs
│   ├── frame_001.npy
│   ├── frame_001_obj_0.png
│   ├── frame_002.npy
│   ├── frame_002_obj_0.png
│   └── ...
└── masks_vis/           # Visualization images
    ├── frame_001.png
    ├── frame_002.png
    └── ...
```

## Technical Details

### Model Configuration
- Uses SAM2.1 Hiera Large model by default
- Supports CUDA, MPS, and CPU inference
- Automatic mixed precision for CUDA devices

### Performance Optimizations
- Frame caching for smooth navigation
- Optimized matplotlib rendering
- Background frame preloading
- Memory management for large video sequences
- The previous SAM2 inference state is released (GPU memory freed) before a new sequence is
  loaded, instead of both sets of encoded frames coexisting
- ERP pole-rotated frames are rotated once per source directory and cached for the session, and
  the rotation grid is computed once per sequence rather than per frame

### Supported Platforms
- Linux (tested on Ubuntu)

## Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Ensure SAM2 checkpoint file is in correct location
   - Check file permissions and path

2. **Memory Issues**
   - Reduce frame cache size in code
   - Close other applications to free memory

3. **No Frames Loaded**
   - Check image file formats (JPG, PNG, JPEG)
   - Ensure directory contains image files
   - Verify file naming convention

4. **ERP Rotation Issues**
   - The rotated frames are written to `<input_dir>.erp_rotated` next to the source directory, so
     the parent of that directory must be writable; the status bar reports the error if it is not
   - Rotating a whole sequence takes time on the first load, since it runs on the same device as
     SAM2; reloading the same directory in the same session reuses the cache instead of redoing it



## License

This tool is built on top of Meta's Segment Anything 2 model. Please refer to the original SAM2 license for model usage terms.

## Contributing

Feel free to submit issues and enhancement requests. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

- Meta AI for the Segment Anything 2 model
- Open source contributors to the supporting libraries
