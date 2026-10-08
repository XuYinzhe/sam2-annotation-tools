import os
import gc
import shutil
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QWidget, QFileDialog, QSpinBox, QGroupBox, QRadioButton,
                             QCheckBox, QShortcut, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence
from sam2.build_sam import build_sam2_video_predictor

from erp_rotation import (pole_to_target_lat, rotate_erp_frames_to_dir,
                          rotate_mask_back_to_erp)


ERP_ROTATED_SUFFIX = ".erp_rotated"
ERP_MASK_CLEANUP = dict(fill_holes=True, keep_largest=True, min_area=None)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        # Optimize figure rendering
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        super(MplCanvas, self).__init__(self.fig)


class SAM2AnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM2 Video Annotation Tool")
        self.setGeometry(200, 200, 2000, 1500)
        
        # Optimize Matplotlib settings
        plt.rcParams['path.simplify'] = True
        plt.rcParams['path.simplify_threshold'] = 1.0
        plt.rcParams['agg.path.chunksize'] = 10000
        
        # Initialize variables
        self.data_dir = ""      # source ERP directory, also the base for save paths
        self.active_dir = ""    # directory currently displayed / loaded by SAM2 (rotated cache or source)
        self.frame_names = []
        self.current_frame_idx = 0
        self.inference_state = None
        self.predictor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ann_obj_id = 0  # Starting from 0 instead of 1
        self.video_segments = {}
        self.point_mode = "positive"  # Default mode for point adding
        self.add_point_mode = False   # Flag for adding points mode

        # ERP pole-rotation state
        self.erp_pole = None            # None = disabled; 'bottom'/'top' = rotate that pole to the center
        self.erp_cache_dir = None       # rotated-frame cache currently in use
        self.erp_caches = {}            # pole -> cache dir, everything created this session (removed on exit)
        self.erp_rotating = False       # guard while a rotation pass is running
        self._source_files = None       # cached listing of the source ERP directory
        self._inference_dir = None      # directory the current SAM2 state was built from
        
        # Add storage for each object ID's prompts
        self.object_prompts = {
            0: {
                0: {'points': [], 'labels': [], 'box': None}
            }
        }  # {obj_id: {'points': [], 'labels': [], 'box': None}}
        self.segmented_objects = set()

        # Frame cache
        self.frame_cache = {}
        self.max_cache_size = 2000  # Maximum number of cached frames
        
        # Animation variables
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animation_step)
        self.animation_playing = False
        self._animation_last_update = 0
        self.animation_interval = 1  # Show every Nth frame (default: show every frame)
        self.animation_counter = 0
        
        # Setup UI
        self.init_ui()
        
        # Setup SAM2 model
        self.setup_sam2()
        
    def setup_sam2(self):
        try:
            # Ask user for the checkpoint file location if not found
            sam2_checkpoint = "./segment-anything-2/checkpoints/sam2.1_hiera_large.pt"
            while not os.path.exists(sam2_checkpoint):
                sam2_checkpoint = QFileDialog.getOpenFileName(
                    self, "Select SAM2 Checkpoint File", "", "Model Files (*.pt)")[0]
                if not sam2_checkpoint:  # User canceled
                    self.status_label.setText("SAM2 model loading canceled. Please restart application.")
                    return
            
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
            self.status_label.setText("SAM2 model loaded successfully")
            print("loaded SAM2 model")
        except Exception as e:
            print(str(e))
            self.status_label.setText(f"Error loading SAM2 model: {str(e)}")
            print("failed to load SAM2 model")
    
    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Left panel for controls - increase width
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # Input file path section
        path_group = QGroupBox("File Path Settings")
        path_layout = QVBoxLayout()
        
        self.path_input = QLineEdit()
        self.path_button = QPushButton("Browse...")
        self.path_button.clicked.connect(self.browse_directory)
        
        path_layout.addWidget(QLabel("Input Directory:"))
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.path_button)
        
        self.save_path_input = QLineEdit()
        path_layout.addWidget(QLabel("Save Subfolder: (default: masks)"))
        path_layout.addWidget(self.save_path_input)
        
        self.erp_rotate_checkbox = QCheckBox("Rotate bottom pole to equator (360 ERP)")
        self.erp_rotate_checkbox.setToolTip(
            "Load the sequence with the bottom pole rotated to the image center, "
            "so pole-area objects (camera operator, stick, cart) can be annotated "
            "without ERP distortion. Masks are rotated back to the original ERP "
            "coordinates and cleaned (fill holes + keep largest component) when saved."
        )
        self.erp_rotate_checkbox.toggled.connect(self.on_erp_rotate_toggled)
        path_layout.addWidget(self.erp_rotate_checkbox)
        
        path_group.setLayout(path_layout)
        left_layout.addWidget(path_group)
        
        # Frame navigation section
        frame_group = QGroupBox("Frame Navigation")
        frame_layout = QVBoxLayout()
        
        frame_buttons = QHBoxLayout()
        self.prev_button = QPushButton("Previous Frame (r)")
        self.prev_button.clicked.connect(self.prev_frame)
        self.next_button = QPushButton("Next Frame (f)")
        self.next_button.clicked.connect(self.next_frame)
        frame_buttons.addWidget(self.prev_button)
        frame_buttons.addWidget(self.next_button)
        
        frame_jump = QHBoxLayout()
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setMinimum(0)
        self.frame_spinbox.setMaximum(0)  # Will be updated when loading data
        self.frame_spinbox.valueChanged.connect(self.jump_to_frame)
        self.jump_button = QPushButton("Jump")
        self.jump_button.clicked.connect(self.jump_to_frame)
        frame_jump.addWidget(QLabel("Frame:"))
        frame_jump.addWidget(self.frame_spinbox)
        frame_jump.addWidget(self.jump_button)
        
        # Animation controls
        animation_layout = QHBoxLayout()
        self.animation_button = QPushButton("Animation")
        self.animation_button.clicked.connect(self.toggle_animation)
        
        # Change FPS to interval (show every Nth frame)
        self.interval_spinbox = QSpinBox()
        self.interval_spinbox.setMinimum(1)
        self.interval_spinbox.setMaximum(30)
        self.interval_spinbox.setValue(1)  # Default: show every frame
        self.interval_spinbox.valueChanged.connect(self.update_interval)
        
        animation_layout.addWidget(self.animation_button)
        animation_layout.addWidget(QLabel("Interval:"))
        animation_layout.addWidget(self.interval_spinbox)
        
        frame_layout.addLayout(frame_buttons)
        frame_layout.addLayout(frame_jump)
        frame_layout.addLayout(animation_layout)
        frame_group.setLayout(frame_layout)
        left_layout.addWidget(frame_group)
        
        # Annotation tools section
        annotation_group = QGroupBox("Annotation Tools")
        annotation_layout = QVBoxLayout()
        
        # Object ID input
        obj_id_layout = QHBoxLayout()
        obj_id_layout.addWidget(QLabel("Object ID:"))
        self.obj_id_spinbox = QSpinBox()
        self.obj_id_spinbox.setMinimum(0)  # Starting from 0 instead of 1
        self.obj_id_spinbox.setMaximum(100)
        self.obj_id_spinbox.setValue(0)  # Default to 0
        self.obj_id_spinbox.valueChanged.connect(self.update_object_id)
        obj_id_layout.addWidget(self.obj_id_spinbox)
        annotation_layout.addLayout(obj_id_layout)
        
        # Point selection mode
        point_mode_layout = QHBoxLayout()
        self.pos_point_radio = QRadioButton("Positive Points")
        self.neg_point_radio = QRadioButton("Negative Points")
        self.pos_point_radio.setChecked(True)
        self.pos_point_radio.toggled.connect(self.set_point_mode)
        point_mode_layout.addWidget(self.pos_point_radio)
        point_mode_layout.addWidget(self.neg_point_radio)
        annotation_layout.addLayout(point_mode_layout)
        
        # Add point button
        self.add_point_button = QPushButton("Add Point (a)")
        self.add_point_button.clicked.connect(self.enable_point_selection)
        annotation_layout.addWidget(self.add_point_button)
        
        # Add annotation buttons
        self.add_bbox_button = QPushButton("Add Bounding Box")
        self.add_bbox_button.clicked.connect(self.enable_bbox_selection)
        annotation_layout.addWidget(self.add_bbox_button)
        
        self.clear_annotations_button = QPushButton("Clear Annotations")
        self.clear_annotations_button.clicked.connect(self.clear_annotations)
        annotation_layout.addWidget(self.clear_annotations_button)
        
        # Reset state button
        self.reset_state_button = QPushButton("Reset State")
        self.reset_state_button.clicked.connect(self.reset_state)
        annotation_layout.addWidget(self.reset_state_button)
        
        annotation_group.setLayout(annotation_layout)
        left_layout.addWidget(annotation_group)
        
        # Segmentation section
        segment_group = QGroupBox("Segmentation")
        segment_layout = QVBoxLayout()
        
        self.segment_current_button = QPushButton("Segment Current Frame (s)")
        self.segment_current_button.clicked.connect(self.segment_current_frame)
        segment_layout.addWidget(self.segment_current_button)
        
        self.segment_all_button = QPushButton("Segment All Frames")
        self.segment_all_button.clicked.connect(self.segment_all_frames)
        segment_layout.addWidget(self.segment_all_button)
        
        self.segment_current_obj_all_button = QPushButton("Segment Current Object in All Frames")
        self.segment_current_obj_all_button.clicked.connect(self.segment_current_object_all_frames)
        segment_layout.addWidget(self.segment_current_obj_all_button)
        
        self.forward_only_checkbox = QCheckBox("Only segment forward from current frame")
        self.forward_only_checkbox.setToolTip(
            "Propagate only from the current frame to the last frame. "
            "Masks already computed for earlier frames are kept."
        )
        segment_layout.addWidget(self.forward_only_checkbox)
        
        self.clear_current_obj_button = QPushButton("Clear Current Object From Current Frame")
        self.clear_current_obj_button.setToolTip(
            "Delete the current object's masks and prompts on this frame and all later frames. "
            "Data on earlier frames is kept."
        )
        self.clear_current_obj_button.clicked.connect(self.clear_current_object_from_current_frame)
        segment_layout.addWidget(self.clear_current_obj_button)
        
        segment_group.setLayout(segment_layout)
        left_layout.addWidget(segment_group)
        
        # Save section
        save_group = QGroupBox("Save Results")
        save_layout = QVBoxLayout()
        
        self.save_button = QPushButton("Save Masks")
        self.save_button.clicked.connect(self.save_masks)
        save_layout.addWidget(self.save_button)
        
        self.save_binary_button = QPushButton("Save Binary Masks")
        self.save_binary_button.clicked.connect(self.save_binary_masks)
        save_layout.addWidget(self.save_binary_button)
        
        self.save_current_obj_binary_button = QPushButton("Save Current Object Binary Masks")
        self.save_current_obj_binary_button.clicked.connect(self.save_current_obj_binary_masks)
        save_layout.addWidget(self.save_current_obj_binary_button)
        
        save_group.setLayout(save_layout)
        left_layout.addWidget(save_group)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)  # Allow word wrap for status messages
        left_layout.addWidget(self.status_label)
        
        left_panel.setLayout(left_layout)
        left_panel.setFixedWidth(600)  # Increase width from 300 to 400
        
        # Right panel for image display
        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.canvas)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Keyboard shortcuts
        QShortcut(QKeySequence(Qt.Key_R), self).activated.connect(self.prev_frame)
        QShortcut(QKeySequence(Qt.Key_F), self).activated.connect(self.next_frame)
        QShortcut(QKeySequence(Qt.Key_A), self).activated.connect(self.enable_point_selection)
        QShortcut(QKeySequence(Qt.Key_S), self).activated.connect(self.segment_current_frame)
        
        # Initialize UI state
        self.update_ui_state()
    
    def update_ui_state(self):
        has_data = len(self.frame_names) > 0
        
        # Update frame navigation controls
        self.prev_button.setEnabled(has_data and self.current_frame_idx > 0 and not self.animation_playing)
        self.next_button.setEnabled(has_data and self.current_frame_idx < len(self.frame_names) - 1 and not self.animation_playing)
        self.frame_spinbox.setEnabled(has_data and not self.animation_playing)
        self.jump_button.setEnabled(has_data and not self.animation_playing)
        self.animation_button.setEnabled(has_data)
        self.interval_spinbox.setEnabled(has_data)
        
        # Update animation button text
        if self.animation_playing:
            self.animation_button.setText("Stop")
        else:
            self.animation_button.setText("Animation")
        
        # Update annotation controls
        annotation_enabled = has_data and self.inference_state is not None and not self.animation_playing
        self.add_point_button.setEnabled(annotation_enabled)
        self.add_bbox_button.setEnabled(annotation_enabled)
        self.pos_point_radio.setEnabled(annotation_enabled)
        self.neg_point_radio.setEnabled(annotation_enabled)
        self.obj_id_spinbox.setEnabled(annotation_enabled)
        has_annotations = False
        if self.ann_obj_id in self.object_prompts:
            if self.current_frame_idx in self.object_prompts[self.ann_obj_id]:
                has_annotations = len(self.object_prompts[self.ann_obj_id][self.current_frame_idx]['points']) > 0 or self.object_prompts[self.ann_obj_id][self.current_frame_idx]['box'] is not None
        self.clear_annotations_button.setEnabled(annotation_enabled and has_annotations)
        self.reset_state_button.setEnabled(annotation_enabled)
        
        # Update segmentation controls
        self.segment_current_button.setEnabled(annotation_enabled and has_annotations)
        self.segment_all_button.setEnabled(annotation_enabled and has_annotations)
        self.segment_current_obj_all_button.setEnabled(annotation_enabled and has_annotations)
        self.forward_only_checkbox.setEnabled(annotation_enabled)
        
        # Update save controls
        has_segments = len(self.video_segments) > 0
        has_current_obj_segments = any(self.ann_obj_id in obj_masks for obj_masks in self.video_segments.values())
        has_current_obj_segments_from_current = any(
            self.ann_obj_id in obj_masks
            for frame_idx, obj_masks in self.video_segments.items()
            if frame_idx >= self.current_frame_idx
        )
        has_current_obj_prompts_from_current = any(
            frame_idx >= self.current_frame_idx
            and (prompts['box'] is not None or len(prompts['points']) > 0)
            for frame_idx, prompts in self.object_prompts.get(self.ann_obj_id, {}).items()
        )
        self.save_button.setEnabled(has_segments and not self.animation_playing)
        self.save_binary_button.setEnabled(has_segments and not self.animation_playing)
        self.save_current_obj_binary_button.setEnabled(has_current_obj_segments and not self.animation_playing)
        self.clear_current_obj_button.setEnabled(
            (has_current_obj_segments_from_current or has_current_obj_prompts_from_current)
            and not self.animation_playing
        )
    
    def update_object_id(self):
        self.ann_obj_id = self.obj_id_spinbox.value()
        if self.ann_obj_id not in self.object_prompts:
            self.object_prompts[self.ann_obj_id] = {}
            self.object_prompts[self.ann_obj_id][self.current_frame_idx] = {}
            self.object_prompts[self.ann_obj_id][self.current_frame_idx]['points'] = []
            self.object_prompts[self.ann_obj_id][self.current_frame_idx]['labels'] = []
            self.object_prompts[self.ann_obj_id][self.current_frame_idx]['box'] = None
        self.status_label.setText(f"Object ID set to {self.ann_obj_id}")
        self.display_current_frame()  # Refresh display to show masks for the selected object ID
    
    def set_point_mode(self):
        if self.pos_point_radio.isChecked():
            self.point_mode = "positive"
        else:
            self.point_mode = "negative"
    
    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory with Video Frames")
        if directory:
            self.path_input.setText(directory)
            self.load_frames(directory)
    
    def on_erp_rotate_toggled(self, checked):
        if self.erp_rotating:
            return
        # Reloading the sequence discards prompts and masks, so confirm first
        source = self.data_dir or self.path_input.text().strip()
        if source and os.path.isdir(source) and self._has_annotation_work():
            what = "enable" if checked else "disable"
            answer = QMessageBox.question(
                self, "Reload sequence?",
                f"To {what} ERP pole rotation the sequence will be reloaded, "
                "which discards all current points and masks.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                self.erp_rotate_checkbox.blockSignals(True)
                self.erp_rotate_checkbox.setChecked(not checked)
                self.erp_rotate_checkbox.blockSignals(False)
                return
        
        if checked:
            self.erp_pole = 'bottom'
            self.status_label.setText("ERP pole rotation enabled: bottom pole to equator")
        else:
            self.erp_pole = None
            self.status_label.setText("ERP pole rotation disabled")
        # Reload the current directory so the new view takes effect
        if source and os.path.isdir(source):
            if self.erp_pole:
                self.path_input.setText(source)
            self.load_frames(source)
    
    def _has_annotation_work(self):
        """True if the session holds points, boxes or masks that a reload would discard."""
        if self.video_segments:
            return True
        for obj_prompts in self.object_prompts.values():
            for prompts in obj_prompts.values():
                if prompts.get('box') is not None or prompts.get('points'):
                    return True
        return False
    
    def _erp_rotated_cache_dir(self, source_dir):
        """Sibling directory holding the pole-rotated frames of source_dir."""
        return os.path.normpath(source_dir) + ERP_ROTATED_SUFFIX
    
    def _build_erp_rotated_frames(self, source_dir, cache_dir):
        """Rotate every frame of source_dir into cache_dir (JPEG, original stems)."""
        self.erp_rotating = True
        self.erp_rotate_checkbox.setEnabled(False)
        try:
            target_lat = pole_to_target_lat(self.erp_pole)
            total = 0
    
            def report(done, total_count):
                nonlocal total
                total = total_count
                if done == total_count or done % 25 == 0:
                    self.status_label.setText(
                        f"Rotating frames (pole {self.erp_pole} to equator): "
                        f"{done}/{total_count}..."
                    )
                    QApplication.processEvents()
    
            rotate_erp_frames_to_dir(source_dir, cache_dir, target_lat,
                                     device=str(self.device), progress=report)
            self.erp_caches[self.erp_pole] = (cache_dir, source_dir)
            self.status_label.setText(
                f"Rotated {total} frames to ERP pole-centered view"
            )
            return True
        except Exception as e:
            self.status_label.setText(f"Error rotating ERP frames: {str(e)}")
            return False
        finally:
            self.erp_rotating = False
            self.erp_rotate_checkbox.setEnabled(True)
    
    def _has_erp_cache(self, pole, cache_dir, source_dir):
        """True if a cache built earlier this session still matches pole and source."""
        entry = self.erp_caches.get(pole) if pole else None
        return bool(entry) and entry == (cache_dir, source_dir)
    
    @staticmethod
    def _remove_erp_cache(cache_dir):
        """Delete a rotated-frame cache directory (never touches the source directory)."""
        try:
            shutil.rmtree(cache_dir)
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"Could not remove rotated frame cache {cache_dir}: {e}")
    
    def _detach_video_segments(self):
        """Copy numpy mask arrays out of view of the soon-to-be-replaced inference state."""
        self.video_segments = {
            frame_idx: {obj_id: np.array(mask, copy=True) for obj_id, mask in obj_masks.items()}
            for frame_idx, obj_masks in self.video_segments.items()
        }
    
    def _release_inference_state(self):
        """Free the GPU memory held by the current SAM2 inference state.
    
        init_state() encodes every frame onto the GPU, so the old state must be
        dropped before a new one is created, otherwise the peak doubles and can
        run out of memory. Safe to call when no state exists.
        """
        old_state = self.inference_state
        if old_state is None:
            return
        self._detach_video_segments()
        self.inference_state = None
        self._inference_dir = None
        try:
            self.predictor.reset_state(old_state)
        except Exception as e:
            print(f"Could not reset previous inference state: {e}")
        del old_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def load_frames(self, directory, reload_view=True):
        """Load a frame sequence.
        
        reload_view=False keeps the current view directory (i.e. the images are
        identical to what is already loaded), skipping any GPU work.
        """
        try:
            directory = os.path.normpath(directory)
            active_dir = directory
            cache_dir = None
            if self.erp_pole:
                cache_dir = self._erp_rotated_cache_dir(directory)
                if self._has_erp_cache(self.erp_pole, cache_dir, directory):
                    active_dir = cache_dir  # built earlier this session, reuse it
                else:
                    # Switching source/pole: drop whatever this pole had cached before
                    old_entry = self.erp_caches.get(self.erp_pole)
                    if old_entry:
                        self._remove_erp_cache(old_entry[0])
                    self._remove_erp_cache(cache_dir)  # clear leftovers from a previous run
                    if not self._build_erp_rotated_frames(directory, cache_dir):
                        return
                    active_dir = cache_dir

            if not reload_view and active_dir == self.active_dir:
                return  # nothing to do, the same frames are already loaded

            self.data_dir = directory
            self.path_input.setText(directory)
            # Get only image files
            new_frame_names = [
                p for p in os.listdir(active_dir)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
            ]
            
            if not new_frame_names:
                self.status_label.setText("No image files found in the selected directory")
                return
            
            # The new sequence is usable only now, so commit the rotation state here
            same_sequence = (active_dir == self._inference_dir
                             and sorted(new_frame_names) == sorted(self.frame_names))
            self.frame_names = new_frame_names
            self.active_dir = active_dir
            self.erp_cache_dir = cache_dir
            self._source_files = None
            
            # Sort frames by number
            self.frame_names.sort()
            
            # Initialize segmentation state
            self.current_frame_idx = 0
            self.frame_spinbox.setMaximum(len(self.frame_names) - 1)
            self.frame_spinbox.setValue(0)
            
            # Clear frame cache
            self.frame_cache = {}
            
            # Initialize inference state. The previous state holds all previously
            # loaded frames on the GPU, so drop it before loading the new sequence,
            # otherwise both sets of encoded frames coexist and double the peak.
            if self.predictor:
                if same_sequence and self.inference_state is not None:
                    print(f"SAM2 state kept: {len(self.frame_names)} frames already loaded")
                else:
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    self._release_inference_state()
                    new_state = self.predictor.init_state(video_path=self.active_dir)
                    self.inference_state = new_state
                    self._inference_dir = active_dir
                    self.predictor.reset_state(new_state)
                    if torch.cuda.is_available():
                        print(f"SAM2 state loaded: {len(self.frame_names)} frames, "
                              f"GPU peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB "
                              f"(reserved {torch.cuda.memory_reserved() / 1e9:.2f} GB)")
            self.ann_obj_id = 0  # Start from 0
            self.obj_id_spinbox.setValue(0)
            self.object_prompts = {0: {0: {'points': [], 'labels': [], 'box': None}}}
            self.segmented_objects = set()
            self.video_segments = {}
            
            # Preload first few frames to cache
            self.preload_frames(0, len(self.frame_names))
            
            # Display first frame
            self.display_current_frame()
            if self.erp_pole:
                self.status_label.setText(
                    f"Loaded {len(self.frame_names)} frames (bottom pole rotated to equator); "
                    "masks will be rotated back to ERP coordinates on save"
                )
            else:
                self.status_label.setText(f"Loaded {len(self.frame_names)} frames")
            
        except Exception as e:
            self.status_label.setText(f"Error loading frames: {str(e)}")
        
        self.update_ui_state()
    
    def preload_frames(self, start_idx, end_idx):
        """Preload specified range of frames to cache"""
        for idx in range(start_idx, min(end_idx, len(self.frame_names))):
            if idx not in self.frame_cache:
                try:
                    frame_path = os.path.join(self.active_dir, self.frame_names[idx])
                    img = Image.open(frame_path)
                    self.frame_cache[idx] = img
                    
                    # Limit cache size
                    if len(self.frame_cache) > self.max_cache_size:
                        # Remove frames furthest from current frame
                        cache_keys = list(self.frame_cache.keys())
                        cache_keys.sort(key=lambda x: abs(x - self.current_frame_idx))
                        del self.frame_cache[cache_keys[-1]]
                except Exception as e:
                    print(f"Error preloading frame {idx}: {e}")
    
    def get_frame_from_cache(self, idx):
        """Get frame from cache, load if not present"""
        if idx not in self.frame_cache:
            try:
                frame_path = os.path.join(self.active_dir, self.frame_names[idx])
                img = Image.open(frame_path)
                self.frame_cache[idx] = img
                
                # Limit cache size
                if len(self.frame_cache) > self.max_cache_size:
                    cache_keys = list(self.frame_cache.keys())
                    cache_keys.sort(key=lambda x: abs(x - self.current_frame_idx))
                    del self.frame_cache[cache_keys[-1]]
            except Exception as e:
                print(f"Error loading frame {idx}: {e}")
                return None
        
        return self.frame_cache[idx]
    
    def get_source_frame(self, idx):
        """Load a frame from the original ERP directory (for mask visualization after rotate-back).

        The source directory may use different extensions than the rotated cache,
        so match frames by position in the sorted listing instead of by name.
        """
        try:
            if self._source_files is None:
                self._source_files = [
                    p for p in sorted(os.listdir(self.data_dir))
                    if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
                ]
            if idx >= len(self._source_files):
                return None
            return Image.open(os.path.join(self.data_dir, self._source_files[idx]))
        except Exception as e:
            print(f"Error loading source frame {idx}: {e}")
            return None
    
    def display_current_frame(self):
        if not self.frame_names or self.current_frame_idx >= len(self.frame_names):
            return
        
        # Get frame from cache or load it
        img = self.get_frame_from_cache(self.current_frame_idx)
        if img is None:
            return
        
        # Clear axis and display current frame
        self.canvas.axes.clear()
        
        # Disable axis ticks to improve performance
        self.canvas.axes.set_xticks([])
        self.canvas.axes.set_yticks([])
        
        # Display image
        self.canvas.axes.imshow(img)
        self.canvas.axes.set_title(f"Frame {self.current_frame_idx}: {self.frame_names[self.current_frame_idx]}", fontsize=10)
        
        # Optimize display in animation mode
        if self.animation_playing:
            # Only show mask in animation mode, not points or box
            if self.current_frame_idx in self.video_segments:
                for obj_id, mask in self.video_segments[self.current_frame_idx].items():
                    self.show_mask(mask, self.canvas.axes, obj_id=obj_id)
        else:
            # Show all elements in non-animation mode
            for obj_id, prompts in self.object_prompts.items():
                if self.current_frame_idx not in prompts:
                    continue
                else:
                    prompts = prompts[self.current_frame_idx]
                points, labels, box = prompts['points'], prompts['labels'], prompts['box']
            
                if points:
                    points_array = np.array(points)
                    labels_array = np.array(labels)
                    self.show_points(points_array, labels_array, self.canvas.axes)
            
                if box is not None:
                    self.show_box(box, self.canvas.axes)
            
            if self.current_frame_idx in self.video_segments:
                for obj_id, mask in self.video_segments[self.current_frame_idx].items():
                    self.show_mask(mask, self.canvas.axes, obj_id=obj_id)
        
        # Draw image
        self.canvas.draw()
        
        # Preload next frames
        if not self.animation_playing:
            next_idx = self.current_frame_idx + 1
            if next_idx < len(self.frame_names) and next_idx not in self.frame_cache:
                self.preload_frames(next_idx, next_idx + 3)
    
    def prev_frame(self):
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.frame_spinbox.setValue(self.current_frame_idx)
            self.display_current_frame()
            self.update_ui_state()
    
    def next_frame(self):
        if self.current_frame_idx < len(self.frame_names) - 1:
            self.current_frame_idx += 1
            self.frame_spinbox.setValue(self.current_frame_idx)
            self.display_current_frame()
            self.update_ui_state()
    
    def jump_to_frame(self):
        frame_idx = self.frame_spinbox.value()
        if 0 <= frame_idx < len(self.frame_names):
            self.current_frame_idx = frame_idx
            self.display_current_frame()
            self.update_ui_state()
    
    def toggle_animation(self):
        if self.animation_playing:
            # Stop animation
            self.animation_timer.stop()
            self.animation_playing = False
        else:
            # Preload some frames to cache before starting
            for i in range(self.current_frame_idx, min(self.current_frame_idx + 15, len(self.frame_names))):
                self.get_frame_from_cache(i)
            
            # Start animation
            self.animation_timer.start(30)  # Fixed 30ms timeout for consistent UI updates
            self.animation_playing = True
            self._animation_last_update = time.time() * 1000
            self.animation_counter = 0
        
        self.update_ui_state()
    
    def update_interval(self):
        # Update animation interval (show every Nth frame)
        self.animation_interval = self.interval_spinbox.value()
    
    def animation_step(self):
        # Skip frames based on interval setting
        self.animation_counter += 1
        
        # Move to next frame or loop
        if self.current_frame_idx < len(self.frame_names) - 1:
            self.current_frame_idx += 1
        else:
            # Loop back to first frame
            self.current_frame_idx = 0
        
        if self.current_frame_idx % self.animation_interval != 0:
            return
        # Preload next few frames
        next_idx = (self.current_frame_idx + 1) % len(self.frame_names)
        if next_idx not in self.frame_cache:
            QApplication.processEvents()  # Allow UI updates
            self.get_frame_from_cache(next_idx)
        
        # Update spinbox without triggering signals
        self.frame_spinbox.blockSignals(True)
        self.frame_spinbox.setValue(self.current_frame_idx)
        self.frame_spinbox.blockSignals(False)
        
        # Display frame
        self.display_current_frame()
    
    def enable_point_selection(self):
        self.add_point_mode = True
        self.bbox_selection_active = False
        point_type = "positive" if self.point_mode == "positive" else "negative"
        self.status_label.setText(f"Click on the image to add a {point_type} point")
    
    def on_canvas_click(self, event):
        if event.xdata is None or event.ydata is None or not self.inference_state or self.animation_playing:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        if self.current_frame_idx not in self.object_prompts[self.ann_obj_id]:
            self.object_prompts[self.ann_obj_id][self.current_frame_idx] = {'points': [], 'labels': [], 'box': None}
        if hasattr(self, 'bbox_selection_active') and self.bbox_selection_active:
            # Bounding box selection mode
            if not hasattr(self, 'bbox_start'):
                # First click - start of bounding box
                self.bbox_start = (x, y)
                self.status_label.setText(f"Bounding box started at ({x}, {y}). Click again to complete.")
            else:
                # Second click - end of bounding box
                x1, y1 = self.bbox_start
                x2, y2 = x, y
                self.object_prompts[self.ann_obj_id][self.current_frame_idx]['box'] = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], dtype=np.float32)
                self.bbox_selection_active = False
                delattr(self, 'bbox_start')
                self.status_label.setText(f"Bounding box added: {self.object_prompts[self.ann_obj_id][self.current_frame_idx]['box']}")
                self.display_current_frame()
        elif self.add_point_mode:
            self.object_prompts[self.ann_obj_id][self.current_frame_idx]['points'].append([x, y])
            label = 1 if self.point_mode == "positive" else 0
            self.object_prompts[self.ann_obj_id][self.current_frame_idx]['labels'].append(label)
            point_type = "positive" if label == 1 else "negative"
            self.status_label.setText(f"Added {point_type} point at ({x}, {y})")
            self.add_point_mode = False  # Turn off point mode after adding a point
            self.display_current_frame()
        
        self.update_ui_state()
    
    def enable_bbox_selection(self):
        self.bbox_selection_active = True
        self.add_point_mode = False
        self.status_label.setText("Click and drag to define a bounding box")
    
    def clear_annotations(self):
        self.object_prompts[self.ann_obj_id][self.current_frame_idx]['points'] = []
        self.object_prompts[self.ann_obj_id][self.current_frame_idx]['labels'] = []
        self.object_prompts[self.ann_obj_id][self.current_frame_idx]['box'] = None
        self.status_label.setText("Annotations cleared")
        self.display_current_frame()
        self.update_ui_state()
    
    def reset_state(self):
        if not self.inference_state or not self.predictor:
            return
        
        try:
            self.predictor.reset_state(self.inference_state)
            self.ann_obj_id = 0  # Start from 0
            self.obj_id_spinbox.setValue(0)
            self.object_prompts = {
                0: {
                    0: {'points': [], 'labels': [], 'box': None}
                }
            }
            self.segmented_objects = set()
            self.video_segments = {}
            self.status_label.setText("State reset successfully")
            self.current_frame_idx = 0
            self.frame_spinbox.setValue(0)
            self.display_current_frame()
        except Exception as e:
            self.status_label.setText(f"Error resetting state: {str(e)}")
        
        self.update_ui_state()
    
    def _segment_object(self, obj_id):
        """Run SAM2 inference for a single object on the current frame."""
        prompts = self.object_prompts[obj_id].get(self.current_frame_idx, None)
        if prompts is None:
            return False
        self.segmented_objects.add(obj_id)
        if prompts['box'] is None and len(prompts['points']) == 0:
            return False
        
        # Apply segmentation
        points_array = np.array(prompts['points'], dtype=np.float32) if prompts['points'] else None
        labels_array = np.array(prompts['labels'], dtype=np.int32) if prompts['labels'] else None
        
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=self.current_frame_idx,
            obj_id=obj_id,
            points=points_array,
            labels=labels_array,
            box=prompts['box'],
        )
        
        # Map client obj_id to the index in the returned logits tensor
        obj_idx = out_obj_ids.index(obj_id)
        mask = (out_mask_logits[obj_idx] > 0.0).cpu().numpy()
        if mask.ndim > 2:
            mask = mask.squeeze()
            if mask.ndim == 1:
                mask = mask[np.newaxis, :]
        if self.current_frame_idx not in self.video_segments:
            self.video_segments[self.current_frame_idx] = {}
        self.video_segments[self.current_frame_idx][obj_id] = mask
        return True
    
    def segment_current_frame(self):
        if not self.inference_state or not self.object_prompts:
            return
        
        try:
            if self.ann_obj_id in self.object_prompts:
                if self._segment_object(self.ann_obj_id):
                    self.display_current_frame()
                    self.status_label.setText(
                        f"Segmentation completed for object {self.ann_obj_id} in frame {self.current_frame_idx}"
                    )
                else:
                    self.status_label.setText(
                        f"No prompts for object {self.ann_obj_id} in frame {self.current_frame_idx}"
                    )
            else:
                self.status_label.setText(f"Object {self.ann_obj_id} has no prompts")
            
        except Exception as e:
            self.status_label.setText(f"Segmentation error: {str(e)}")
        
        self.update_ui_state()
    
    def segment_all_frames(self):
        if not self.inference_state or not self.object_prompts:
            return
        
        try:
            # First, add prompts for all objects on the current frame
            segmented_any = False
            for obj_id in self.object_prompts:
                if self._segment_object(obj_id):
                    segmented_any = True
            
            if not segmented_any:
                self.status_label.setText("No prompts to propagate")
                return
            
            forward_only = self.forward_only_checkbox.isChecked()
            start_frame_idx = self.current_frame_idx if forward_only else None
            
            # Then propagate to all frames (or only forward from the current frame)
            if forward_only:
                self.status_label.setText(
                    f"Propagating segmentation from frame {start_frame_idx} to the end..."
                )
                # Keep masks already computed for earlier frames
                self.video_segments = {
                    frame_idx: obj_masks
                    for frame_idx, obj_masks in self.video_segments.items()
                    if frame_idx < start_frame_idx
                }
            else:
                self.status_label.setText("Propagating segmentation to all frames...")
                self.video_segments = {}  # Reset segments
            QApplication.processEvents()  # Update UI
            
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.inference_state, start_frame_idx=start_frame_idx
            ):
                self.video_segments[out_frame_idx] = {}
                for i, out_obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                    if mask.ndim > 2:
                        mask = mask.squeeze()
                        if mask.ndim == 1:
                            mask = mask[np.newaxis, :]
                    self.video_segments[out_frame_idx][out_obj_id] = mask
                
                # Update UI occasionally to show progress
                if out_frame_idx % 10 == 0:
                    self.status_label.setText(f"Processed frame {out_frame_idx}/{len(self.frame_names)}")
                    QApplication.processEvents()
            
            if forward_only:
                self.status_label.setText(
                    f"Segmentation completed for frames {start_frame_idx}-{len(self.frame_names) - 1}"
                )
            else:
                self.status_label.setText(f"Segmentation completed for all {len(self.frame_names)} frames")
            self.display_current_frame()
            
        except Exception as e:
            self.status_label.setText(f"Segmentation error: {str(e)}")
        
        self.update_ui_state()
    
    def segment_current_object_all_frames(self):
        """Propagate only the current active object to all frames."""
        if not self.inference_state or self.ann_obj_id not in self.object_prompts:
            return
        
        current_obj_id = self.ann_obj_id
        current_prompts = self.object_prompts[current_obj_id]
        
        forward_only = self.forward_only_checkbox.isChecked()
        start_frame_idx = self.current_frame_idx if forward_only else None
        
        # Check if current object has any prompts within the propagation range
        has_prompt = False
        for frame_idx, prompts in current_prompts.items():
            if forward_only and frame_idx < start_frame_idx:
                continue
            if prompts['box'] is not None or len(prompts['points']) > 0:
                has_prompt = True
                break
        
        if not has_prompt:
            if forward_only:
                self.status_label.setText(
                    f"Object {current_obj_id} has no prompts at or after frame {start_frame_idx}"
                )
            else:
                self.status_label.setText(f"Object {current_obj_id} has no prompts")
            return
        
        try:
            # Save masks we must restore after propagation: other objects everywhere,
            # plus the current object on frames before the propagation start
            saved_video_segments = {}
            for frame_idx, obj_masks in self.video_segments.items():
                saved_other_masks = {}
                for obj_id, mask in obj_masks.items():
                    if obj_id != current_obj_id:
                        saved_other_masks[obj_id] = mask.copy()
                    elif forward_only and frame_idx < start_frame_idx:
                        saved_other_masks[obj_id] = mask.copy()
                if saved_other_masks:
                    saved_video_segments[frame_idx] = saved_other_masks
            
            # Reset predictor state and re-add only current object
            self.predictor.reset_state(self.inference_state)
            
            # Add all prompts for the current object (only within the propagation range)
            for frame_idx, prompts in current_prompts.items():
                if forward_only and frame_idx < start_frame_idx:
                    continue
                if prompts['box'] is None and len(prompts['points']) == 0:
                    continue
                points_array = np.array(prompts['points'], dtype=np.float32) if prompts['points'] else None
                labels_array = np.array(prompts['labels'], dtype=np.int32) if prompts['labels'] else None
                
                self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=current_obj_id,
                    points=points_array,
                    labels=labels_array,
                    box=prompts['box'],
                )
            
            # Propagate current object to all frames (or only forward from the current frame)
            if forward_only:
                self.status_label.setText(
                    f"Propagating object {current_obj_id} from frame {start_frame_idx} to the end..."
                )
                # Keep masks already computed for earlier frames
                self.video_segments = {
                    frame_idx: obj_masks
                    for frame_idx, obj_masks in self.video_segments.items()
                    if frame_idx < start_frame_idx
                }
            else:
                self.status_label.setText(f"Propagating object {current_obj_id} to all frames...")
                self.video_segments = {}
            QApplication.processEvents()
            
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.inference_state, start_frame_idx=start_frame_idx
            ):
                self.video_segments[out_frame_idx] = {}
                for i, out_obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                    if mask.ndim > 2:
                        mask = mask.squeeze()
                        if mask.ndim == 1:
                            mask = mask[np.newaxis, :]
                    self.video_segments[out_frame_idx][out_obj_id] = mask
                
                if out_frame_idx % 10 == 0:
                    self.status_label.setText(f"Processed frame {out_frame_idx}/{len(self.frame_names)}")
                    QApplication.processEvents()
            
            # Merge saved masks back into video_segments
            for frame_idx, obj_masks in saved_video_segments.items():
                if frame_idx not in self.video_segments:
                    self.video_segments[frame_idx] = {}
                for obj_id, mask in obj_masks.items():
                    self.video_segments[frame_idx][obj_id] = mask
            
            if forward_only:
                self.status_label.setText(
                    f"Segmentation completed for object {current_obj_id} in frames "
                    f"{start_frame_idx}-{len(self.frame_names) - 1}"
                )
            else:
                self.status_label.setText(
                    f"Segmentation completed for object {current_obj_id} in all {len(self.frame_names)} frames"
                )
            self.display_current_frame()
            
        except Exception as e:
            self.status_label.setText(f"Segmentation error: {str(e)}")
        
        self.update_ui_state()
    
    def clear_current_object_from_current_frame(self):
        """Delete the current object's masks and prompts on the current frame and all later frames."""
        obj_id = self.ann_obj_id
        mask_frames = sorted(
            frame_idx
            for frame_idx, obj_masks in self.video_segments.items()
            if frame_idx >= self.current_frame_idx and obj_id in obj_masks
        )
        prompt_frames = [
            frame_idx
            for frame_idx, prompts in self.object_prompts.get(obj_id, {}).items()
            if frame_idx >= self.current_frame_idx
        ]
        non_empty_prompt_frames = [
            frame_idx for frame_idx in prompt_frames
            if self.object_prompts[obj_id][frame_idx]['box'] is not None
            or self.object_prompts[obj_id][frame_idx]['points']
        ]
        if not mask_frames and not non_empty_prompt_frames:
            self.status_label.setText(
                f"Object {obj_id} has no masks or prompts from frame {self.current_frame_idx} onward"
            )
            return
        
        details = []
        if mask_frames:
            details.append(f"{len(mask_frames)} masks")
        if non_empty_prompt_frames:
            details.append(f"{len(non_empty_prompt_frames)} prompt frames")
        details_text = " and ".join(details)
        first_frame = min(mask_frames + non_empty_prompt_frames)
        last_frame = max(mask_frames + non_empty_prompt_frames)
        
        answer = QMessageBox.question(
            self, "Clear object data?",
            f"Delete object {obj_id}'s {details_text} on frames {first_frame}-{last_frame}?\n\n"
            "Data on earlier frames is kept.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        
        for frame_idx in mask_frames:
            del self.video_segments[frame_idx][obj_id]
            if not self.video_segments[frame_idx]:
                del self.video_segments[frame_idx]
        for frame_idx in prompt_frames:
            del self.object_prompts[obj_id][frame_idx]
        
        self.status_label.setText(
            f"Cleared object {obj_id}'s {details_text} on frames {first_frame}-{last_frame}"
        )
        self.display_current_frame()
        self.update_ui_state()
    
    def _mask_for_saving(self, mask, frame_idx, obj_id):
        """Normalize a mask to a single (H, W) binary map, rotating it back to ERP
        coordinates when pole rotation is active."""
        mask = np.asarray(mask)
        if mask.ndim > 2:
            mask = mask.squeeze()
            if mask.ndim == 1:
                mask = mask[np.newaxis, :]
        if mask.ndim != 2:
            raise ValueError(f"unexpected mask shape for frame {frame_idx}, obj {obj_id}: {mask.shape}")

        binary = (mask > 0).astype(np.uint8)
        if not self.erp_pole:
            return binary

        return rotate_mask_back_to_erp(
            binary, target_lat=pole_to_target_lat(self.erp_pole),
            device=str(self.device), threshold=0.0, **ERP_MASK_CLEANUP,
        )
    
    def save_masks(self):
        if not self.video_segments:
            return
        
        # Get save path
        subfolder = self.save_path_input.text().strip()
        if not subfolder:
            subfolder = "masks"
        
        # Create save directory in parent of data_dir
        parent_dir = os.path.dirname(os.path.normpath(self.data_dir))
        save_dir = os.path.join(parent_dir, subfolder)
        os.makedirs(save_dir, exist_ok=True)
        if subfolder == "masks":
            vis_dir = os.path.join(parent_dir, "masks_vis")
        else:
            vis_dir = os.path.join(parent_dir, f"{subfolder}_vis")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Save masks as numpy arrays
        saved_count = 0
        pallete = np.random.randint(0, 256, (100, 3), dtype=np.uint8)
        for frame_idx, obj_masks in self.video_segments.items():
            frame_name = os.path.splitext(self.frame_names[frame_idx])[0]
            masks = []
            for obj_id, mask in obj_masks.items():
                masks.append(self._mask_for_saving(mask, frame_idx, obj_id))
            masks = np.stack(masks, axis=0) # (num_masks, H, W)
            save_path = os.path.join(save_dir, f"{frame_name}.npy")
            np.save(save_path, masks)

            # visualize the masks over the frame they belong to (rotated or original view)
            if self.erp_pole:
                img = self.get_source_frame(frame_idx)
                img = np.array(img) if img is not None else np.array(self.frame_cache[frame_idx])
            else:
                img = np.array(self.frame_cache[frame_idx])
            if masks.sum() == 0:
                vis_mask = img
            else:
                masks = np.concatenate([np.zeros((1, *masks.shape[1:])), masks], axis=0)
                masks = np.argmax(masks, axis=0).squeeze()
                mask_image = pallete[masks]
                if img.shape[-1] == 4:
                    mask_image = np.concatenate([mask_image, np.ones_like(mask_image[:, :, :1]) * 255], axis=-1)
                vis_mask = mask_image * 0.6 + img * 0.4
                vis_mask = vis_mask * (masks[..., None] > 0) + img * (masks[..., None] == 0)
            vis_path = os.path.join(vis_dir, f"{frame_name}.png")
            Image.fromarray(vis_mask.astype(np.uint8)).save(vis_path)

            saved_count += 1
            
            # Periodically update UI to show progress
            if saved_count % 20 == 0:
                self.status_label.setText(f"Saved {saved_count} masks...")
                QApplication.processEvents()
        
        self.status_label.setText(f"Saved {saved_count} masks to {save_dir}")
    
    def save_binary_masks(self):
        if not self.video_segments:
            return
        
        # Get save path
        subfolder = self.save_path_input.text().strip()
        if not subfolder:
            subfolder = "masks"
        
        # Create save directory in parent of data_dir
        parent_dir = os.path.dirname(os.path.normpath(self.data_dir))
        binary_save_dir = os.path.join(parent_dir, f"{subfolder}")
        os.makedirs(binary_save_dir, exist_ok=True)
        
        saved_count = 0
        for frame_idx, obj_masks in self.video_segments.items():
            frame_name = os.path.splitext(self.frame_names[frame_idx])[0]
            for obj_id, mask in obj_masks.items():
                try:
                    binary_mask = self._mask_for_saving(mask, frame_idx, obj_id) * 255
                except ValueError as e:
                    print(f"Warning: {e}")
                    continue
                save_path = os.path.join(binary_save_dir, f"{frame_name}_obj_{obj_id}.png")
                Image.fromarray(binary_mask).save(save_path)
                saved_count += 1
            
            # Periodically update UI to show progress
            if saved_count % 20 == 0:
                self.status_label.setText(f"Saved {saved_count} binary masks...")
                QApplication.processEvents()
        
        self.status_label.setText(f"Saved {saved_count} binary masks to {binary_save_dir}")
    
    def save_current_obj_binary_masks(self):
        if not self.video_segments:
            return
        
        # Get save path
        subfolder = self.save_path_input.text().strip()
        if not subfolder:
            subfolder = "masks"
        
        # Create save directory in parent of data_dir
        parent_dir = os.path.dirname(os.path.normpath(self.data_dir))
        binary_save_dir = os.path.join(parent_dir, f"{subfolder}")
        os.makedirs(binary_save_dir, exist_ok=True)
        
        current_obj_id = self.ann_obj_id
        saved_count = 0
        for frame_idx, obj_masks in self.video_segments.items():
            if current_obj_id not in obj_masks:
                continue
            frame_name = os.path.splitext(self.frame_names[frame_idx])[0]
            try:
                binary_mask = self._mask_for_saving(obj_masks[current_obj_id], frame_idx, current_obj_id) * 255
            except ValueError as e:
                print(f"Warning: {e}")
                continue
            save_path = os.path.join(binary_save_dir, f"{frame_name}_obj_{current_obj_id}.png")
            Image.fromarray(binary_mask).save(save_path)
            saved_count += 1
            
            # Periodically update UI to show progress
            if saved_count % 20 == 0:
                self.status_label.setText(f"Saved {saved_count} binary masks for object {current_obj_id}...")
                QApplication.processEvents()
        
        self.status_label.setText(f"Saved {saved_count} binary masks for object {current_obj_id} to {binary_save_dir}")
    
    
    # Helper functions to display masks, points, and boxes
    def show_mask(self, mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id % 10  # Ensure we don't exceed colormap range
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    def show_points(self, coords, labels, ax, marker_size=200):
        if len(coords) == 0:
            return
        
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        
        if len(pos_points) > 0:
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
                       s=marker_size, edgecolor='white', linewidth=1.25)
        
        if len(neg_points) > 0:
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
                       s=marker_size, edgecolor='white', linewidth=1.25)
        
        # Add text labels for position values
        for i, (x, y) in enumerate(pos_points):
            ax.text(x, y-10, f'+{i}, [{x}, {y}]', color='green', fontsize=16, ha='center', va='center')

        for i, (x, y) in enumerate(neg_points):
            ax.text(x, y-10, f'-{i}, [{x}, {y}]', color='red', fontsize=16, ha='center', va='center')

    
    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        
        # Add text labels for box values
        ax.text(x0, y0-10, f'box, [{int(x0)}, {int(y0)}, {int(box[2])}, {int(box[3])}]', 
                color='green', fontsize=16, ha='center', va='center')
    
    def closeEvent(self, event):
        # Stop animation timer when closing the application
        if self.animation_timer.isActive():
            self.animation_timer.stop()
        # Free the GPU frames held by SAM2 before shutting down
        self._release_inference_state()
        # Remove every rotated-frame cache this session created
        for cache_dir, _source in self.erp_caches.values():
            self._remove_erp_cache(cache_dir)
        self.erp_caches.clear()
        # Clean up resources
        self.frame_cache.clear()
        event.accept()

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(torch.__version__)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    app = QApplication(sys.argv)
    window = SAM2AnnotationTool()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()