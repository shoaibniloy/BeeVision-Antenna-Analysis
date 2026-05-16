import os
import tempfile
import sys
from pathlib import Path

# ── Fix: suppress Ubuntu font OpenType crash BEFORE Qt/matplotlib load ────────
# Qt's font system crashes on malformed OpenType fonts (Ubuntu, script 12).
# Force matplotlib to use a safe font and disable its font scanner.
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ['QT_QPA_FONTDIR'] = ''          # stop Qt scanning system font dirs early

# Silence fontconfig / FreeType warnings to stderr
import warnings
warnings.filterwarnings('ignore', message='.*OpenType.*')
warnings.filterwarnings('ignore', message='.*Glyph.*')
warnings.filterwarnings('ignore', message='.*findfont.*')

# Clear matplotlib font cache so it rebuilds cleanly without Ubuntu
import shutil, glob
_mpl_cache = os.path.join(os.path.expanduser('~'), '.cache', 'matplotlib')
for _f in glob.glob(os.path.join(_mpl_cache, 'fontlist*.json')):
    try:
        os.remove(_f)
    except OSError:
        pass
# ─────────────────────────────────────────────────────────────────────────────

import cv2
try:
    # Try to import extended image processing module for skeleton thinning
    import cv2.ximgproc as ximgproc
    XIMGPROC_AVAILABLE = True
except (ImportError, AttributeError):
    XIMGPROC_AVAILABLE = False
    print("[WARNING] cv2.ximgproc not available - using fallback skeleton method")
import numpy as np
from collections import deque
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QScrollArea,
    QLabel, QVBoxLayout, QHBoxLayout, QFileDialog, QGroupBox, QTabWidget,
    QFrame, QGridLayout, QSizePolicy, QSpacerItem, QProgressBar, QSlider,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QSpinBox, QDoubleSpinBox, QCheckBox
)
from PyQt6.QtCore import QThread, QObject, pyqtSignal, Qt, QPropertyAnimation, QEasingCurve, QSize
from PyQt6.QtGui import QColor, QPalette, QImage, QPixmap, QFont
from ultralytics import YOLO
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict  # ADD THIS LINE
from dataclasses import dataclass     # ADD THIS LINE
from collections import deque
import time
import json
import csv
import io

try:
    import matplotlib
    matplotlib.use('Agg')

    # Redirect stderr during font manager init to suppress Ubuntu OpenType crash
    import io as _mpl_io, contextlib as _mpl_ctx
    with _mpl_ctx.redirect_stderr(_mpl_io.StringIO()):
        import matplotlib.font_manager as _fm
        # Force rebuild without cache so bad fonts are skipped cleanly
        try:
            _fm._load_fontmanager(try_read_cache=False)
        except Exception:
            pass

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap

    # Force safe fonts — never Ubuntu
    plt.rcParams['font.family']        = 'DejaVu Sans'
    plt.rcParams['font.sans-serif']    = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['text.usetex']        = False

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    mpatches = None
    LinearSegmentedColormap = None

# Ensure LinearSegmentedColormap is always importable at module level
if LinearSegmentedColormap is None:
    try:
        from matplotlib.colors import LinearSegmentedColormap
    except Exception:
        pass

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import torch
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
    TORCH_DEVICE = "cuda" if TORCH_CUDA_AVAILABLE else "cpu"
    CUDA_DEVICE_COUNT = torch.cuda.device_count() if TORCH_CUDA_AVAILABLE else 0
    CUDA_DEVICE_NAME = torch.cuda.get_device_name(0) if TORCH_CUDA_AVAILABLE else "None"
except ImportError:
    TORCH_CUDA_AVAILABLE = False
    TORCH_DEVICE = "cpu"
    CUDA_DEVICE_COUNT = 0
    CUDA_DEVICE_NAME = "None"

print("=" * 60)
print("CUDA AVAILABILITY DEBUG INFO")
print("=" * 60)
print(f"CuPy Available: {CUDA_AVAILABLE}")
print(f"PyTorch CUDA Available: {TORCH_CUDA_AVAILABLE}")
print(f"PyTorch Device: {TORCH_DEVICE}")
print(f"CUDA Device Count: {CUDA_DEVICE_COUNT}")
print(f"CUDA Device Name: {CUDA_DEVICE_NAME}")
print(f"Numba Available: {NUMBA_AVAILABLE}")
print("=" * 60)

temp_dir = os.path.expanduser('~/tmp')
os.makedirs(temp_dir, exist_ok=True)
os.environ['TMPDIR'] = temp_dir
tempfile.tempdir = temp_dir

# ============================================================================
# IMAGE SEQUENCE CAPTURE — Treats a folder of images as a 60 FPS video stream
# ============================================================================

class ImageSequenceCapture:
    """
    Mimics the cv2.VideoCapture interface but reads still images from a folder.
    Each image is treated as one frame in a virtual video at IMAGE_SEQUENCE_FPS.

    Supported image formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp

    Usage:
        cap = ImageSequenceCapture('/path/to/images')
        fps  = cap.get(cv2.CAP_PROP_FPS)          # → 60.0
        n    = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # → number of images
        ret, frame = cap.read()                    # standard OpenCV loop
        cap.release()                              # no-op, here for compatibility
    """

    IMAGE_SEQUENCE_FPS = 60.0  # Virtual frame-rate injected into the pipeline

    # Extensions recognised as image files (case-insensitive)
    _IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    def __init__(self, folder_path: str):
        self._folder = Path(folder_path)
        self._frames = sorted(
            [p for p in self._folder.iterdir()
             if p.suffix.lower() in self._IMAGE_EXTS],
            key=lambda p: p.name          # sort lexicographically by filename
        )
        self._index = 0
        self._opened = len(self._frames) > 0

        if self._opened:
            print(f"[ImageSequenceCapture] Folder   : {self._folder}")
            print(f"[ImageSequenceCapture] Images   : {len(self._frames)}")
            print(f"[ImageSequenceCapture] Virtual FPS: {self.IMAGE_SEQUENCE_FPS}")
            for i, p in enumerate(self._frames[:5]):
                print(f"  [{i:04d}] {p.name}")
            if len(self._frames) > 5:
                print(f"  ... and {len(self._frames) - 5} more")
        else:
            print(f"[ImageSequenceCapture] WARNING: No images found in '{folder_path}'")

    # ------------------------------------------------------------------
    # cv2.VideoCapture compatibility interface
    # ------------------------------------------------------------------

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop_id: int):
        """Return video-property values matching the cv2 property constants."""
        if prop_id == cv2.CAP_PROP_FPS:
            return self.IMAGE_SEQUENCE_FPS
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._frames))
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self._index)
        # Default: unsupported property
        return 0.0

    def read(self):
        """
        Read the next image.
        Returns (True, frame_bgr) when an image is available,
        (False, None) when the sequence is exhausted.
        """
        if self._index >= len(self._frames):
            return False, None

        img_path = self._frames[self._index]
        frame = cv2.imread(str(img_path))

        if frame is None:
            print(f"[ImageSequenceCapture] WARNING: Could not read '{img_path.name}' — skipping")
            self._index += 1
            return False, None          # caller will handle read failure as normal

        self._index += 1
        return True, frame

    def release(self):
        """No-op — kept for drop-in compatibility with cv2.VideoCapture."""
        pass


# ============================================================================
# BEEVISION THEME - ANTENNA DASHBOARD COMPONENTS
# ============================================================================

class BeeVisionTheme:
    """Centralized theme constants for BeeVision styling"""
    
    # Colors
    DARK_BG = "#1a1a1a"
    CARD_BG = "#2d2d2d"
    DARKER_BG = "#232323"
    PRIMARY_BLUE = "#00b4d8"
    LIGHT_BLUE = "#48cae4"
    DARK_BLUE = "#0077b6"
    TEXT_PRIMARY = "#e0e0e0"
    TEXT_SECONDARY = "#b0b0b0"
    TEXT_DISABLED = "#666666"
    BORDER_COLOR = "#444444"
    
    # Status Colors
    SUCCESS = "#00ff00"
    WARNING = "#ffa500"
    ERROR = "#ff0000"
    INFO = "#ffff00"
    
    # Fonts
    FONT_FAMILY = "Segoe UI"
    
    @staticmethod
    def get_card_style():
        """Standard card stylesheet"""
        return f"""
            QFrame {{
                background-color: {BeeVisionTheme.CARD_BG};
                border: 1px solid {BeeVisionTheme.PRIMARY_BLUE};
                border-radius: 8px;
                padding: 12px;
            }}
        """
    
    @staticmethod
    def get_section_header_style(expanded=False):
        """Expandable section header stylesheet"""
        bg_color = BeeVisionTheme.CARD_BG if expanded else BeeVisionTheme.DARKER_BG
        border_color = BeeVisionTheme.PRIMARY_BLUE if expanded else BeeVisionTheme.BORDER_COLOR
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: {BeeVisionTheme.PRIMARY_BLUE};
                border: 1px solid {border_color};
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: bold;
                text-align: left;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background-color: #2a2a2a;
                border: 1px solid {BeeVisionTheme.PRIMARY_BLUE};
            }}
            QPushButton:pressed {{
                background-color: {BeeVisionTheme.DARK_BG};
            }}
        """
    
    @staticmethod
    def get_progress_bar_style():
        """Progress bar for distributions"""
        return f"""
            QProgressBar {{
                background-color: {BeeVisionTheme.DARKER_BG};
                border: 1px solid {BeeVisionTheme.BORDER_COLOR};
                border-radius: 4px;
                text-align: center;
                color: {BeeVisionTheme.TEXT_PRIMARY};
                height: 20px;
                font-weight: bold;
            }}
            QProgressBar::chunk {{
                background-color: {BeeVisionTheme.PRIMARY_BLUE};
                border-radius: 3px;
            }}
        """


# ============================================================================
# RESEARCH VISUALIZATION TAB - Morphological Processing Pipeline
# ============================================================================

class ZoomableImageLabel(QLabel):
    """
    A QLabel that displays an image and opens a zoom dialog when clicked.
    """
    clicked = pyqtSignal(str)  # Emits panel_id when clicked
    
    def __init__(self, panel_id, parent=None):
        super().__init__(parent)
        self.panel_id = panel_id
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click to zoom")
        self.current_pixmap = None
        self.current_array = None  # Store raw array for full-res zoom
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.panel_id)
        super().mousePressEvent(event)
    
    def setImageArray(self, img_array):
        """Store the raw image array for full-resolution zoom"""
        self.current_array = img_array.copy() if img_array is not None else None


class ZoomDialog(QWidget):
    """
    Fullscreen overlay dialog for zoomed image viewing.
    Supports LIVE VIDEO - continuously updates while open.
    Click anywhere or press Escape to close.
    """
    
    closed = pyqtSignal()  # Signal emitted when dialog is closed
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 230);")
        
        # Track current panel being zoomed
        self.current_panel_id = None
        self.current_title = ""
        self.current_description = ""
        
        # Cache display size (calculated once when shown)
        self._display_width = 800
        self._display_height = 600
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title bar
        title_layout = QHBoxLayout()
        
        self.title_label = QLabel("ZOOMED VIEW")
        self.title_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.title_label.setStyleSheet("color: #00b4d8; background: transparent;")
        title_layout.addWidget(self.title_label)
        
        title_layout.addStretch()
        
        # Live indicator
        self.live_label = QLabel("🔴 LIVE")
        self.live_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.live_label.setStyleSheet("color: #ff4444; background: transparent;")
        title_layout.addWidget(self.live_label)
        
        title_layout.addStretch()
        
        # Close hint
        close_hint = QLabel("Press ESC to close")
        close_hint.setFont(QFont("Segoe UI", 10))
        close_hint.setStyleSheet("color: #888888; background: transparent;")
        title_layout.addWidget(close_hint)
        
        # Close button
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(40, 40)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #c62828;
                color: white;
                border: none;
                border-radius: 20px;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e53935;
            }
        """)
        close_btn.clicked.connect(self.close)
        title_layout.addWidget(close_btn)
        
        layout.addLayout(title_layout)
        
        # Image container with border
        image_container = QFrame()
        image_container.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 3px solid #00b4d8;
                border-radius: 12px;
            }
        """)
        container_layout = QVBoxLayout(image_container)
        container_layout.setContentsMargins(10, 10, 10, 10)
        
        # Image label (no scroll area for better performance)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background: #000000;")
        self.image_label.setScaledContents(False)  # Don't auto-scale content
        container_layout.addWidget(self.image_label)
        
        layout.addWidget(image_container, stretch=1)
        
        # Info bar
        self.info_label = QLabel("")
        self.info_label.setFont(QFont("Courier New", 10))
        self.info_label.setStyleSheet("color: #00ff00; background: #0a0a0a; padding: 8px; border: 1px solid #333; border-radius: 4px;")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)
    
    def set_panel(self, panel_id, title, description):
        """Set which panel is being zoomed"""
        self.current_panel_id = panel_id
        self.current_title = title
        self.current_description = description
        self.title_label.setText(f"🔍 {title}")
    
    def update_image(self, img_array):
        """Update the displayed image - called continuously for live video"""
        if img_array is None:
            self.image_label.setText("No image data")
            return
        
        # Make a copy and ensure uint8
        img_array = img_array.copy()
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)
        
        # Get image dimensions
        h, w = img_array.shape[:2]
        
        # Convert grayscale to RGB (NO COLORMAP - keep as grayscale)
        if len(img_array.shape) == 2:
            colored = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            colored = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Make contiguous for QImage
        colored = np.ascontiguousarray(colored)
        
        # Create QImage
        h, w, ch = colored.shape
        bytes_per_line = ch * w
        q_img = QImage(colored.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        
        # Use CACHED size (set in showFullScreen) to avoid feedback loop
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self._display_width, self._display_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        
        # Update info
        self.info_label.setText(
            f"{self.current_description}  |  Original: {w}×{h} px  |  "
            f"Display: {scaled_pixmap.width()}×{scaled_pixmap.height()} px  |  "
            f"🔴 LIVE VIDEO"
        )
    
    def keyPressEvent(self, event):
        """Close on Escape key"""
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        super().keyPressEvent(event)
    
    def closeEvent(self, event):
        """Emit closed signal when dialog closes"""
        self.current_panel_id = None
        self.closed.emit()
        super().closeEvent(event)
    
    def showFullScreen(self):
        """Override to ensure proper fullscreen display and cache display size"""
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        # Cache the display size for image scaling
        self._display_width = int(screen.width() * 0.80)
        self._display_height = int(screen.height() * 0.70)
        super().showFullScreen()


class ResearchVisualizationWidget(QWidget):
    """
    Widget displaying all intermediate morphological processing stages
    for research paper documentation.
    
    Layout (3x3 grid):
    ┌─────────────────┬─────────────────┬─────────────────┐
    │   ORIGINAL      │ Vertical Tophat │Horizontal Tophat│
    │   Grayscale     │                 │                 │
    ├─────────────────┼─────────────────┼─────────────────┤
    │ Diagonal Tophat │Combined Maximum │  Final Binary   │
    │                 │                 │   (Otsu+Open)   │
    ├─────────────────┼─────────────────┼─────────────────┤
    │   ANNOTATED     │                 │                 │
    │ Opening+Overlay │                 │                 │
    └─────────────────┴─────────────────┴─────────────────┘
    
    Click any panel to zoom in for detailed viewing.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #1a1a1a;")
        
        # Zoom dialog (reusable)
        self.zoom_dialog = None
        
        # Store current image data for each panel
        self.panel_data = {}
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Title
        title = QLabel("🔬 MORPHOLOGICAL PROCESSING PIPELINE - FOR RESEARCH")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #00b4d8; background: transparent;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        # Subtitle with processing info
        self.subtitle = QLabel("Morphological: Original → Tophat → Combined → Binary → Annotated (G)  •  Antenna: Blob (H) → Closing (I) → Skeleton (J) → Endpoints (K) → BFS (L) → Keypoints (M)  •  Click to ZOOM")
        self.subtitle.setFont(QFont("Segoe UI", 9))
        self.subtitle.setStyleSheet("color: #888888; background: transparent;")
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.subtitle)
        
        # Grid for 6 visualization panels
        grid_widget = QWidget()
        grid_widget.setStyleSheet("background: transparent;")
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(8)
        
        # Create panels for morphological + antenna processing
        self.panels = {}
        self.panel_configs = {
            # Morphological processing pipeline (original)
            'original': ("ORIGINAL GRAYSCALE", "Raw frame - antennae are dark"),
            'vertical': ("VERTICAL TOPHAT", "15×1 kernel → ↕ bright"),
            'horizontal': ("HORIZONTAL TOPHAT", "1×15 kernel → ↔ bright"),
            'diagonal': ("DIAGONAL TOPHAT", "9×9 cross → ↗↘ bright"),
            'combined': ("COMBINED MAXIMUM", "max(vertical,horizontal,diagonal)"),
            'binary': ("FINAL BINARY", "Clean B/W mask"),
            'annotated': ("(G) ANNOTATED OPENING", "Opening + ROI Triangles"),
            # NEW: Antenna processing pipeline - continuing alphabetical sequence
            'antenna_original': ("(H) ORIGINAL ANTENNA BLOB", "Raw binary mask of antenna region"),
            'antenna_closing': ("(I) AFTER MORPHOLOGICAL CLOSING", "Gap bridging (3×3 ellipse, 2 iterations)"),
            'antenna_skeleton': ("(J) ZHANG-SUEN SKELETONIZATION", "1-pixel wide skeleton result"),
            'antenna_endpoints': ("(K) ENDPOINT DETECTION", "Base (green, near head) & tip (red, far)"),
            'antenna_bfs': ("(L) BFS PATH TRACING", "Breadth-first search from base to tip"),
            'antenna_keypoints': ("(M) FINAL KEYPOINT PLACEMENT", "Joint @ 30% (p24), Tip @ 100% (p81)"),
        }
        
        panel_positions = [
            # Row 0: Morphological tophat stages
            ('original', 0, 0),    
            ('vertical', 0, 1),    
            ('horizontal', 0, 2),  
            # Row 1: Combined stages
            ('diagonal', 1, 0),    
            ('combined', 1, 1),    
            ('binary', 1, 2),      
            # Row 2: Annotated opening + antenna pipeline start
            ('annotated', 2, 0),   
            ('antenna_original', 2, 1),
            ('antenna_closing', 2, 2),
            # Row 3: Antenna skeleton processing
            ('antenna_skeleton', 3, 0),
            ('antenna_endpoints', 3, 1),
            ('antenna_bfs', 3, 2),
            # Row 4: Final antenna keypoints
            ('antenna_keypoints', 4, 0),
        ]
        
        for panel_id, row, col in panel_positions:
            title_text, desc_text = self.panel_configs[panel_id]
            panel = self._create_panel(panel_id, title_text, desc_text)
            grid_layout.addWidget(panel['frame'], row, col)
            self.panels[panel_id] = panel
        
        main_layout.addWidget(grid_widget, stretch=1)
        
        # Kernel visualization section
        kernel_section = self._create_kernel_section()
        main_layout.addWidget(kernel_section)
        
        # Processing stats
        self.stats_label = QLabel("Waiting for video...  •  Click any panel to zoom")
        self.stats_label.setFont(QFont("Courier New", 9))
        self.stats_label.setStyleSheet("color: #00ff00; background: #0a0a0a; padding: 8px; border: 1px solid #333;")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.stats_label)
    
    def _create_panel(self, panel_id, title_text, desc_text):
        """Create a single visualization panel with title, zoomable image, and description"""
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.Box)
        frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 2px solid #00b4d8;
                border-radius: 8px;
            }
            QFrame:hover {
                border: 2px solid #48cae4;
                background-color: #353535;
            }
        """)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        
        # Title with zoom hint
        title = QLabel(f"🔍 {title_text}")
        title.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        title.setStyleSheet("color: #00b4d8; border: none; background: transparent;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Zoomable image display
        image_label = ZoomableImageLabel(panel_id)
        image_label.setText("Waiting...")
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setStyleSheet("""
            QLabel {
                background: #1a1a1a; 
                border: 1px solid #444; 
                color: #666;
            }
            QLabel:hover {
                border: 1px solid #00b4d8;
                background: #222222;
            }
        """)
        image_label.setMinimumSize(250, 180)
        image_label.setScaledContents(False)
        image_label.clicked.connect(self._on_panel_clicked)
        layout.addWidget(image_label, stretch=1)
        
        # Description
        desc = QLabel(f"{desc_text}  •  Click to zoom")
        desc.setFont(QFont("Courier New", 8))
        desc.setStyleSheet("color: #888888; border: none; background: transparent;")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)
        
        return {
            'frame': frame,
            'title': title,
            'image': image_label,
            'desc': desc,
            'title_text': title_text,
            'desc_text': desc_text
        }
    
    def _on_panel_clicked(self, panel_id):
        """Handle panel click - show zoom dialog with LIVE video"""
        if panel_id not in self.panel_data or self.panel_data[panel_id] is None:
            return
        
        # Create zoom dialog if needed
        if self.zoom_dialog is None:
            self.zoom_dialog = ZoomDialog()
            self.zoom_dialog.closed.connect(self._on_zoom_closed)
        
        # Get panel info
        title_text, desc_text = self.panel_configs[panel_id]
        
        # Set which panel is being zoomed
        self.zoom_dialog.set_panel(panel_id, title_text, desc_text)
        
        # Show initial image
        img_array = self.panel_data[panel_id]
        self.zoom_dialog.update_image(img_array)
        
        # Show fullscreen
        self.zoom_dialog.showFullScreen()
        self.zoom_dialog.activateWindow()
    
    def _on_zoom_closed(self):
        """Called when zoom dialog is closed"""
        pass  # Nothing special needed
    
    def _create_kernel_section(self):
        """Create section showing kernel visualizations"""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #232323;
                border: 1px solid #444;
                border-radius: 6px;
            }
        """)
        
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("STRUCTURING ELEMENTS:")
        title.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        title.setStyleSheet("color: #00b4d8; border: none;")
        layout.addWidget(title)
        
        # Kernel displays
        self.kernel_displays = {}
        kernels = [
            ('vertical', "For ↕\n15×1 —"),
            ('horizontal', "For ↔\n1×15 |"),
            ('diagonal', "For ↗↘\n9×9 +"),
            ('opening', "Clean\n3×3 ○"),
        ]
        
        for kernel_id, label_text in kernels:
            kernel_widget = QWidget()
            kernel_widget.setStyleSheet("background: transparent;")
            kernel_layout = QVBoxLayout(kernel_widget)
            kernel_layout.setContentsMargins(0, 0, 0, 0)
            kernel_layout.setSpacing(2)
            
            # Kernel image
            kernel_img = QLabel()
            kernel_img.setFixedSize(40, 40)
            kernel_img.setStyleSheet("background: #1a1a1a; border: 1px solid #555;")
            kernel_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            kernel_layout.addWidget(kernel_img, alignment=Qt.AlignmentFlag.AlignCenter)
            
            # Kernel label
            kernel_label = QLabel(label_text)
            kernel_label.setFont(QFont("Segoe UI", 7))
            kernel_label.setStyleSheet("color: #888; border: none;")
            kernel_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            kernel_layout.addWidget(kernel_label)
            
            layout.addWidget(kernel_widget)
            self.kernel_displays[kernel_id] = kernel_img
        
        layout.addStretch()
        
        # Draw initial kernel visualizations
        self._draw_kernel_visualizations()
        
        return frame
    
    def _draw_kernel_visualizations(self):
        """Draw the structuring elements as small images"""
        # For VERTICAL detection: HORIZONTAL kernel (15×1) - wide bar
        vert_kernel = np.zeros((21, 21), dtype=np.uint8)
        vert_kernel[10, 3:18] = 255  # 15 pixels wide, horizontal line
        self._set_kernel_pixmap('vertical', vert_kernel)
        
        # For HORIZONTAL detection: VERTICAL kernel (1×15) - tall bar
        horiz_kernel = np.zeros((21, 21), dtype=np.uint8)
        horiz_kernel[3:18, 10] = 255  # 15 pixels tall, vertical line
        self._set_kernel_pixmap('horizontal', horiz_kernel)
        
        # Cross kernel (9×9) - plus sign shape
        cross_kernel = np.zeros((21, 21), dtype=np.uint8)
        cross_kernel[10, 6:15] = 255  # Horizontal part (9 pixels)
        cross_kernel[6:15, 10] = 255  # Vertical part (9 pixels)
        self._set_kernel_pixmap('diagonal', cross_kernel)
        
        # Opening kernel (3×3 ellipse)
        open_kernel = np.zeros((21, 21), dtype=np.uint8)
        cv2.ellipse(open_kernel, (10, 10), (2, 2), 0, 0, 360, 255, -1)
        self._set_kernel_pixmap('opening', open_kernel)
    
    def _set_kernel_pixmap(self, kernel_id, kernel_array):
        """Convert kernel array to QPixmap and set on label"""
        # Scale up for visibility
        scaled = cv2.resize(kernel_array, (40, 40), interpolation=cv2.INTER_NEAREST)
        
        # Convert to QImage
        h, w = scaled.shape
        q_img = QImage(scaled.data, w, h, w, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        
        if kernel_id in self.kernel_displays:
            self.kernel_displays[kernel_id].setPixmap(pixmap)
    
    def update_visualization(self, research_data):
        """Update all panels with new visualization data"""
        if research_data is None:
            return
        
        # Update each panel - maps data keys to panel IDs
        panel_mapping = {
            # Morphological processing
            'original_grayscale': 'original',
            'vertical_tophat': 'vertical',
            'horizontal_tophat': 'horizontal',
            'diagonal_tophat': 'diagonal',
            'combined_max': 'combined',
            'final_binary': 'binary',
            'annotated_opening': 'annotated',
            # NEW: Antenna processing pipeline
            'antenna_original_blob': 'antenna_original',
            'antenna_after_closing': 'antenna_closing',
            'antenna_skeleton': 'antenna_skeleton',
            'antenna_endpoints': 'antenna_endpoints',
            'antenna_bfs_path': 'antenna_bfs',
            'antenna_final_keypoints': 'antenna_keypoints',
        }
        
        for data_key, panel_id in panel_mapping.items():
            if data_key in research_data and research_data[data_key] is not None:
                img_array = research_data[data_key]
                # Store raw data for zoom functionality
                self.panel_data[panel_id] = img_array.copy()
                self._update_panel_image(panel_id, img_array)
                
                # Update zoom dialog if it's showing this panel (LIVE VIDEO)
                if (self.zoom_dialog is not None and 
                    self.zoom_dialog.isVisible() and 
                    self.zoom_dialog.current_panel_id == panel_id):
                    self.zoom_dialog.update_image(img_array)
        
        # Update stats
        if 'stats' in research_data:
            stats = research_data['stats']
            v_max = stats.get('v_max', 0)
            h_max = stats.get('h_max', 0)
            d_max = stats.get('d_max', 0)
            
            # Include antenna processing stats if available
            antenna_info = ""
            if 'antenna_endpoints_count' in stats:
                antenna_info = f" | Antenna Endpoints: {stats['antenna_endpoints_count']}"
            if 'antenna_path_length' in stats:
                antenna_info += f" | Path Length: {stats['antenna_path_length']}"
            
            # Determine threshold mode label
            thresh_val = stats.get('otsu_threshold', 0)
            pct = stats.get('roi_percentile', 0)
            if pct > 0:
                thresh_mode = f"Percentile Top {pct:.1f}%"
                thresh_display = thresh_mode
            elif stats.get('manual_threshold', False):
                thresh_mode = "Manual"
                thresh_display = f"{thresh_val:.2f} ({thresh_mode})"
            else:
                thresh_mode = "Otsu"
                thresh_display = f"{thresh_val} ({thresh_mode})"
            
            stats_text = (
                f"Frame: {stats.get('frame_id', 0):,} | "
                f"Tophat Max: V={v_max} H={h_max} D={d_max} | "
                f"Thresh: {thresh_display} | "
                f"Binary px: {stats.get('white_pixel_count', 0):,}"
                f"{antenna_info} | "
                f"{stats.get('processing_time_ms', 0):.1f}ms"
            )
            self.stats_label.setText(stats_text)
    
    def _update_panel_image(self, panel_id, img_array):
        """
        Update a single panel's image display.
        
        Panel A (original): Normal grayscale - bee visible with dark antennae
        Panels B-E (tophat outputs): Bright structures on dark background - NO colormap
        Panel F (binary): Pure black/white
        """
        if panel_id not in self.panels:
            return
        
        panel = self.panels[panel_id]
        image_label = panel['image']
        
        # Ensure array is uint8
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)
        
        # Store raw array in the zoomable label
        image_label.setImageArray(img_array)
        
        # All panels shown as grayscale (no colormaps)
        # This makes the paper figures cleaner and more scientific
        if len(img_array.shape) == 2:
            # Grayscale - convert to RGB for display
            colored = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            colored = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Create QImage
        h, w, ch = colored.shape
        bytes_per_line = ch * w
        q_img = QImage(colored.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        image_label.setPixmap(scaled_pixmap)


# ============================================================================
# ANTENNA PROCESSING PIPELINE - NEW VISUALIZATION STAGES
# ============================================================================

class TemporalAntennaTracker:
    """
    Layers 2 & 3 of the three-layer antenna keypoint stabilisation system.

    Layer 2 — Exponential Moving Average (EMA):
        smoothed = α × detected + (1 − α) × previous_smoothed
        α is user-configurable via a slider (default 0.6).

    Layer 3 — Confidence-Decayed Gap Fill:
        When detection fails for a side, the last smoothed position is
        carried forward with confidence multiplied by *decay_rate* each
        frame.  After *max_gap_frames* consecutive misses (or conf < 0.15)
        the side is treated as truly lost and returns None.

    Usage
    -----
    tracker.update(bee_id, 'left', base, joint, tip)  # on detection
    tracker.mark_miss(bee_id, 'left')                  # on failure
    base, joint, tip, conf = tracker.get_smoothed(bee_id, 'left')
    """

    def __init__(self, ema_alpha=0.6, max_gap_frames=10, decay_rate=0.85):
        self.ema_alpha = ema_alpha
        self.max_gap_frames = max_gap_frames
        self.decay_rate = decay_rate
        # {bee_id: {'left': {...}, 'right': {...}}}
        self._state = {}

    # ── helpers ──────────────────────────────────────────────────────────
    def _ensure_bee(self, bee_id):
        if bee_id not in self._state:
            self._state[bee_id] = {
                side: {
                    'base': None, 'joint': None, 'tip': None,
                    'conf': 0.0, 'miss_count': 999,
                }
                for side in ('left', 'right')
            }

    @staticmethod
    def _ema_pt(alpha, new_pt, old_pt):
        """Blend (x, y) tuples with EMA."""
        if old_pt is None:
            return new_pt
        return (alpha * new_pt[0] + (1.0 - alpha) * old_pt[0],
                alpha * new_pt[1] + (1.0 - alpha) * old_pt[1])

    # ── public API ──────────────────────────────────────────────────────
    def update(self, bee_id, side, base, joint, tip):
        """Feed a new detection for *side* ('left' | 'right')."""
        self._ensure_bee(bee_id)
        s = self._state[bee_id][side]
        a = self.ema_alpha
        for key, pt in (('base', base), ('joint', joint), ('tip', tip)):
            if pt is not None:
                s[key] = self._ema_pt(a, pt, s[key])
        s['conf'] = 1.0
        s['miss_count'] = 0

    def mark_miss(self, bee_id, side):
        """Call when detection fails for *side* this frame."""
        self._ensure_bee(bee_id)
        s = self._state[bee_id][side]
        s['miss_count'] += 1
        s['conf'] *= self.decay_rate

    def get_smoothed(self, bee_id, side):
        """Return (base, joint, tip, conf).
        
        ALWAYS returns the last known position — never None.
        Confidence decays on misses (minimum floor 0.10) but positions
        are carried forward indefinitely so both antennae are always
        available for plotting.
        """
        self._ensure_bee(bee_id)
        s = self._state[bee_id][side]
        # Floor confidence at 0.10 so gap-filled points are still drawn
        # but downstream can distinguish them from fresh detections.
        conf = max(s['conf'], 0.10) if s['base'] is not None else 0.0
        return s['base'], s['joint'], s['tip'], conf

    def reset(self, bee_id=None):
        if bee_id is None:
            self._state.clear()
        elif bee_id in self._state:
            del self._state[bee_id]


class AntennaProcessingPipeline:
    """
    Complete antenna processing pipeline with visualization at each stage.
    Implements the morphological processing steps for antenna keypoint detection.
    
    Pipeline Stages:
    H. Original antenna blob - raw binary mask
    I. After morphological closing - gap bridging
    J. Zhang-Suen skeletonization - thinning result
    K. Endpoint detection - base (green) and tip (red) identified
    L. BFS path tracing - from base to tip
    M. Final keypoint placement - joint at 30% (p24) and tip at 100% (p81)
    """
    
    def __init__(self):
        self.intermediate_results = {}
    
    def process_antenna(self, blob_mask, head_point, body_center,
                         left_roi_mask=None, right_roi_mask=None):
        """
        Process antenna blob through complete morphological pipeline.
        
        Three-layer stabilisation (Layer 1 = ROI-anchored tracing):
        1. Original blob
        2. Morphological closing (gap bridging)
        3. Zhang-Suen skeletonization (thinning)
        4. ROI-guided path tracing (when ROI masks provided) OR
           component-based path tracing (legacy fallback)
        5. Place keypoints along this path: tip at 100%, joint at 30%
        
        Args:
            blob_mask: Binary mask of antenna blob (numpy array)
            head_point: (x, y) coordinates of head point
            body_center: (x, y) coordinates of body center
            left_roi_mask:  Optional uint8 mask (same size as blob_mask)
                            isolating the LEFT antenna ROI region
            right_roi_mask: Optional uint8 mask (same size as blob_mask)
                            isolating the RIGHT antenna ROI region
            
        Returns:
            dict: Contains all intermediate processing stages and final keypoints
        """
        results = {
            'original_blob':    None,
            'after_closing':    None,
            'skeleton':         None,
            'endpoints':        None,
            # Per-antenna results — index 0 = antenna A, index 1 = antenna B
            # (both are filled when two valid antennas are found; only [0] when one)
            'bfs_paths':        [],   # list of paths, each a list of (x,y)
            'base_points':      [],   # list of base (x,y) per antenna
            'tip_points':       [],   # list of tip  (x,y) per antenna
            'joint_points':     [],   # list of joint (x,y) at 30% per antenna
            'endpoint_list':    [],
            # Legacy single-antenna keys kept for any other callers
            'bfs_path':         None,
            'final_keypoints':  None,
            'base_point':       None,
            'tip_point':        None,
            'joint_point':      None,
        }
        
        if blob_mask is None or blob_mask.size == 0:
            print(f"[ANTENNA] Blob mask is None or empty")
            return results
        
        try:
            # Stage H: Original antenna blob
            results['original_blob'] = blob_mask.copy()
            print(f"[ANTENNA] Stage H: Original blob size = {blob_mask.shape}")
            
            # Stage I: Morphological closing to bridge gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            closed = cv2.morphologyEx(blob_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            results['after_closing'] = closed.copy()
            print(f"[ANTENNA] Stage I: After closing")

            # ── Stage I.5: Dual-Antenna Component Selection ─────────────────
            # Bees have TWO antennas — both are separate disconnected blobs in
            # the closed binary.  We score every component by:
            #
            #     score = area / (dist_to_head + 1)
            #
            # …and keep the TOP-2 highest-scoring components (one per antenna).
            # Components smaller than MIN_AREA pixels are treated as noise and
            # are never selected regardless of score.
            MIN_AREA = 8   # px — below this a blob is noise, not an antenna

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                closed, connectivity=8
            )

            if head_point is not None and len(head_point) == 2:
                hpx, hpy = float(head_point[0]), float(head_point[1])
            else:
                hpx = closed.shape[1] / 2.0
                hpy = closed.shape[0] / 2.0

            # Stage 3+5: elongation-aware scoring with hard elongation guard
            #
            # Old score:  area / (dist + 1)
            # New score:  area * elongation / (dist + 1)
            #
            # elongation = max_bbox_dim / min_bbox_dim
            #   ~1.0  → compact blob  (body part, noise dot)  → rejected
            #   ~3-10 → elongated rod (actual antenna)         → kept
            #
            # MIN_ELONGATION is a hard filter applied BEFORE scoring so that
            # compact blobs can never win regardless of area or proximity.
            MIN_ELONGATION = 1.2   # below this → definitely not an antenna (bees: 1.2-1.6 typical)

            scored = []
            for lbl in range(1, num_labels):       # 0 = background
                area   = int(stats[lbl, cv2.CC_STAT_AREA])
                if area < MIN_AREA:
                    continue                        # too small → noise

                bw     = float(stats[lbl, cv2.CC_STAT_WIDTH])
                bh     = float(stats[lbl, cv2.CC_STAT_HEIGHT])
                elong  = max(bw, bh) / (min(bw, bh) + 1e-6)

                if elong < MIN_ELONGATION:
                    print(f"[ANTENNA] Component {lbl}: area={area}, "
                          f"elongation={elong:.2f} < {MIN_ELONGATION} → rejected (compact)")
                    continue                        # compact blob → not an antenna

                cx    = float(centroids[lbl][0])
                cy    = float(centroids[lbl][1])
                dist  = np.sqrt((cx - hpx) ** 2 + (cy - hpy) ** 2)
                score = area * elong / (dist + 1.0)
                scored.append((score, lbl, area, elong))
                print(f"[ANTENNA] Component {lbl}: area={area}, "
                      f"elong={elong:.2f}, dist={dist:.1f}, score={score:.2f}")

            if not scored:
                print("[ANTENNA] Stage I.5: No elongated components found — "
                      "retrying without elongation filter")
                # Graceful fallback: drop elongation guard, keep area guard only
                for lbl in range(1, num_labels):
                    area = int(stats[lbl, cv2.CC_STAT_AREA])
                    if area < MIN_AREA:
                        continue
                    bw   = float(stats[lbl, cv2.CC_STAT_WIDTH])
                    bh   = float(stats[lbl, cv2.CC_STAT_HEIGHT])
                    elong = max(bw, bh) / (min(bw, bh) + 1e-6)
                    cx   = float(centroids[lbl][0])
                    cy   = float(centroids[lbl][1])
                    dist = np.sqrt((cx - hpx) ** 2 + (cy - hpy) ** 2)
                    score = area * elong / (dist + 1.0)
                    scored.append((score, lbl, area, elong))

            if not scored:
                print("[ANTENNA] Stage I.5: No valid components found at all")
                return results

            # Sort descending by score, take best 2
            scored.sort(key=lambda x: x[0], reverse=True)
            top2 = scored[:2]
            print(f"[ANTENNA] Stage I.5: Keeping {len(top2)} antenna component(s): "
                  + ", ".join(f"lbl={t[1]} area={t[2]} elong={t[3]:.2f}"
                               for t in top2))

            # Build a combined mask from the selected components
            dual_mask = np.zeros_like(closed)
            for _, lbl, _, _ in top2:
                dual_mask = cv2.bitwise_or(dual_mask,
                    ((labels == lbl).astype(np.uint8)) * 255)

            results['after_closing'] = dual_mask.copy()
            # ─────────────────────────────────────────────────────────────────

            # Stage J: Zhang-Suen on the dual-antenna mask
            # Both antennas are skeletonized together; we then process each
            # selected component's skeleton separately in Stage K+L.
            skeleton_raw = self.zhang_suen_thinning(dual_mask)
            skeleton_pixels_raw = np.count_nonzero(skeleton_raw)
            print(f"[ANTENNA] Stage J (raw): Skeleton has {skeleton_pixels_raw} pixels")

            if skeleton_pixels_raw == 0:
                print("[ANTENNA] No skeleton pixels after first pass - aborting")
                return results

            # ── Stage J.5: Dilate → Re-skeletonize ──────────────────────────
            # The first skeleton often has micro-gaps (1-2px breaks) where the
            # antenna passed through a slightly brighter background pixel and
            # was missed by the binary threshold.  These gaps cause BFS to stop
            # early, placing the tip at a mid-antenna break.
            #
            # Fix:
            #   1. Dilate the 1-px skeleton by a 3×3 ellipse (1 iteration).
            #      This thickens it to ~3px and bridges micro-gaps.
            #   2. Re-skeletonize the thickened result.
            #      The second skeleton is a single connected 1-px curve with
            #      far fewer breaks and no staircase jagging on diagonals.
            #   3. BFS runs on this second (clean) skeleton.
            #
            # Dilation radius is deliberately small (1 iter, 3×3) so nearby
            # parallel structures never merge into one blob.
            k_dilate  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            skel_thick = cv2.dilate(skeleton_raw, k_dilate, iterations=1)
            skeleton   = self.zhang_suen_thinning(skel_thick)
            skeleton_pixels = np.count_nonzero(skeleton)
            print(f"[ANTENNA] Stage J.5 (clean): Skeleton has {skeleton_pixels} px "
                  f"(was {skeleton_pixels_raw} before dilate+reskeletonize)")

            if skeleton_pixels == 0:
                # Dilate+reskeletonize occasionally collapses a tiny blob;
                # fall back to the first skeleton in that case.
                print("[ANTENNA] Re-skeletonize collapsed skeleton — using raw")
                skeleton        = skeleton_raw
                skeleton_pixels = skeleton_pixels_raw

            results['skeleton'] = skeleton.copy()
            # ─────────────────────────────────────────────────────────────────

            # ── Stage K+L: BFS path tracing — ROI-guided or component-based ──
            #
            # LAYER 1 (ROI-guided):  When left/right ROI masks are provided,
            #   split the skeleton by ROI side.  Each side gets its own BFS
            #   from the skeleton pixel nearest to head_point *within that ROI*.
            #   This guarantees one antenna per side and eliminates
            #   cross-contamination between antennae.
            #
            # FALLBACK (component-based):  When no ROI masks are provided,
            #   use the existing approach — BFS per connected component.
            #
            all_endpoints = []

            def _collect_endpoints(skel_img):
                """Return topological endpoints of a binary skeleton."""
                eps = []
                pts = np.argwhere(skel_img > 0)
                h_s, w_s = skel_img.shape
                for (y, x) in pts:
                    if y == 0 or y >= h_s - 1 or x == 0 or x >= w_s - 1:
                        continue
                    patch = skel_img[y - 1:y + 2, x - 1:x + 2].copy()
                    patch[1, 1] = 0
                    if int(np.sum(patch > 0)) == 1:
                        eps.append((x, y))
                return eps

            def _process_side_skeleton(side_skel, side_label):
                """BFS a single side's skeleton and append results."""
                if np.count_nonzero(side_skel) < 2:
                    print(f"[ANTENNA] {side_label}: <2 skeleton pixels — skipped")
                    return
                path = self.find_longest_antenna_path(
                    side_skel, head_point, body_center)
                if len(path) < 2:
                    print(f"[ANTENNA] {side_label}: path too short — skipped")
                    return
                results['bfs_paths'].append(path)
                results['base_points'].append(path[0])
                results['tip_points'].append(path[-1])
                joint_idx = int(len(path) * 0.30)
                results['joint_points'].append(path[joint_idx])
                all_endpoints.extend(_collect_endpoints(side_skel))
                print(f"[ANTENNA] {side_label}: path={len(path)} pts, "
                      f"base={path[0]}, tip={path[-1]}")

            roi_guided = (left_roi_mask is not None and
                          right_roi_mask is not None)

            if roi_guided:
                # ── LAYER 1: ROI-anchored tracing ──────────────────────────
                print("[ANTENNA] Stage K+L: Using ROI-guided skeleton split")
                for side_mask, side_name in ((right_roi_mask, 'Right-ROI'),
                                              (left_roi_mask,  'Left-ROI')):
                    side_skel = cv2.bitwise_and(skeleton, side_mask)
                    _process_side_skeleton(side_skel, side_name)
            else:
                # ── FALLBACK: component-based tracing (original) ───────────
                print("[ANTENNA] Stage K+L: Using component-based skeleton split")
                for _, lbl, _, _ in top2:
                    comp_mask  = ((labels == lbl).astype(np.uint8)) * 255
                    comp_skel  = cv2.bitwise_and(skeleton, comp_mask)
                    _process_side_skeleton(comp_skel, f'Component-{lbl}')

            results['endpoint_list'] = all_endpoints

            # ── GUARANTEE: Always produce exactly 2 antenna paths ──────────
            # If BFS found < 2 paths, synthesize the missing one(s) from kp0
            # along the ROI direction so downstream never sees a gap.
            SYNTH_LEN    = 40   # pixels — default antenna length for fallback
            SYNTH_POINTS = 20   # number of points in synthetic path

            if head_point is not None and len(head_point) == 2:
                hp_x, hp_y = int(head_point[0]), int(head_point[1])
            else:
                hp_x, hp_y = blob_mask.shape[1] // 2, blob_mask.shape[0] // 2

            # Heading vector (kp0 → forward)
            if body_center is not None and len(body_center) == 2:
                _hdx = hp_x - int(body_center[0])
                _hdy = hp_y - int(body_center[1])
                _hmag = np.sqrt(_hdx**2 + _hdy**2)
                if _hmag > 1e-6:
                    _hdx /= _hmag; _hdy /= _hmag
                else:
                    _hdx, _hdy = 0.0, -1.0
            else:
                _hdx, _hdy = 0.0, -1.0

            # Perpendicular (pointing right when facing forward)
            _perp_x, _perp_y = -_hdy, _hdx

            def _synth_path(direction_x, direction_y, label):
                """Create a straight synthetic path from kp0 along a direction."""
                pts = []
                for i in range(SYNTH_POINTS):
                    t = i / max(1, SYNTH_POINTS - 1)
                    px = int(hp_x + direction_x * SYNTH_LEN * t)
                    py = int(hp_y + direction_y * SYNTH_LEN * t)
                    pts.append((px, py))
                print(f"[ANTENNA] {label}: SYNTHETIC fallback path "
                      f"({SYNTH_POINTS} pts, {SYNTH_LEN}px)")
                return pts

            # ROI-guided direction or heading ± 35° offset
            import math as _math
            _angle_offset = _math.radians(35)

            def _roi_centroid_dir(roi_mask):
                """Direction from kp0 to centroid of ROI mask pixels."""
                if roi_mask is None:
                    return None
                ys, xs = np.where(roi_mask > 0)
                if len(xs) < 5:
                    return None
                cx, cy = float(np.mean(xs)), float(np.mean(ys))
                dx, dy = cx - hp_x, cy - hp_y
                mag = np.sqrt(dx**2 + dy**2)
                if mag < 1e-6:
                    return None
                return dx / mag, dy / mag

            # Directions for right (antenna 0) and left (antenna 1)
            right_dir = _roi_centroid_dir(right_roi_mask)
            if right_dir is None:
                # Heading rotated +35° (clockwise = right side)
                cos_a, sin_a = _math.cos(_angle_offset), _math.sin(_angle_offset)
                right_dir = (_hdx * cos_a + _hdy * sin_a,
                             _hdy * cos_a - _hdx * sin_a)

            left_dir = _roi_centroid_dir(left_roi_mask)
            if left_dir is None:
                cos_a, sin_a = _math.cos(-_angle_offset), _math.sin(-_angle_offset)
                left_dir = (_hdx * cos_a + _hdy * sin_a,
                            _hdy * cos_a - _hdx * sin_a)

            n_found = len(results['bfs_paths'])

            if n_found == 0:
                # Both missing — synthesize both
                for d, lbl in ((right_dir, 'Right-SYNTH'), (left_dir, 'Left-SYNTH')):
                    sp = _synth_path(d[0], d[1], lbl)
                    results['bfs_paths'].append(sp)
                    results['base_points'].append(sp[0])
                    results['tip_points'].append(sp[-1])
                    results['joint_points'].append(sp[int(len(sp) * 0.30)])
            elif n_found == 1:
                # One missing — figure out which side the found one belongs to
                # by checking dot product with right_dir vs left_dir
                found_tip = results['tip_points'][0]
                dx_ft = found_tip[0] - hp_x
                dy_ft = found_tip[1] - hp_y
                dot_right = dx_ft * right_dir[0] + dy_ft * right_dir[1]
                dot_left  = dx_ft * left_dir[0]  + dy_ft * left_dir[1]

                if dot_right >= dot_left:
                    # Found path is more right-aligned → missing is left
                    miss_dir, miss_lbl = left_dir, 'Left-SYNTH'
                else:
                    # Found path is more left-aligned → missing is right
                    # Insert synthetic at index 0 so order stays right=0, left=1
                    miss_dir, miss_lbl = right_dir, 'Right-SYNTH'

                sp = _synth_path(miss_dir[0], miss_dir[1], miss_lbl)

                if miss_lbl.startswith('Right'):
                    # Right is missing → insert at front
                    results['bfs_paths'].insert(0, sp)
                    results['base_points'].insert(0, sp[0])
                    results['tip_points'].insert(0, sp[-1])
                    results['joint_points'].insert(0, sp[int(len(sp) * 0.30)])
                else:
                    results['bfs_paths'].append(sp)
                    results['base_points'].append(sp[0])
                    results['tip_points'].append(sp[-1])
                    results['joint_points'].append(sp[int(len(sp) * 0.30)])

            # Now we are guaranteed len(results['bfs_paths']) >= 2

            # ── Stage M: Populate legacy single-antenna keys with antenna-A ──
            # (for backward compatibility with any code that reads 'bfs_path' etc.)
            results['bfs_path']       = results['bfs_paths'][0]
            results['base_point']     = results['base_points'][0]
            results['tip_point']      = results['tip_points'][0]
            results['joint_point']    = results['joint_points'][0]
            results['final_keypoints'] = {
                'base':  results['base_points'][0],
                'joint': results['joint_points'][0],
                'tip':   results['tip_points'][0],
            }

            print(f"[ANTENNA] Final: {len(results['bfs_paths'])} antenna path(s) found")
            
        except Exception as e:
            print(f"[ANTENNA] Error in processing pipeline: {e}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def zhang_suen_thinning(self, binary_img):
        """
        Fast skeletonization using cv2.ximgproc.thinning when available,
        falling back to a vectorised numpy implementation (no Python loops).

        Stage 1 fix: the old pure-Python loop implementation was O(N²) per
        iteration and took hundreds of ms on each antenna ROI.  ximgproc runs
        the same Zhang-Suen algorithm in C++ in < 1 ms.  The numpy fallback
        is ~10-20× faster than the old loop version.
        """
        if binary_img is None or binary_img.size == 0:
            return binary_img

        img = binary_img.astype(np.uint8)
        img = (img > 0).astype(np.uint8) * 255   # ensure 0/255

        # ── Fast path: OpenCV ximgproc (C++ implementation) ──────────────────
        if XIMGPROC_AVAILABLE:
            return cv2.ximgproc.thinning(img,
                                         thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

        # ── Fallback: vectorised numpy Zhang-Suen ────────────────────────────
        # Works on a 0/1 array; uses array slicing instead of Python loops.
        sk = (img > 0).astype(np.uint8)

        def _iter(sk, sub):
            """One sub-iteration of Zhang-Suen using boolean array ops."""
            r = sk[1:-1, 1:-1]
            # 8-neighbours in clockwise order: P2..P9
            p2 = sk[:-2, 1:-1];  p3 = sk[:-2, 2:]
            p4 = sk[1:-1, 2:];   p5 = sk[2:, 2:]
            p6 = sk[2:, 1:-1];   p7 = sk[2:, :-2]
            p8 = sk[1:-1, :-2];  p9 = sk[:-2, :-2]

            # Number of ON neighbours
            N  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

            # Number of 0→1 transitions in the cyclic sequence P2..P9
            S  = ((p2 == 0) & (p3 == 1)).astype(np.uint8) +                  ((p3 == 0) & (p4 == 1)).astype(np.uint8) +                  ((p4 == 0) & (p5 == 1)).astype(np.uint8) +                  ((p5 == 0) & (p6 == 1)).astype(np.uint8) +                  ((p6 == 0) & (p7 == 1)).astype(np.uint8) +                  ((p7 == 0) & (p8 == 1)).astype(np.uint8) +                  ((p8 == 0) & (p9 == 1)).astype(np.uint8) +                  ((p9 == 0) & (p2 == 1)).astype(np.uint8)

            cond_common = (r == 1) & (N >= 2) & (N <= 6) & (S == 1)

            if sub == 1:
                cond = cond_common & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                cond = cond_common & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)

            changed = int(np.sum(cond))
            sk[1:-1, 1:-1][cond] = 0
            return sk, changed

        for _ in range(200):
            sk, c1 = _iter(sk, 1)
            sk, c2 = _iter(sk, 2)
            if c1 + c2 == 0:
                break

        return (sk * 255).astype(np.uint8)
    
    def find_longest_antenna_path(self, skeleton, head_point, body_center):
        """
        Mature BFS antenna path extraction on Zhang-Suen skeleton.

        Improvements over the basic implementation:
        ─────────────────────────────────────────────
        0. PRE-CLEAN   – Iterative spur pruning removes branches shorter than
                         SPUR_MAX_LEN pixels.  This eliminates false endpoints
                         caused by morphological noise and diagonal staircase
                         artefacts, leaving only the main antenna trunk(s).

        1. BASE        – Skeleton pixel nearest to head_point (kp0).

        2. GAP-BRIDGE BFS – Standard 8-connected flood *with* a 2-px jump
                         tolerance.  When a pixel has no direct skeleton
                         neighbours, BFS searches a 5×5 window for the nearest
                         skeleton pixel to cross micro-gaps that would
                         otherwise truncate the path.

        3. ENDPOINT COLLECTION – Topological endpoints (1-neighbour) on the
                         pruned skeleton.

        4. TIP SELECTION – Blended score with ALL signals normalised to [0,1]:
                         A) Path length (hop-count from base)     — 50%
                         B) Local tangent alignment (last 8 hops) — 30%
                         C) Global heading projection (normalised)— 20%

        5. PATH RECONSTRUCTION – Parent-pointer backtrack.

        6. KP0 ANCHOR  – Prepend head_point so base is always kp0.

        7. PATH SMOOTHING – Moving-average smoothing (window=5) removes
                         staircase jagging on diagonals.

        Returns
        -------
        list : (x, y) from base (kp0) to tip. Empty if skeleton unusable.
        """
        if skeleton is None or skeleton.size == 0:
            print("[ANTENNA_PATH] Skeleton is None or empty")
            return []

        # ── Step 0: SPUR PRUNING ─────────────────────────────────────────
        # Iteratively remove short branches (endpoints with few neighbours)
        # until only the main trunk remains.  This is critical for clean
        # tip selection — without it, every diagonal staircase creates a
        # 1-2px spur with a false endpoint.
        SPUR_MAX_LEN = 6   # branches shorter than this get pruned
        pruned = skeleton.copy()
        h_sk, w_sk = pruned.shape

        for _prune_round in range(SPUR_MAX_LEN):
            removed = 0
            pts = np.argwhere(pruned > 0)
            for (y, x) in pts:
                if y <= 0 or y >= h_sk - 1 or x <= 0 or x >= w_sk - 1:
                    continue
                patch = pruned[y - 1:y + 2, x - 1:x + 2].copy()
                patch[1, 1] = 0
                if int(np.sum(patch > 0)) == 1:
                    # This is an endpoint — trace inward and check spur len
                    spur_len = 0
                    cx, cy = x, y
                    visited_spur = set()
                    while True:
                        visited_spur.add((cx, cy))
                        spur_len += 1
                        if spur_len > SPUR_MAX_LEN:
                            break   # long enough to keep
                        # Find the single neighbour
                        nbs = []
                        for ddx in (-1, 0, 1):
                            for ddy in (-1, 0, 1):
                                if ddx == 0 and ddy == 0:
                                    continue
                                nnx, nny = cx + ddx, cy + ddy
                                if (0 <= nnx < w_sk and 0 <= nny < h_sk and
                                        pruned[nny, nnx] > 0 and
                                        (nnx, nny) not in visited_spur):
                                    nbs.append((nnx, nny))
                        if len(nbs) != 1:
                            break   # junction or dead end
                        cx, cy = nbs[0]

                    if spur_len <= SPUR_MAX_LEN:
                        # Remove this spur
                        for (sx, sy) in visited_spur:
                            pruned[sy, sx] = 0
                            removed += 1
            if removed == 0:
                break

        # Use pruned skeleton if it still has enough pixels; otherwise keep original
        pruned_count = np.count_nonzero(pruned)
        if pruned_count >= 3:
            work_skel = pruned
            print(f"[ANTENNA_PATH] Spur pruning: {np.count_nonzero(skeleton)} → "
                  f"{pruned_count} px ({np.count_nonzero(skeleton) - pruned_count} pruned)")
        else:
            work_skel = skeleton
            print(f"[ANTENNA_PATH] Spur pruning removed too much — using original skeleton")

        skeleton_yx = np.argwhere(work_skel > 0)
        if len(skeleton_yx) == 0:
            print("[ANTENNA_PATH] No skeleton pixels found")
            return []

        print(f"[ANTENNA_PATH] {len(skeleton_yx)} skeleton pixels for BFS")

        # ── Resolve reference point and heading vector ──────────────────────
        if head_point is not None and len(head_point) == 2:
            ref_x, ref_y = int(head_point[0]), int(head_point[1])
        else:
            ref_x, ref_y = work_skel.shape[1] // 2, work_skel.shape[0] // 2

        heading_x, heading_y = 0.0, 0.0
        if body_center is not None and len(body_center) == 2:
            bx, by = int(body_center[0]), int(body_center[1])
            hx, hy = ref_x - bx, ref_y - by
            mag = np.sqrt(hx * hx + hy * hy)
            if mag > 1e-6:
                heading_x, heading_y = hx / mag, hy / mag

        # ── Step 1: BASE = skeleton pixel nearest to head_point ──────────
        dists_sq = ((skeleton_yx[:, 1] - ref_x) ** 2 +
                    (skeleton_yx[:, 0] - ref_y) ** 2)
        base_idx = int(np.argmin(dists_sq))
        base_pt  = (int(skeleton_yx[base_idx, 1]),
                    int(skeleton_yx[base_idx, 0]))
        print(f"[ANTENNA_PATH] BASE = {base_pt}  "
              f"(dist to kp0 = {np.sqrt(dists_sq[base_idx]):.1f}px)")

        # ── Step 2: GAP-BRIDGING BFS ─────────────────────────────────────
        # Standard 8-connected BFS, but when we reach a pixel whose 3×3
        # neighbourhood has no unvisited skeleton pixels, we search a 5×5
        # window (jump radius 2) to bridge micro-gaps of 1-2 px.
        GAP_BRIDGE_RADIUS = 2

        parent  = {base_pt: None}
        queue   = deque([base_pt])
        sk_h, sk_w = work_skel.shape

        while queue:
            x, y = queue.popleft()
            found_direct = False
            # Standard 8-connected neighbours first
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    if ddx == 0 and ddy == 0:
                        continue
                    nx, ny = x + ddx, y + ddy
                    if (0 <= nx < sk_w and 0 <= ny < sk_h and
                            work_skel[ny, nx] > 0 and
                            (nx, ny) not in parent):
                        parent[(nx, ny)] = (x, y)
                        queue.append((nx, ny))
                        found_direct = True

            # Gap-bridging: if no direct neighbours found, try 5×5 window
            if not found_direct:
                best_gap = None
                best_gap_dist = 999
                for ddx in range(-GAP_BRIDGE_RADIUS, GAP_BRIDGE_RADIUS + 1):
                    for ddy in range(-GAP_BRIDGE_RADIUS, GAP_BRIDGE_RADIUS + 1):
                        if abs(ddx) <= 1 and abs(ddy) <= 1:
                            continue  # already checked
                        nx, ny = x + ddx, y + ddy
                        if (0 <= nx < sk_w and 0 <= ny < sk_h and
                                work_skel[ny, nx] > 0 and
                                (nx, ny) not in parent):
                            d = abs(ddx) + abs(ddy)
                            if d < best_gap_dist:
                                best_gap_dist = d
                                best_gap = (nx, ny)
                if best_gap is not None:
                    parent[best_gap] = (x, y)
                    queue.append(best_gap)

        bfs_count = len(parent)
        skel_count = len(skeleton_yx)
        print(f"[ANTENNA_PATH] BFS reached {bfs_count}/{skel_count} skeleton pixels "
              f"({bfs_count / max(1, skel_count) * 100:.0f}% coverage)")

        # ── Step 3: Collect reachable ENDPOINTS ──────────────────────────
        endpoints = []
        for (x, y) in parent:
            if x <= 0 or x >= sk_w - 1 or y <= 0 or y >= sk_h - 1:
                continue
            patch = work_skel[y - 1:y + 2, x - 1:x + 2].copy()
            patch[1, 1] = 0
            if int(np.sum(patch > 0)) == 1:
                endpoints.append((x, y))

        if not endpoints:
            # Fallback: farthest reachable pixel from base
            endpoints = list(parent.keys())

        print(f"[ANTENNA_PATH] {len(endpoints)} endpoint candidates")

        # ── Step 4: TIP selection — fully normalised blended score ───────
        LOCAL_WIN = 8

        def _hop_count(ep):
            cur, n = ep, 0
            while cur is not None:
                cur = parent.get(cur)
                n += 1
            return n

        def _local_tangent(ep):
            pts, cur = [], ep
            for _ in range(LOCAL_WIN + 1):
                pts.append(cur)
                nxt = parent.get(cur)
                if nxt is None:
                    break
                cur = nxt
            if len(pts) < 2:
                return 0.0, 0.0
            dx = pts[0][0] - pts[-1][0]
            dy = pts[0][1] - pts[-1][1]
            mag = np.sqrt(dx * dx + dy * dy)
            if mag < 1e-6:
                return 0.0, 0.0
            return dx / mag, dy / mag

        tip_candidates = [ep for ep in endpoints if ep != base_pt]
        if not tip_candidates:
            print("[ANTENNA_PATH] No tip candidates — returning base-only path")
            return [base_pt]

        hop_counts = {ep: _hop_count(ep) for ep in tip_candidates}
        max_hops   = max(hop_counts.values()) or 1

        # Pre-compute max projection for signal C normalisation
        projs = {}
        for ep in tip_candidates:
            if heading_x != 0.0 or heading_y != 0.0:
                projs[ep] = ((ep[0] - ref_x) * heading_x +
                             (ep[1] - ref_y) * heading_y)
            else:
                projs[ep] = np.sqrt((ep[0] - base_pt[0]) ** 2 +
                                    (ep[1] - base_pt[1]) ** 2)

        proj_min = min(projs.values())
        proj_max = max(projs.values())
        proj_range = proj_max - proj_min if (proj_max - proj_min) > 1e-6 else 1.0

        best_score = -float('inf')
        tip_pt     = None

        for ep in tip_candidates:
            hops = hop_counts[ep]

            # A: path-length score [0,1]
            score_len = hops / max_hops

            # B: local tangent alignment [0,1]
            if heading_x != 0.0 or heading_y != 0.0:
                tx, ty     = _local_tangent(ep)
                tang_align = tx * heading_x + ty * heading_y   # [-1,1]
                score_tang = (tang_align + 1.0) / 2.0          # [0,1]
            else:
                score_tang = 0.5

            # C: global heading projection — NORMALISED to [0,1]
            score_proj = (projs[ep] - proj_min) / proj_range

            blended = 0.50 * score_len + 0.30 * score_tang + 0.20 * score_proj

            if blended > best_score:
                best_score = blended
                tip_pt     = ep

        if tip_pt is None:
            print("[ANTENNA_PATH] Could not determine tip point")
            return []

        print(f"[ANTENNA_PATH] TIP  = {tip_pt}  "
              f"(hops={hop_counts.get(tip_pt, 0)}, score={best_score:.3f})")

        # ── Step 5: Reconstruct path BASE → TIP ─────────────────────────
        if tip_pt not in parent:
            path = self._trace_skeleton_path(work_skel, base_pt, tip_pt)
        else:
            path = []
            cur  = tip_pt
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()

        if not path:
            return []

        # ── Step 6: Hard-anchor base to head_point (kp0) ────────────────
        head_xy = (ref_x, ref_y)
        if path[0] != head_xy:
            path.insert(0, head_xy)

        # ── Step 7: PATH SMOOTHING — moving average ─────────────────────
        # Removes staircase jagging on diagonal skeleton segments.
        # Uses a small window (5) to preserve natural curvature while
        # eliminating 1-px zigzag noise.
        SMOOTH_WIN = 5
        if len(path) > SMOOTH_WIN + 2:
            half = SMOOTH_WIN // 2
            smoothed = [path[0]]   # keep kp0 anchor untouched
            for i in range(1, len(path) - 1):
                lo = max(1, i - half)          # never smooth past kp0 anchor
                hi = min(len(path) - 1, i + half + 1)
                window = path[lo:hi]
                avg_x = int(round(sum(p[0] for p in window) / len(window)))
                avg_y = int(round(sum(p[1] for p in window) / len(window)))
                smoothed.append((avg_x, avg_y))
            smoothed.append(path[-1])          # keep tip untouched
            path = smoothed

        print(f"[ANTENNA_PATH] Final path: {len(path)} points  "
              f"base(kp0)={path[0]}  tip={path[-1]}")
        return path

    def _trace_skeleton_path(self, skeleton, start_pt, end_pt):
        """
        BFS path trace strictly along skeleton pixels from start_pt to end_pt.
        Uses parent-pointer tracking for efficient path reconstruction.

        This method is used as a fallback by find_longest_antenna_path and
        also kept for backward compatibility with other callers.

        Returns
        -------
        list : (x, y) coordinates from start_pt to end_pt, or [] if unreachable.
        """
        if skeleton is None or start_pt is None or end_pt is None:
            return []

        # BFS with parent-pointer tracking (no per-node path list in queue)
        parent = {start_pt: None}
        queue  = deque([start_pt])

        while queue:
            x, y = queue.popleft()

            if (x, y) == end_pt:
                # Reconstruct path by backtracking
                path = []
                cur  = end_pt
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < skeleton.shape[1] and
                            0 <= ny < skeleton.shape[0] and
                            skeleton[ny, nx] > 0 and
                            (nx, ny) not in parent):
                        parent[(nx, ny)] = (x, y)
                        queue.append((nx, ny))

        return []  # end_pt not reachable from start_pt
    
    def detect_endpoints(self, skeleton, head_point, body_center):
        """
        DEPRECATED: Use find_longest_antenna_path instead.
        Kept for backward compatibility.
        """
        # This is now just a wrapper that uses the new method
        path = self.find_longest_antenna_path(skeleton, head_point, body_center)
        
        if len(path) < 2:
            return [], None, None
        
        # Extract base and tip from path
        base_pt = path[0]
        tip_pt = path[-1]
        
        # Find all endpoints for visualization
        skeleton_points = np.argwhere(skeleton > 0)
        endpoints = []
        for pt in skeleton_points:
            y, x = pt
            if y > 0 and y < skeleton.shape[0]-1 and x > 0 and x < skeleton.shape[1]-1:
                neighbors_region = skeleton[y-1:y+2, x-1:x+2].copy()
                neighbors_region[1, 1] = 0
                neighbor_count = np.sum(neighbors_region > 0)
                if neighbor_count == 1:
                    endpoints.append((x, y))
        
        return endpoints, base_pt, tip_pt
    
    def bfs_trace_path(self, skeleton, start_pt, end_pt):
        """
        DEPRECATED: This is now handled by find_longest_antenna_path.
        Kept for backward compatibility.
        """
        return self._trace_skeleton_path(skeleton, start_pt, end_pt)


def create_research_visualization_tab():
    """
    Create the "For Research" tab with morphological pipeline visualization.
    
    Returns:
        tuple: (tab_widget, research_viz_widget)
    """
    tab = QWidget()
    tab.setStyleSheet("background-color: #1a1a1a;")
    
    layout = QVBoxLayout(tab)
    layout.setContentsMargins(0, 0, 0, 0)
    
    # Create the research visualization widget
    research_widget = ResearchVisualizationWidget()
    
    # Wrap in a scroll area to make panels scrollable
    scroll_area = QScrollArea()
    scroll_area.setWidget(research_widget)
    scroll_area.setWidgetResizable(True)
    scroll_area.setStyleSheet("""
        QScrollArea {
            background-color: #1a1a1a;
            border: none;
        }
        QScrollBar:vertical {
            background: #2d2d2d;
            width: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background: #00b4d8;
            border-radius: 6px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background: #48cae4;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QScrollBar:horizontal {
            background: #2d2d2d;
            height: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:horizontal {
            background: #00b4d8;
            border-radius: 6px;
            min-width: 20px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #48cae4;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
    """)
    
    layout.addWidget(scroll_area)
    
    return tab, research_widget


# ============================================================================
# MORPHOLOGICAL PROCESSING PIPELINE (for research visualization)
# ============================================================================

class MorphologicalPipelineProcessor:
    """
    Computes all intermediate stages of morphological processing
    for research visualization and paper documentation.
    
    Pipeline stages:
    A. Original Grayscale Frame - raw input showing bee with dark antennae
    B. Vertical Tophat - extracts vertical structures (1×7 kernel)
    C. Horizontal Tophat - extracts horizontal structures (7×1 kernel)
    D. Diagonal Tophat - extracts diagonal structures (5×5 cross kernel)
    E. Combined Maximum - max(B, C, D) - all orientations
    F. Final Binary - Otsu + Gaussian + morphological opening
    """
    
    def __init__(self, kernel_size=15):
        # =====================================================================
        # DYNAMIC kernel sizing based on antenna thickness
        # Small kernels (3-7px) for thin antennae
        # Large kernels (15-21px) for thick antennae
        # =====================================================================
        
        self.kernel_size = kernel_size
        
        # To detect VERTICAL structures: use HORIZONTAL kernel
        self.kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
        
        # To detect HORIZONTAL structures: use VERTICAL kernel
        self.kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
        
        # Diagonal kernel scales proportionally (60% of main kernel)
        diag_size = max(3, int(kernel_size * 0.6))
        if diag_size % 2 == 0:  # Must be odd for cross kernel
            diag_size += 1
        self.kernel_diagonal = cv2.getStructuringElement(cv2.MORPH_CROSS, (diag_size, diag_size))
        
        # Opening kernel for noise removal in final binary
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        
        # Layer 2+3: Temporal antenna stabilisation
        self.temporal_tracker = TemporalAntennaTracker()
    
    def _draw_dashed_rectangle(self, img, pt1, pt2, color, thickness=1, dash_length=10):
        """Draw a dashed rectangle"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Top edge
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        
        # Bottom edge
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        
        # Left edge
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        
        # Right edge
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
    
    def _draw_roi_triangles_on_frame(self, frame, all_bee_data, worker_instance):
        """
        Draw ROI triangles on a visualization frame.
        
        Args:
            frame: BGR image to draw on
            all_bee_data: Dict of {bee_id: (keypoints, box)}
            worker_instance: InferenceWorker instance with get_angled_roi_lines method and roi_thickness
        """
        if all_bee_data is None or len(all_bee_data) == 0:
            return frame
        
        # Get thickness from worker instance (0 = hidden)
        thickness = getattr(worker_instance, 'roi_thickness', 1)
        if thickness == 0:
            return frame  # Don't draw if thickness is 0
        
        for bee_id, (keypoints, box) in all_bee_data.items():
            if box is None or keypoints is None:
                continue
            
            try:
                # Get ROI polygon points from worker (returns 6 values with polygon ROIs)
                result = worker_instance.get_angled_roi_lines(keypoints, box, bee_id)
                if len(result) == 6:
                    head_pos, center_end, left_end, right_end, left_roi, right_roi = result
                else:
                    head_pos, center_end, left_end, right_end = result[:4]
                    left_roi = [head_pos, center_end, left_end]
                    right_roi = [head_pos, center_end, right_end]
                
                # Draw polygon ROI outlines with dynamic thickness
                if left_roi is not None and len(left_roi) >= 3:
                    left_poly = np.array(left_roi, dtype=np.int32)
                    cv2.polylines(frame, [left_poly], True, (255, 200, 100), thickness)
                if right_roi is not None and len(right_roi) >= 3:
                    right_poly = np.array(right_roi, dtype=np.int32)
                    cv2.polylines(frame, [right_poly], True, (100, 200, 255), thickness)
            except Exception as e:
                # Skip if triangle drawing fails
                pass
        
        return frame
    
    def compute_all_stages(self, frame_gray, frame_id=0, all_bee_data=None, antenna_pipeline=None, worker_instance=None):
        """
        Compute all intermediate morphological processing stages.
        
        Args:
            frame_gray: Grayscale frame
            frame_id: Current frame number
            all_bee_data: Dict of {bee_id: (keypoints, box)} for overlay visualization
            antenna_pipeline: AntennaProcessingPipeline instance for antenna-specific stages
            worker_instance: InferenceWorker instance for ROI triangle computation
        
        Returns dict with:
        - original_grayscale: Original frame
        - vertical_tophat: Vertical tophat  
        - horizontal_tophat: Horizontal tophat
        - diagonal_tophat: Diagonal tophat
        - combined_max: Combined maximum
        - final_binary: Final binary
        - annotated_opening: Annotated with overlays (G)
        - antenna_original_blob: Original antenna blob (H)
        - antenna_after_closing: After morphological closing (I)
        - antenna_skeleton: Zhang-Suen skeleton (J)
        - antenna_endpoints: Endpoint detection visualization (K)
        - antenna_bfs_path: BFS path tracing visualization (L)
        - antenna_final_keypoints: Final keypoints visualization (M)
        """
        start_time = time.perf_counter()
        
        if frame_gray is None:
            return None
        
        # =====================================================================
        # PANEL A: Original Grayscale Frame
        # Shows the raw input - bee with antennae as dark thin structures
        # =====================================================================
        original_gray = frame_gray.copy()
        
        # Invert: dark antennae become bright for tophat processing
        inverted = 255 - frame_gray
        
        # =====================================================================
        # PANEL B: Vertical Tophat Output
        # Detects VERTICAL antenna segments (going ↑↓)
        # Uses HORIZONTAL kernel (15×1) - vertical lines don't fit → appear bright
        # Result: Bright where VERTICAL antenna segments are, dark elsewhere
        # =====================================================================
        tophat_v_raw = cv2.morphologyEx(inverted, cv2.MORPH_TOPHAT, self.kernel_vertical)
        
        # =====================================================================
        # PANEL C: Horizontal Tophat Output
        # Detects HORIZONTAL antenna segments (going ←→)
        # Uses VERTICAL kernel (1×15) - horizontal lines don't fit → appear bright
        # Result: Bright where HORIZONTAL antenna segments are, dark elsewhere
        # =====================================================================
        tophat_h_raw = cv2.morphologyEx(inverted, cv2.MORPH_TOPHAT, self.kernel_horizontal)
        
        # =====================================================================
        # PANEL D: Diagonal Tophat Output
        # Uses 9×9 cross kernel
        # Result: Bright where DIAGONAL antenna segments are, dark elsewhere
        # Different from B and C
        # =====================================================================
        tophat_d_raw = cv2.morphologyEx(inverted, cv2.MORPH_TOPHAT, self.kernel_diagonal)
        
        # =====================================================================
        # PANEL E: Combined Tophat (Element-wise Maximum)
        # Union of B, C, D - shows ALL antenna orientations
        # More bright structures than any individual panel
        # =====================================================================
        combined_raw = cv2.max(tophat_v_raw, cv2.max(tophat_h_raw, tophat_d_raw))
        
        # =====================================================================
        # NORMALIZE each tophat to 0-255 for visibility
        # This makes the faint structures clearly visible
        # =====================================================================
        def normalize_to_255(img):
            """Normalize image to full 0-255 range"""
            min_val = float(img.min())
            max_val = float(img.max())
            if max_val - min_val < 1:
                return np.zeros_like(img)
            normalized = ((img.astype(np.float32) - min_val) / (max_val - min_val) * 255)
            return normalized.astype(np.uint8)
        
        tophat_vertical = normalize_to_255(tophat_v_raw)
        tophat_horizontal = normalize_to_255(tophat_h_raw)
        tophat_diagonal = normalize_to_255(tophat_d_raw)
        combined = normalize_to_255(combined_raw)
        
        # Convert grayscale panels to BGR for ROI triangle drawing
        tophat_vertical_bgr = cv2.cvtColor(tophat_vertical, cv2.COLOR_GRAY2BGR)
        tophat_horizontal_bgr = cv2.cvtColor(tophat_horizontal, cv2.COLOR_GRAY2BGR)
        tophat_diagonal_bgr = cv2.cvtColor(tophat_diagonal, cv2.COLOR_GRAY2BGR)
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        
        # Draw ROI triangles on all panels if worker_instance is provided
        if worker_instance is not None:
            tophat_vertical_bgr = self._draw_roi_triangles_on_frame(tophat_vertical_bgr, all_bee_data, worker_instance)
            tophat_horizontal_bgr = self._draw_roi_triangles_on_frame(tophat_horizontal_bgr, all_bee_data, worker_instance)
            tophat_diagonal_bgr = self._draw_roi_triangles_on_frame(tophat_diagonal_bgr, all_bee_data, worker_instance)
            combined_bgr = self._draw_roi_triangles_on_frame(combined_bgr, all_bee_data, worker_instance)
        
        # =====================================================================
        # PANEL F: Final Binary Image
        # 1. Gaussian smoothing - fills small gaps
        # 2. Per-ROI percentile thresholding OR Otsu (auto)
        #    Percentile mode: within each antenna ROI polygon, keep the top N%
        #    brightest tophat pixels. Antennae are always the strongest local
        #    response, so they survive regardless of absolute brightness.
        # 3. Morphological opening - removes noise
        # Result: Pure black/white, clean antenna structures
        # =====================================================================
        smoothed = cv2.GaussianBlur(combined_raw, (3, 3), 0.5)
        
        # Read percentile from worker (0 = Auto/Otsu, >0 = keep top N%)
        roi_percentile = 0.0
        if worker_instance is not None and hasattr(worker_instance, 'binary_threshold'):
            roi_percentile = worker_instance.binary_threshold
        
        if roi_percentile > 0 and all_bee_data and worker_instance is not None:
            # ── PER-ROI PERCENTILE + STRONGEST-COMPONENT FILTER ──
            # For each antenna ROI:
            #   1. Percentile threshold → candidate pixels
            #   2. Connected components analysis
            #   3. Keep ONLY the single strongest component (highest mean tophat)
            #      — this is the antenna. All weaker blobs = noise → discarded.
            binary = np.zeros_like(smoothed)
            otsu_thresh = 0
            
            for bee_id, (keypoints, box) in all_bee_data.items():
                if box is None or keypoints is None:
                    continue
                try:
                    result_roi = worker_instance.get_angled_roi_lines(keypoints, box, bee_id)
                    if len(result_roi) == 6:
                        _, _, _, _, left_roi, right_roi = result_roi
                    else:
                        head_pos, center_end, left_end, right_end = result_roi[:4]
                        left_roi = [head_pos, center_end, left_end]
                        right_roi = [head_pos, center_end, right_end]
                    
                    # Process each ROI side independently
                    for roi_poly in [left_roi, right_roi]:
                        if roi_poly is None or len(roi_poly) < 3:
                            continue
                        
                        # Create mask for this ROI side
                        side_mask = np.zeros_like(smoothed)
                        cv2.fillPoly(side_mask, [np.array(roi_poly, dtype=np.int32)], 255)
                        
                        # Extract tophat values within this ROI
                        roi_pixels = smoothed[side_mask > 0]
                        if len(roi_pixels) < 10:
                            continue
                        
                        # Step 1: Percentile threshold
                        cutoff = np.percentile(roi_pixels, 100.0 - roi_percentile)
                        cutoff = max(cutoff, 1)
                        
                        # Candidate binary within this ROI
                        side_binary = ((smoothed >= cutoff) & (side_mask > 0)).astype(np.uint8) * 255
                        
                        # Step 2: Connected components — find the strongest one
                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                            side_binary, connectivity=8
                        )
                        
                        if num_labels <= 1:
                            # Only background, nothing survived
                            continue
                        
                        # Step 3: Score each component by mean tophat intensity
                        best_label = -1
                        best_score = -1
                        for lbl in range(1, num_labels):  # skip background (0)
                            component_mask = (labels == lbl)
                            mean_intensity = float(smoothed[component_mask].mean())
                            area = stats[lbl, cv2.CC_STAT_AREA]
                            # Score = mean intensity × sqrt(area) — rewards both brightness and size
                            score = mean_intensity * np.sqrt(area)
                            if score > best_score:
                                best_score = score
                                best_label = lbl
                        
                        # Step 4: Keep only the strongest component
                        if best_label > 0:
                            strongest = ((labels == best_label).astype(np.uint8)) * 255
                            binary = cv2.bitwise_or(binary, strongest)
                            
                except Exception:
                    pass
        else:
            # Auto (Otsu) - original behavior, global threshold
            otsu_thresh, binary = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        final_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel_open)
        
        # Convert final_binary to BGR for ROI triangle drawing
        final_binary_bgr = cv2.cvtColor(final_binary, cv2.COLOR_GRAY2BGR)
        if worker_instance is not None:
            final_binary_bgr = self._draw_roi_triangles_on_frame(final_binary_bgr, all_bee_data, worker_instance)
        
        # Calculate stats
        processing_time = (time.perf_counter() - start_time) * 1000
        white_pixel_count = cv2.countNonZero(final_binary)
        v_max = int(tophat_v_raw.max())
        h_max = int(tophat_h_raw.max())
        d_max = int(tophat_d_raw.max())
        
        # =====================================================================
        # PANEL G: Annotated Opening - Shows morphological opening with overlays
        # 1. Convert binary to BGR for colored annotations
        # 2. Draw ROI triangles only (no bboxes, no keypoints, no skeleton)
        # Result: Clean binary mask with ROI triangles for antenna detection regions
        # =====================================================================
        annotated_opening = cv2.cvtColor(final_binary, cv2.COLOR_GRAY2BGR)
        
        # Draw ROI triangles only
        if all_bee_data is not None and len(all_bee_data) > 0:
            if worker_instance is not None:
                annotated_opening = self._draw_roi_triangles_on_frame(annotated_opening, all_bee_data, worker_instance)
        
        # =====================================================================
        # NEW: ANTENNA PROCESSING PIPELINE VISUALIZATION (Panels H-M)
        # Process antenna regions and generate visualization for each stage
        # =====================================================================
        antenna_stages = self._generate_antenna_processing_stages(
            final_binary, frame_gray, all_bee_data, antenna_pipeline,
            worker_instance=worker_instance
        )
        
        result = {
            'original_grayscale': original_gray,
            'vertical_tophat': tophat_vertical_bgr,
            'horizontal_tophat': tophat_horizontal_bgr,
            'diagonal_tophat': tophat_diagonal_bgr,
            'combined_max': combined_bgr,
            'final_binary': final_binary_bgr,
            'annotated_opening': annotated_opening,
            'stats': {
                'frame_id': frame_id,
                'otsu_threshold': int(otsu_thresh),
                'manual_threshold': roi_percentile > 0,
                'roi_percentile': roi_percentile,
                'white_pixel_count': white_pixel_count,
                'processing_time_ms': processing_time,
                'v_max': v_max,
                'h_max': h_max,
                'd_max': d_max
            }
        }
        
        # Add antenna processing stages if available
        if antenna_stages:
            result.update(antenna_stages)
        
        return result
    
    def _generate_antenna_processing_stages(self, final_binary, frame_gray, all_bee_data, antenna_pipeline,
                                              worker_instance=None):
        """
        Generate visualization images for antenna processing stages H-M.
        Shows FULL FRAME with overlays — not cutout ROI regions.
        
        Three-layer stabilisation:
          Layer 1: ROI-guided skeleton split (needs worker_instance for ROI geometry)
          Layer 2: EMA temporal smoothing (via self.temporal_tracker)
          Layer 3: Confidence-decayed gap fill (via self.temporal_tracker)
        
        Returns dict with visualization images for:
        - antenna_original_blob (H): Full frame with original blob highlighted
        - antenna_after_closing (I): Full frame with closing result
        - antenna_skeleton (J): Full frame with skeleton overlay
        - antenna_endpoints (K): Full frame with endpoints marked
        - antenna_bfs_path (L): Full frame with BFS path drawn
        - antenna_final_keypoints (M): Full frame with final keypoints
        """
        if antenna_pipeline is None or all_bee_data is None or len(all_bee_data) == 0:
            return {}
        
        h, w = frame_gray.shape
        
        # Full-frame base images
        binary_bgr = cv2.cvtColor(final_binary, cv2.COLOR_GRAY2BGR)
        gray_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        
        # Initialize full-frame canvases
        stages = {
            'antenna_original_blob': binary_bgr.copy(),
            'antenna_after_closing': binary_bgr.copy(),
            'antenna_skeleton': binary_bgr.copy(),
            'antenna_endpoints': gray_bgr.copy(),
            'antenna_bfs_path': gray_bgr.copy(),
            'antenna_final_keypoints': gray_bgr.copy(),
        }
        
        try:
            for bee_id, (keypoints, box) in all_bee_data.items():
                if keypoints is None or box is None:
                    continue
                
                head_point = (int(keypoints[0][0]), int(keypoints[0][1])) if len(keypoints) > 0 else None
                body_center_x = int(np.mean([kp[0] for kp in keypoints[:5] if len(kp) >= 2]))
                body_center_y = int(np.mean([kp[1] for kp in keypoints[:5] if len(kp) >= 2]))
                body_center = (body_center_x, body_center_y)

                if head_point is None:
                    continue

                # ── Compute heading vector (body_center → head_point) ──────────
                # Forward-facing direction of the bee.
                hdx = head_point[0] - body_center[0]
                hdy = head_point[1] - body_center[1]
                hmag = np.sqrt(hdx * hdx + hdy * hdy)
                if hmag > 1e-6:
                    hdx /= hmag
                    hdy /= hmag
                else:
                    hdx, hdy = 0.0, -1.0  # fallback: facing up

                # ── Build axis-aligned ROI centered on head_point
                #    • FORWARD_PAD px ahead along heading (where antennas go)
                #    • BACK_PAD px behind head (small buffer)
                #    • SIDE_PAD px perpendicular (covers both left & right antenna)
                FORWARD_PAD = 130
                BACK_PAD    = 20
                SIDE_PAD    = 80

                hx, hy = head_point
                fwd_x  = int(hx + hdx * FORWARD_PAD)
                fwd_y  = int(hy + hdy * FORWARD_PAD)
                back_x = int(hx - hdx * BACK_PAD)
                back_y = int(hy - hdy * BACK_PAD)

                rx_min = min(fwd_x, back_x) - SIDE_PAD
                rx_max = max(fwd_x, back_x) + SIDE_PAD
                ry_min = min(fwd_y, back_y) - SIDE_PAD
                ry_max = max(fwd_y, back_y) + SIDE_PAD

                head_region_x1 = max(0, rx_min)
                head_region_x2 = min(w, rx_max)
                head_region_y1 = max(0, ry_min)
                head_region_y2 = min(h, ry_max)

                if head_region_x2 <= head_region_x1 or head_region_y2 <= head_region_y1:
                    continue

                # Offset for mapping local → global coordinates
                ox, oy = head_region_x1, head_region_y1
                
                antenna_region = final_binary[head_region_y1:head_region_y2, head_region_x1:head_region_x2].copy()
                
                if antenna_region.size == 0:
                    continue
                
                # Local coordinates for pipeline
                local_head = (head_point[0] - ox if head_point else antenna_region.shape[1] // 2,
                              head_point[1] - oy if head_point else antenna_region.shape[0] // 2)
                local_body_center = (body_center[0] - ox, body_center[1] - oy)
                
                # ── LAYER 1: Build local ROI masks from worker geometry ────────
                left_roi_local = None
                right_roi_local = None
                if worker_instance is not None and hasattr(worker_instance, 'get_angled_roi_lines'):
                    try:
                        roi_result = worker_instance.get_angled_roi_lines(keypoints, box, bee_id)
                        if len(roi_result) == 6:
                            _, _, _, _, l_roi_poly, r_roi_poly = roi_result
                        else:
                            hp, ce, le, re = roi_result[:4]
                            l_roi_poly = [hp, ce, le]
                            r_roi_poly = [hp, ce, re]

                        region_h = head_region_y2 - head_region_y1
                        region_w = head_region_x2 - head_region_x1

                        # Convert global ROI polygons to local coords
                        if l_roi_poly is not None and len(l_roi_poly) >= 3:
                            l_local = np.array([(int(p[0] - ox), int(p[1] - oy))
                                                for p in l_roi_poly], dtype=np.int32)
                            left_roi_local = np.zeros((region_h, region_w), dtype=np.uint8)
                            cv2.fillPoly(left_roi_local, [l_local], 255)

                        if r_roi_poly is not None and len(r_roi_poly) >= 3:
                            r_local = np.array([(int(p[0] - ox), int(p[1] - oy))
                                                for p in r_roi_poly], dtype=np.int32)
                            right_roi_local = np.zeros((region_h, region_w), dtype=np.uint8)
                            cv2.fillPoly(right_roi_local, [r_local], 255)
                    except Exception as _roi_err:
                        print(f"[LAYER1] ROI mask build failed: {_roi_err}")

                # ── Sync EMA alpha from worker slider ──────────────────────────
                if worker_instance is not None and hasattr(worker_instance, 'ema_alpha'):
                    self.temporal_tracker.ema_alpha = worker_instance.ema_alpha

                # Process through antenna pipeline (Layer 1: ROI-guided)
                antenna_results = antenna_pipeline.process_antenna(
                    antenna_region, local_head, local_body_center,
                    left_roi_mask=left_roi_local,
                    right_roi_mask=right_roi_local
                )
                
                # Helper: map local point → global
                def to_global(pt):
                    if pt is None:
                        return None
                    return (pt[0] + ox, pt[1] + oy)
                
                # Unique color per bee
                hue = (bee_id * 50) % 180
                cb = cv2.cvtColor(np.uint8([[[hue, 220, 240]]]), cv2.COLOR_HSV2BGR)[0][0]
                bee_color = (int(cb[0]), int(cb[1]), int(cb[2]))
                
                # ── H: Original blob overlay ──
                if antenna_results['original_blob'] is not None:
                    blob = antenna_results['original_blob']
                    blob_mask = blob > 0
                    # Highlight blob pixels in bee color on full frame
                    roi_slice = stages['antenna_original_blob'][oy:oy+blob.shape[0], ox:ox+blob.shape[1]]
                    roi_slice[blob_mask] = bee_color
                
                # ── I: After closing overlay ──
                if antenna_results['after_closing'] is not None:
                    closed = antenna_results['after_closing']
                    closed_mask = closed > 0
                    roi_slice = stages['antenna_after_closing'][oy:oy+closed.shape[0], ox:ox+closed.shape[1]]
                    roi_slice[closed_mask] = bee_color
                
                # ── J: Skeleton overlay ──
                if antenna_results['skeleton'] is not None:
                    skel = antenna_results['skeleton']
                    skel_mask = skel > 0
                    roi_slice = stages['antenna_skeleton'][oy:oy+skel.shape[0], ox:ox+skel.shape[1]]
                    roi_slice[skel_mask] = bee_color
                
                # ── K: Endpoint detection — BOTH antennas ─────────────────
                # Skeleton in grey. Each antenna: base=GREEN(B1/B2), tip=RED(T1/T2).
                # Branch junctions (rare) shown as small cyan dots.
                if antenna_results['skeleton'] is not None:
                    skel      = antenna_results['skeleton']
                    skel_mask = skel > 0
                    roi_slice = stages['antenna_endpoints'][
                        oy:oy + skel.shape[0], ox:ox + skel.shape[1]]
                    roi_slice[skel_mask] = [180, 180, 180]   # grey skeleton

                    # Collect all base/tip locals to skip in branch endpoint loop
                    used_pts = set(antenna_results['base_points']) |                                set(antenna_results['tip_points'])

                    # Branch-junction endpoints (cyan, small)
                    for ep in antenna_results['endpoint_list']:
                        if ep in used_pts:
                            continue
                        gp = to_global(ep)
                        if gp:
                            cv2.circle(stages['antenna_endpoints'],
                                       gp, 3, (0, 255, 255), -1)

                    # Per-antenna: base GREEN, tip RED, labelled 1/2
                    for ant_i, (bp, tp) in enumerate(zip(
                            antenna_results['base_points'],
                            antenna_results['tip_points'])):
                        lbl = str(ant_i + 1)
                        bg = to_global(bp)
                        tg = to_global(tp)
                        if bg:
                            cv2.circle(stages['antenna_endpoints'],
                                       bg, 7, (0, 255, 0), -1)
                            cv2.circle(stages['antenna_endpoints'],
                                       bg, 8, (255, 255, 255), 1)
                            cv2.putText(stages['antenna_endpoints'],
                                        f"B{lbl}",
                                        (bg[0] + 9, bg[1] - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                        (0, 255, 0), 1)
                        if tg:
                            cv2.circle(stages['antenna_endpoints'],
                                       tg, 7, (0, 0, 255), -1)
                            cv2.circle(stages['antenna_endpoints'],
                                       tg, 8, (255, 255, 255), 1)
                            cv2.putText(stages['antenna_endpoints'],
                                        f"T{lbl}",
                                        (tg[0] + 9, tg[1] - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                        (0, 0, 255), 1)

                # ── L: BFS path tracing — BOTH antennas, blue→red gradient ───
                for ant_i, path in enumerate(antenna_results['bfs_paths']):
                    if len(path) < 2:
                        continue
                    n_pts = len(path)
                    lbl   = str(ant_i + 1)
                    for i in range(n_pts - 1):
                        gp1 = to_global(path[i])
                        gp2 = to_global(path[i + 1])
                        if gp1 and gp2:
                            t = i / max(1, n_pts - 1)   # 0=base, 1=tip
                            r = int(255 * t)
                            b = int(255 * (1.0 - t))
                            g = int(255 * (1.0 - abs(2.0 * t - 1.0)))
                            cv2.line(stages['antenna_bfs_path'],
                                     gp1, gp2, (b, g, r), 2)
                    start_g = to_global(path[0])
                    end_g   = to_global(path[-1])
                    if start_g:
                        cv2.circle(stages['antenna_bfs_path'],
                                   start_g, 6, (0, 255, 0), -1)
                        cv2.putText(stages['antenna_bfs_path'], f"B{lbl}",
                                    (start_g[0] + 7, start_g[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                    (0, 255, 0), 1)
                    if end_g:
                        cv2.circle(stages['antenna_bfs_path'],
                                   end_g, 6, (0, 0, 255), -1)
                        cv2.putText(stages['antenna_bfs_path'], f"T{lbl}",
                                    (end_g[0] + 7, end_g[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                    (0, 0, 255), 1)

                # ── M: Final keypoints — BOTH antennas ───────────────────────
                for ant_i, (bp, jp, tp) in enumerate(zip(
                        antenna_results['base_points'],
                        antenna_results['joint_points'],
                        antenna_results['tip_points'])):
                    lbl   = str(ant_i + 1)
                    base_g  = to_global(bp)
                    joint_g = to_global(jp)
                    tip_g   = to_global(tp)

                    # Connecting line: base → joint → tip
                    if base_g and joint_g:
                        cv2.line(stages['antenna_final_keypoints'],
                                 base_g, joint_g, (150, 150, 150), 2)
                    if joint_g and tip_g:
                        cv2.line(stages['antenna_final_keypoints'],
                                 joint_g, tip_g, (150, 150, 150), 2)

                    if base_g:
                        cv2.circle(stages['antenna_final_keypoints'],
                                   base_g, 7, (0, 255, 0), -1)
                        cv2.circle(stages['antenna_final_keypoints'],
                                   base_g, 8, (255, 255, 255), 2)
                        cv2.putText(stages['antenna_final_keypoints'],
                                    f"BASE{lbl}",
                                    (base_g[0] + 9, base_g[1] - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                    (0, 255, 0), 1)
                    if joint_g:
                        cv2.circle(stages['antenna_final_keypoints'],
                                   joint_g, 7, (0, 255, 255), -1)
                        cv2.circle(stages['antenna_final_keypoints'],
                                   joint_g, 8, (255, 255, 255), 2)
                        cv2.putText(stages['antenna_final_keypoints'],
                                    f"JNT{lbl}",
                                    (joint_g[0] + 9, joint_g[1] - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                    (0, 255, 255), 1)
                    if tip_g:
                        cv2.circle(stages['antenna_final_keypoints'],
                                   tip_g, 7, (0, 0, 255), -1)
                        cv2.circle(stages['antenna_final_keypoints'],
                                   tip_g, 8, (255, 255, 255), 2)
                        cv2.putText(stages['antenna_final_keypoints'],
                                    f"TIP{lbl}",
                                    (tip_g[0] + 9, tip_g[1] - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                    (0, 0, 255), 1)
                
                # ── LAYERS 2+3: Temporal EMA + Gap Fill → Write back ──────────
                #
                # Keypoint layout (matches AntennaGeometry):
                #   kp[5]  right scape   (antenna 0 joint  @ 30%)
                #   kp[6]  right flagellum (antenna 0 tip   @ 100%)
                #   kp[7]  left scape    (antenna 1 joint  @ 30%)
                #   kp[8]  left flagellum (antenna 1 tip   @ 100%)
                #
                # Antenna 0 in BFS order maps to 'right', antenna 1 to 'left'.
                # When only one antenna is detected, mark the missing side as
                # a miss so the temporal tracker can gap-fill it.
                bfs_bases  = antenna_results['base_points']   # local coords
                bfs_joints = antenna_results['joint_points']
                bfs_tips   = antenna_results['tip_points']

                def _local_to_global_kp(pt):
                    if pt is None:
                        return None
                    return (pt[0] + ox, pt[1] + oy)

                # Feed detections into temporal tracker
                tracker = self.temporal_tracker
                for side_idx, side_name in ((0, 'right'), (1, 'left')):
                    if side_idx < len(bfs_joints) and side_idx < len(bfs_tips):
                        b_g = _local_to_global_kp(bfs_bases[side_idx]) if side_idx < len(bfs_bases) else None
                        j_g = _local_to_global_kp(bfs_joints[side_idx])
                        t_g = _local_to_global_kp(bfs_tips[side_idx])
                        tracker.update(bee_id, side_name, b_g, j_g, t_g)
                    else:
                        tracker.mark_miss(bee_id, side_name)

                # Read back smoothed positions (Layer 2 EMA + Layer 3 gap fill)
                if keypoints is not None:
                    try:
                        keypoints = list(keypoints)  # make mutable

                        r_base, r_joint, r_tip, r_conf = tracker.get_smoothed(bee_id, 'right')
                        l_base, l_joint, l_tip, l_conf = tracker.get_smoothed(bee_id, 'left')

                        if r_joint and len(keypoints) > 5:
                            keypoints[5] = (float(r_joint[0]), float(r_joint[1]),
                                            max(r_conf, 0.15))
                        if r_tip and len(keypoints) > 6:
                            keypoints[6] = (float(r_tip[0]), float(r_tip[1]),
                                            max(r_conf, 0.15))
                        if l_joint and len(keypoints) > 7:
                            keypoints[7] = (float(l_joint[0]), float(l_joint[1]),
                                            max(l_conf, 0.15))
                        if l_tip and len(keypoints) > 8:
                            keypoints[8] = (float(l_tip[0]), float(l_tip[1]),
                                            max(l_conf, 0.15))

                        all_bee_data[bee_id] = (keypoints, box)
                        print(f"[KP_WRITEBACK] bee {bee_id}: "
                              f"kp5={keypoints[5][:2]}, kp6={keypoints[6][:2]}, "
                              f"kp7={keypoints[7][:2]}, kp8={keypoints[8][:2]}  "
                              f"(r_conf={r_conf:.2f}, l_conf={l_conf:.2f})")
                    except Exception as _wb_err:
                        print(f"[KP_WRITEBACK] Warning: {_wb_err}")
                # ─────────────────────────────────────────────────────────────

                # Stats
                if 'stats' not in stages:
                    stages['stats'] = {}
                stages['stats']['antenna_endpoints_count'] = len(antenna_results['endpoint_list'])
                stages['stats']['antenna_path_length'] = len(antenna_results.get('bfs_path', []))

                # Process ALL bees (not just first)
                
        except Exception as e:
            print(f"Error generating antenna processing stages: {e}")
        
        return stages
    
    def _resize_to_canvas(self, img, canvas_size):
        """Resize image to fit canvas while maintaining aspect ratio"""
        if img is None or img.size == 0:
            return np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        
        h, w = img.shape[:2]
        
        # Calculate scaling factor to fit in canvas
        scale = min(canvas_size / h, canvas_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        if len(img.shape) == 2:
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # Center in canvas
            canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        else:
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        
        # Calculate position to center the image
        y_offset = (canvas_size - new_h) // 2
        x_offset = (canvas_size - new_w) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas


def create_metric_card(title, initial_value="0", description="", value_size=12):
    """
    Create a standardized metric card with BeeVision theming.
    
    Args:
        title: Card title text
        initial_value: Initial display value
        description: Optional subtitle text
        value_size: Font size for value (14=standard, 18=large)
    
    Returns:
        tuple: (card_frame, value_label)
    """
    card = QFrame()
    card.setFrameShape(QFrame.Shape.Box)
    card.setFrameShadow(QFrame.Shadow.Raised)
    card.setLineWidth(1)
    card.setStyleSheet(BeeVisionTheme.get_card_style())
    
    layout = QVBoxLayout(card)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(4)
    
    # Title
    title_label = QLabel(title)
    title_label.setFont(QFont(BeeVisionTheme.FONT_FAMILY, 9, QFont.Weight.Bold))
    title_label.setStyleSheet(f"color: {BeeVisionTheme.PRIMARY_BLUE}; border: none;")
    title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(title_label)
    
    # Value
    value_label = QLabel(initial_value)
    value_label.setFont(QFont(BeeVisionTheme.FONT_FAMILY, value_size, QFont.Weight.Bold))
    value_label.setStyleSheet(f"color: {BeeVisionTheme.TEXT_PRIMARY}; border: none;")
    value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(value_label)
    
    # Description (optional)
    if description:
        desc_label = QLabel(description)
        desc_label.setFont(QFont(BeeVisionTheme.FONT_FAMILY, 8))
        desc_label.setStyleSheet(f"color: {BeeVisionTheme.TEXT_SECONDARY}; border: none;")
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
    
    layout.addStretch()
    
    return card, value_label


def create_progress_card(title, categories):
    """
    Create a card with progress bars for distribution display.
    
    Args:
        title: Card title
        categories: List of tuples [(label, initial_value), ...]
    
    Returns:
        tuple: (card_frame, dict of {label: progress_bar})
    """
    card = QFrame()
    card.setFrameShape(QFrame.Shape.Box)
    card.setFrameShadow(QFrame.Shadow.Raised)
    card.setStyleSheet(BeeVisionTheme.get_card_style())
    
    layout = QVBoxLayout(card)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)
    
    # Title
    title_label = QLabel(title)
    title_label.setFont(QFont(BeeVisionTheme.FONT_FAMILY, 9, QFont.Weight.Bold))
    title_label.setStyleSheet(f"color: {BeeVisionTheme.PRIMARY_BLUE}; border: none;")
    layout.addWidget(title_label)
    
    # Progress bars
    progress_bars = {}
    for label, initial_value in categories:
        # Label + value container
        row_layout = QHBoxLayout()
        
        label_widget = QLabel(label)
        label_widget.setFont(QFont(BeeVisionTheme.FONT_FAMILY, 8))
        label_widget.setStyleSheet(f"color: {BeeVisionTheme.TEXT_SECONDARY}; border: none;")
        label_widget.setMinimumWidth(80)
        row_layout.addWidget(label_widget)
        
        progress = QProgressBar()
        progress.setStyleSheet(BeeVisionTheme.get_progress_bar_style())
        progress.setMaximum(100)
        progress.setValue(initial_value)
        progress.setFormat(f"{initial_value}%")
        progress.setTextVisible(True)
        row_layout.addWidget(progress, stretch=1)
        
        layout.addLayout(row_layout)
        progress_bars[label] = progress
    
    layout.addStretch()
    
    return card, progress_bars


def create_alert_card(severity="none"):
    """
    Create an anomaly alert card.
    
    Args:
        severity: "none", "moderate", or "severe"
    
    Returns:
        tuple: (card_frame, content_layout)
    """
    card = QFrame()
    card.setFrameShape(QFrame.Shape.Box)
    card.setFrameShadow(QFrame.Shadow.Raised)
    
    if severity == "severe":
        border_color = BeeVisionTheme.ERROR
        icon = "🔴"
    elif severity == "moderate":
        border_color = BeeVisionTheme.WARNING
        icon = "🟡"
    else:
        border_color = BeeVisionTheme.SUCCESS
        icon = "✅"
    
    card.setStyleSheet(f"""
        QFrame {{
            background-color: {BeeVisionTheme.CARD_BG};
            border: 2px solid {border_color};
            border-radius: 8px;
            padding: 12px;
        }}
    """)
    
    layout = QVBoxLayout(card)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)
    
    # Header
    header = QLabel(f"{icon} ANOMALY DETECTION")
    header.setFont(QFont(BeeVisionTheme.FONT_FAMILY, 9, QFont.Weight.Bold))
    header.setStyleSheet(f"color: {BeeVisionTheme.PRIMARY_BLUE}; border: none;")
    layout.addWidget(header)
    
    return card, layout


class ExpandableSection(QWidget):
    """
    Collapsible section widget with smooth animations.
    Compatible with BeeVision theming.
    """
    
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.is_expanded = False
        
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Toggle button (header)
        self.toggle_button = QPushButton(f"▶ {title}")
        self.toggle_button.setFont(QFont(BeeVisionTheme.FONT_FAMILY, 10, QFont.Weight.Bold))
        self.toggle_button.setStyleSheet(BeeVisionTheme.get_section_header_style(expanded=False))
        self.toggle_button.clicked.connect(self.toggle)
        self.toggle_button.setMinimumHeight(36)
        self.main_layout.addWidget(self.toggle_button)
        
        # Collapsed summary container
        self.summary_widget = QWidget()
        self.summary_layout = QGridLayout(self.summary_widget)
        self.summary_layout.setContentsMargins(0, 6, 0, 0)
        self.summary_layout.setSpacing(8)
        self.main_layout.addWidget(self.summary_widget)
        
        # Expanded content container
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 6, 0, 0)
        self.content_layout.setSpacing(8)
        self.content_widget.setVisible(False)
        self.main_layout.addWidget(self.content_widget)
        
        self.title = title
    
    def toggle(self):
        """Toggle expanded/collapsed state with animation"""
        self.is_expanded = not self.is_expanded
        
        if self.is_expanded:
            self.toggle_button.setText(f"▼ {self.title}")
            self.toggle_button.setStyleSheet(BeeVisionTheme.get_section_header_style(expanded=True))
            self.summary_widget.setVisible(False)
            self.content_widget.setVisible(True)
        else:
            self.toggle_button.setText(f"▶ {self.title}")
            self.toggle_button.setStyleSheet(BeeVisionTheme.get_section_header_style(expanded=False))
            self.summary_widget.setVisible(True)
            self.content_widget.setVisible(False)
    
    def add_summary_card(self, card, row, col, rowspan=1, colspan=1):
        """Add card to collapsed summary view"""
        self.summary_layout.addWidget(card, row, col, rowspan, colspan)
    
    def add_content_widget(self, widget):
        """Add widget to expanded content view"""
        self.content_layout.addWidget(widget)
    
    def get_content_layout(self):
        """Get the expanded content layout for adding widgets"""
        return self.content_layout


def create_antenna_dashboard_tab():
    """
    Create the complete Antenna Analysis Dashboard tab with BeeVision theming.
    
    Returns:
        tuple: (tab_widget, card_references)
    """
    tab = QWidget()
    tab.setStyleSheet(f"background-color: {BeeVisionTheme.DARK_BG};")
    
    # Main layout
    main_layout = QVBoxLayout(tab)
    main_layout.setContentsMargins(12, 12, 12, 12)
    main_layout.setSpacing(10)
    
    # Title
    title = QLabel("🐝 ANTENNA BEHAVIOR ANALYSIS")
    title.setFont(QFont(BeeVisionTheme.FONT_FAMILY, 12, QFont.Weight.Bold))
    title.setStyleSheet(f"color: {BeeVisionTheme.PRIMARY_BLUE}; background: transparent;")
    title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    main_layout.addWidget(title)
    
    # Scroll area for content
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setStyleSheet("""
        QScrollArea {
            background-color: transparent;
            border: none;
        }
        QScrollBar:vertical {
            background-color: #232323;
            width: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background-color: #444444;
            border-radius: 6px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #00b4d8;
        }
    """)
    
    content_widget = QWidget()
    content_widget.setStyleSheet("background: transparent;")
    content_layout = QVBoxLayout(content_widget)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(10)
    
    # Dictionary to store all card references for updates
    cards = {}
    
    # =========================
    # KEY METRICS (Always Visible)
    # =========================
    key_metrics_layout = QHBoxLayout()
    key_metrics_layout.setSpacing(8)
    
    card1, cards['contact_activity'] = create_metric_card(
        "CONTACT ACTIVITY", "0", "frames (last 10s)", value_size=16
    )
    key_metrics_layout.addWidget(card1)
    
    card2, cards['window_coverage'] = create_metric_card(
        "WINDOW COVERAGE", "0.0%", "300 frames", value_size=16
    )
    key_metrics_layout.addWidget(card2)
    
    card3, cards['antenna_balance'] = create_metric_card(
        "ANTENNA BALANCE", "0:1", "R/L ratio", value_size=16
    )
    key_metrics_layout.addWidget(card3)
    
    content_layout.addLayout(key_metrics_layout)
    
    # =========================
    # SECTION 1: Directional Contact Patterns
    # =========================
    section1 = ExpandableSection("SECTION 1: DIRECTIONAL CONTACT PATTERNS")
    
    # Summary cards (collapsed view)
    summary_card1, cards['cross_pattern_summary'] = create_metric_card(
        "CROSS-PATTERN", "0", "frames (RL+LR)", value_size=12
    )
    section1.add_summary_card(summary_card1, 0, 0)
    
    summary_card2, cards['same_side_summary'] = create_metric_card(
        "SAME-SIDE", "0", "frames (RR+LL)", value_size=12
    )
    section1.add_summary_card(summary_card2, 0, 1)
    
    # Expanded content
    expanded_grid = QGridLayout()
    expanded_grid.setSpacing(12)
    
    # Four directional patterns
    card_rr, cards['rr_frames'] = create_metric_card("RR (R→R)", "0", "frames", value_size=12)
    expanded_grid.addWidget(card_rr, 0, 0)
    
    card_rl, cards['rl_frames'] = create_metric_card("RL (R→L)", "0", "frames", value_size=12)
    expanded_grid.addWidget(card_rl, 0, 1)
    
    card_lr, cards['lr_frames'] = create_metric_card("LR (L→R)", "0", "frames", value_size=12)
    expanded_grid.addWidget(card_lr, 0, 2)
    
    card_ll, cards['ll_frames'] = create_metric_card("LL (L→L)", "0", "frames", value_size=12)
    expanded_grid.addWidget(card_ll, 0, 3)
    
    # Pattern analysis
    card_cross, cards['cross_pattern_detail'] = create_metric_card(
        "CROSS-PATTERN (RL+LR)", "0", "Directional Symmetry: 0:1", value_size=12
    )
    expanded_grid.addWidget(card_cross, 1, 0, 1, 2)
    
    card_same, cards['same_side_detail'] = create_metric_card(
        "SAME-SIDE (RR+LL)", "0", "Pattern Balance: 0:1", value_size=12
    )
    expanded_grid.addWidget(card_same, 1, 2, 1, 2)
    
    # Distribution chart
    dist_card, cards['pattern_distribution'] = create_progress_card(
        "📊 PATTERN DISTRIBUTION",
        [("RR", 0), ("RL", 0), ("LR", 0), ("LL", 0)]
    )
    expanded_grid.addWidget(dist_card, 2, 0, 1, 4)
    
    expanded_container = QWidget()
    expanded_container.setLayout(expanded_grid)
    section1.add_content_widget(expanded_container)
    
    content_layout.addWidget(section1)
    
    # =========================
    # SECTION 2: Antenna Symmetry & Lateralization
    # =========================
    section2 = ExpandableSection("SECTION 2: ANTENNA SYMMETRY & LATERALIZATION")
    
    # Summary cards
    summary2_card1, cards['lateralization_summary'] = create_metric_card(
        "LATERALIZATION INDEX", "0.00", "(-1.0 to +1.0)", value_size=12
    )
    section2.add_summary_card(summary2_card1, 0, 0)
    
    summary2_card2, cards['usage_balance_summary'] = create_metric_card(
        "USAGE BALANCE", "0:1", "Right favored", value_size=12
    )
    section2.add_summary_card(summary2_card2, 0, 1)
    
    # Expanded content
    expanded2_grid = QGridLayout()
    expanded2_grid.setSpacing(12)
    
    card_right, cards['right_usage'] = create_metric_card(
        "RIGHT ANTENNA USAGE", "0", "frames", value_size=12
    )
    expanded2_grid.addWidget(card_right, 0, 0)
    
    card_left, cards['left_usage'] = create_metric_card(
        "LEFT ANTENNA USAGE", "0", "frames", value_size=12
    )
    expanded2_grid.addWidget(card_left, 0, 1)
    
    card_ratio, cards['rl_ratio'] = create_metric_card(
        "R/L RATIO", "0:1", "Right favored", value_size=12
    )
    expanded2_grid.addWidget(card_ratio, 0, 2)
    
    # Lateralization detail
    card_lat, cards['lateralization_detail'] = create_metric_card(
        "LATERALIZATION INDEX", "0.00", "Interpretation: Balanced", value_size=12
    )
    expanded2_grid.addWidget(card_lat, 1, 0, 1, 3)
    
    # Directional symmetry
    card_sym, cards['directional_symmetry'] = create_metric_card(
        "DIRECTIONAL SYMMETRY", "0:1", "RL/LR Ratio", value_size=12
    )
    expanded2_grid.addWidget(card_sym, 2, 0, 1, 3)
    
    expanded2_container = QWidget()
    expanded2_container.setLayout(expanded2_grid)
    section2.add_content_widget(expanded2_container)
    
    content_layout.addWidget(section2)
    
    # =========================
    # SECTION 3: Regional Anatomy Distribution
    # =========================
    section3 = ExpandableSection("SECTION 3: REGIONAL ANATOMY DISTRIBUTION")
    
    # Summary cards
    summary3_card1, cards['most_contacted'] = create_metric_card(
        "MOST CONTACTED", "---", "0 frames", value_size=12
    )
    section3.add_summary_card(summary3_card1, 0, 0)
    
    summary3_card2, cards['least_contacted'] = create_metric_card(
        "LEAST CONTACTED", "---", "0 frames", value_size=12
    )
    section3.add_summary_card(summary3_card2, 0, 1)
    
    summary3_card3, cards['contact_rate'] = create_metric_card(
        "CONTACT RATE", "0.0%", "coverage", value_size=12
    )
    section3.add_summary_card(summary3_card3, 0, 2)
    
    # Expanded content
    expanded3_layout = QVBoxLayout()
    expanded3_layout.setSpacing(12)
    
    # Regional distribution
    region_card, cards['region_distribution'] = create_progress_card(
        "📊 CONTACT DISTRIBUTION BY BODY REGION",
        [
            ("REGION 1: PROTHORAX", 0),
            ("REGION 2: MESOTHORAX", 0),
            ("REGION 3: METATHORAX", 0),
            ("REGION 4: ABDOMEN", 0)
        ]
    )
    expanded3_layout.addWidget(region_card)
    
    # Regional hierarchy
    hierarchy_card, cards['regional_hierarchy'] = create_progress_card(
        "📊 REGIONAL HIERARCHY",
        [("Region 1", 0), ("Region 2", 0), ("Region 3", 0), ("Region 4", 0)]
    )
    expanded3_layout.addWidget(hierarchy_card)
    
    expanded3_container = QWidget()
    expanded3_container.setLayout(expanded3_layout)
    section3.add_content_widget(expanded3_container)
    
    content_layout.addWidget(section3)
    
    # =========================
    # ANOMALY ALERTS (Always Visible)
    # =========================
    anomaly_card, anomaly_layout = create_alert_card(severity="none")
    cards['anomaly_container'] = anomaly_layout
    
    # Default "no anomalies" message
    no_anomaly_label = QLabel("✅ No anomalies detected - All metrics within normal range")
    no_anomaly_label.setFont(QFont(BeeVisionTheme.FONT_FAMILY, 10))
    no_anomaly_label.setStyleSheet(f"color: {BeeVisionTheme.TEXT_PRIMARY}; border: none;")
    no_anomaly_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    anomaly_layout.addWidget(no_anomaly_label)
    cards['no_anomaly_label'] = no_anomaly_label
    
    content_layout.addWidget(anomaly_card)
    
    # Add stretch at bottom
    content_layout.addStretch()
    
    scroll.setWidget(content_widget)
    main_layout.addWidget(scroll)
    
    return tab, cards
# ============================================================================
# SHARED UTILITY CLASSES - Add these BEFORE the @jit decorator
# ============================================================================

class AntennaGeometry:
    """Unified antenna geometric operations"""
    
    @staticmethod
    def extract_antenna_keypoints(keypoints):
        """Extract antenna components from keypoint array
        
        Returns:
            tuple: (right_scape, right_flag, left_scape, left_flag) as np.array
        """
        return (
            np.array([keypoints[5][0], keypoints[5][1]]),  # right scape
            np.array([keypoints[6][0], keypoints[6][1]]),  # right flagellum
            np.array([keypoints[7][0], keypoints[7][1]]),  # left scape
            np.array([keypoints[8][0], keypoints[8][1]]),  # left flagellum
        )
    
    @staticmethod
    def normalize_vector(dx, dy):
        """Safe vector normalization
        
        Returns:
            tuple: (normalized_dx, normalized_dy)
        """
        length = np.sqrt(dx**2 + dy**2)
        if length < 1e-6:
            return 1.0, 0.0
        return dx / length, dy / length
    
    @staticmethod
    def get_safe_roi(cx, cy, radius, frame_shape):
        """Get ROI bounds with frame boundary checking
        
        Args:
            cx, cy: center coordinates
            radius: search radius
            frame_shape: (height, width)
        
        Returns:
            tuple: (y_min, y_max, x_min, x_max)
        """
        h, w = frame_shape
        return (
            max(0, int(cy - radius)),
            min(h, int(cy + radius + 1)),
            max(0, int(cx - radius)),
            min(w, int(cx + radius + 1))
        )


class MorphologyFactory:
    """Cached morphological operations for antenna detection"""
    
    # Cache for expensive kernel creation
    _kernel_cache = {}
    
    @classmethod
    def get_kernel(cls, kernel_type, size):
        """Get or create morphological kernel"""
        key = (kernel_type, size)
        if key not in cls._kernel_cache:
            if kernel_type == 'rect':
                cls._kernel_cache[key] = cv2.getStructuringElement(cv2.MORPH_RECT, size)
            elif kernel_type == 'cross':
                cls._kernel_cache[key] = cv2.getStructuringElement(cv2.MORPH_CROSS, size)
            elif kernel_type == 'ellipse':
                cls._kernel_cache[key] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
        return cls._kernel_cache[key]
    
    @staticmethod
    def compute_darkness_map(frame_gray, mode='thin', kernel_size=7):
        """Compute darkness map with tophat filtering
        
        Args:
            frame_gray: grayscale frame
            mode: 'thin', 'thick', or 'multi' for multi-directional
            kernel_size: Size of structuring element (default 7, use 3-5 for thin antennae)
        
        Returns:
            np.array: processed darkness map
        
        Note: Smaller kernel_size (3-5) better for thin worker bee antennae.
              Larger kernel_size (15-21) better for thick yellowjacket antennae.
        """
        darkest = 255 - frame_gray
        
        if mode == 'thin':
            # Use provided kernel_size for detection sensitivity
            kernel_v = MorphologyFactory.get_kernel('rect', (1, kernel_size))
            kernel_h = MorphologyFactory.get_kernel('rect', (kernel_size, 1))
            
            tophat_v = cv2.morphologyEx(darkest, cv2.MORPH_TOPHAT, kernel_v)
            tophat_h = cv2.morphologyEx(darkest, cv2.MORPH_TOPHAT, kernel_h)
            
            return cv2.max(tophat_v, tophat_h)
        
        elif mode == 'thick':
            # Scale up kernel for thick structures
            thick_size = max(15, kernel_size * 2)
            kernel_v = MorphologyFactory.get_kernel('rect', (1, thick_size))
            kernel_h = MorphologyFactory.get_kernel('rect', (thick_size, 1))
            
            tophat_v = cv2.morphologyEx(darkest, cv2.MORPH_TOPHAT, kernel_v)
            tophat_h = cv2.morphologyEx(darkest, cv2.MORPH_TOPHAT, kernel_h)
            
            return cv2.max(tophat_v, tophat_h)
        
        elif mode == 'multi':
            # Multi-directional: use kernel_size for all directions
            kernel_v = MorphologyFactory.get_kernel('rect', (1, kernel_size))
            kernel_h = MorphologyFactory.get_kernel('rect', (kernel_size, 1))
            
            # Diagonal kernel scales proportionally (70% of main kernel)
            diag_size = max(3, int(kernel_size * 0.7))
            if diag_size % 2 == 0:
                diag_size += 1
            kernel_d = MorphologyFactory.get_kernel('cross', (diag_size, diag_size))
            
            tophat_v = cv2.morphologyEx(darkest, cv2.MORPH_TOPHAT, kernel_v)
            tophat_h = cv2.morphologyEx(darkest, cv2.MORPH_TOPHAT, kernel_h)
            tophat_d = cv2.morphologyEx(darkest, cv2.MORPH_TOPHAT, kernel_d)
            
            return cv2.max(tophat_v, cv2.max(tophat_h, tophat_d))
        
        return darkest

class SkeletonProcessor:
    """Processes antenna skeletons for precise keypoint placement"""
    
    @staticmethod
    def morphological_skeleton(binary_image):
        """Extract 1-pixel-wide skeleton from binary image"""
        if XIMGPROC_AVAILABLE:
            # Use OpenCV's optimized thinning
            skeleton = cv2.ximgproc.thinning(binary_image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        else:
            # Fallback: iterative morphological thinning
            skeleton = binary_image.copy()
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            
            while True:
                eroded = cv2.erode(skeleton, kernel)
                temp = cv2.dilate(eroded, kernel)
                temp = cv2.subtract(skeleton, temp)
                skeleton = eroded.copy()
                
                if cv2.countNonZero(temp) == 0:
                    break
        
        return skeleton
    
    @staticmethod
    def find_skeleton_endpoints(skeleton):
        """Find endpoints (pixels with exactly 1 neighbor) in skeleton"""
        # Kernel to count neighbors
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        
        # Count neighbors for each pixel
        neighbor_count = cv2.filter2D(skeleton // 255, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        
        # Endpoints have exactly 1 neighbor
        endpoints_mask = (skeleton > 0) & (neighbor_count == 1)
        
        # Get coordinates
        endpoint_coords = np.column_stack(np.where(endpoints_mask))
        
        # Convert to (x, y) format
        endpoints = [(int(pt[1]), int(pt[0])) for pt in endpoint_coords]
        
        return endpoints
    
    @staticmethod
    def trace_skeleton_path(skeleton, start_point, end_point):
        """Trace ordered path along skeleton from start to end"""
        # Convert skeleton to graph-like structure
        skeleton_binary = (skeleton > 0).astype(np.uint8)
        
        # BFS to find path
        from collections import deque
        
        queue = deque([start_point])
        visited = {start_point}
        parent = {start_point: None}
        
        # 8-connectivity neighbors
        neighbors_offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        found = False
        while queue and not found:
            current = queue.popleft()
            
            if current == end_point:
                found = True
                break
            
            cx, cy = current
            
            for dx, dy in neighbors_offsets:
                nx, ny = cx + dx, cy + dy
                
                # Check bounds
                if 0 <= ny < skeleton_binary.shape[0] and 0 <= nx < skeleton_binary.shape[1]:
                    if skeleton_binary[ny, nx] > 0 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        parent[(nx, ny)] = current
                        queue.append((nx, ny))
        
        if not found:
            # No path found, return straight line interpolation
            return SkeletonProcessor.interpolate_line(start_point, end_point, 100)
        
        # Reconstruct path
        path = []
        current = end_point
        while current is not None:
            path.append(current)
            current = parent[current]
        
        path.reverse()
        return path
    
    @staticmethod
    def interpolate_line(start, end, num_points):
        """Create virtual straight line with num_points between start and end"""
        x_vals = np.linspace(start[0], end[0], num_points)
        y_vals = np.linspace(start[1], end[1], num_points)
        
        path = [(int(x), int(y)) for x, y in zip(x_vals, y_vals)]
        return path
class AntennaCandidate:
    """Stores antenna candidate structure with motion tracking"""
    
    def __init__(self, structure_id, position, blob):
        self.id = structure_id
        self.positions = [position]  # History of centroid positions
        self.blobs = [blob]  # History of blob structures
        self.motion_score = 0.0
        self.is_antenna = False
        self.side = None  # 'left' or 'right'
    
    def update(self, position, blob):
        """Update with new frame data"""
        if len(self.positions) >= 2:
            displacement = np.sqrt(
                (position[0] - self.positions[-1][0])**2 + 
                (position[1] - self.positions[-1][1])**2
            )
            self.motion_score += displacement
        
        self.positions.append(position)
        self.blobs.append(blob)
    
    def get_average_motion(self):
        """Calculate average motion per frame"""
        if len(self.positions) < 2:
            return 0.0
        return self.motion_score / (len(self.positions) - 1)
    
    def get_latest_position(self):
        """Get most recent position"""
        return self.positions[-1] if self.positions else None
    
    def get_latest_blob(self):
        """Get most recent blob structure"""
        return self.blobs[-1] if self.blobs else None


class MotionBasedAntennaDetector:
    """Detects antennae by tracking motion over multiple frames"""
    
    def __init__(self, observation_frames=6, motion_threshold=0.6):
        self.observation_frames = observation_frames
        self.motion_threshold = motion_threshold
        self.candidates = {}  # {bee_id: [AntennaCandidate]}
        self.frame_count = {}  # {bee_id: current_frame_in_observation}
        self.detection_complete = {}  # {bee_id: bool}
    
    def is_detecting(self, bee_id):
        """Check if still in observation phase for this bee"""
        return bee_id in self.frame_count and self.frame_count[bee_id] < self.observation_frames
    
    def add_observation(self, bee_id, detected_structures):
        """Add frame observation for a bee
        
        Args:
            bee_id: ID of the bee
            detected_structures: list of dicts with keys: 'position', 'blob', 'score'
        """
        if bee_id not in self.candidates:
            self.candidates[bee_id] = []
            self.frame_count[bee_id] = 0
        
        current_frame = self.frame_count[bee_id]
        
        if current_frame == 0:
            # First frame: create candidates
            for i, struct in enumerate(detected_structures):
                candidate = AntennaCandidate(
                    structure_id=i,
                    position=struct['position'],
                    blob=struct['blob']
                )
                self.candidates[bee_id].append(candidate)
        else:
            # Match new structures to existing candidates
            existing_candidates = self.candidates[bee_id]
            
            for struct in detected_structures:
                # Find closest existing candidate
                min_dist = float('inf')
                closest_candidate = None
                
                for candidate in existing_candidates:
                    last_pos = candidate.get_latest_position()
                    dist = np.sqrt(
                        (struct['position'][0] - last_pos[0])**2 + 
                        (struct['position'][1] - last_pos[1])**2
                    )
                    
                    if dist < min_dist and dist < 30:  # Max 30px movement
                        min_dist = dist
                        closest_candidate = candidate
                
                # Update matched candidate
                if closest_candidate is not None:
                    closest_candidate.update(struct['position'], struct['blob'])
        
        self.frame_count[bee_id] += 1
    
    def select_antennae(self, bee_id, head_position):
        """Select the two antennae based on motion after observation period
        
        Returns:
            tuple: (left_antenna_candidate, right_antenna_candidate) or (None, None)
        """
        if bee_id not in self.candidates:
            return None, None
        
        if self.frame_count[bee_id] < self.observation_frames:
            # Still observing
            return None, None
        
        candidates = self.candidates[bee_id]
        
        # Calculate average motion for each
        for candidate in candidates:
            candidate.avg_motion = candidate.get_average_motion()
        
        # Filter by motion threshold
        moving_candidates = [c for c in candidates if c.avg_motion > self.motion_threshold]
        
        if len(moving_candidates) < 2:
            # Not enough moving structures
            return None, None
        
        # Sort by motion (highest first)
        moving_candidates.sort(key=lambda c: c.avg_motion, reverse=True)
        
        # Take top 2
        antenna1 = moving_candidates[0]
        antenna2 = moving_candidates[1]
        
        # Determine left/right based on x-coordinate
        pos1 = antenna1.get_latest_position()
        pos2 = antenna2.get_latest_position()
        
        if pos1[0] < pos2[0]:
            left_antenna = antenna1
            right_antenna = antenna2
        else:
            left_antenna = antenna2
            right_antenna = antenna1
        
        left_antenna.side = 'left'
        right_antenna.side = 'right'
        left_antenna.is_antenna = True
        right_antenna.is_antenna = True
        
        self.detection_complete[bee_id] = True
        
        return left_antenna, right_antenna
    def select_antennae_with_roi(self, bee_id, head_position, left_triangle, right_triangle):
        """Select antennae based on motion AND ROI triangle membership
        
        Args:
            bee_id: ID of the bee
            head_position: (x, y) of head
            left_triangle: [(x1,y1), (x2,y2), (x3,y3)] left ROI
            right_triangle: [(x1,y1), (x2,y2), (x3,y3)] right ROI
        
        Returns:
            tuple: (left_antenna_candidate, right_antenna_candidate) or (None, None)
        """
        if bee_id not in self.candidates:
            return None, None
        
        if self.frame_count[bee_id] < self.observation_frames:
            return None, None
        
        candidates = self.candidates[bee_id]
        
        # Calculate average motion for each
        for candidate in candidates:
            candidate.avg_motion = candidate.get_average_motion()
        
        # Filter by motion threshold
        moving_candidates = [c for c in candidates if c.avg_motion > self.motion_threshold]
        
        if len(moving_candidates) < 2:
            return None, None
        
        # Separate candidates by ROI
        left_candidates = []
        right_candidates = []
        
        for candidate in moving_candidates:
            pos = candidate.get_latest_position()
            
            # Check which triangle this candidate belongs to
            in_left = self._point_in_triangle(pos, left_triangle)
            in_right = self._point_in_triangle(pos, right_triangle)
            
            if in_left:
                left_candidates.append(candidate)
            elif in_right:
                right_candidates.append(candidate)
        
        # Select best from each side
        left_antenna = None
        right_antenna = None
        
        if left_candidates:
            # Sort by motion (highest first)
            left_candidates.sort(key=lambda c: c.avg_motion, reverse=True)
            left_antenna = left_candidates[0]
            left_antenna.side = 'left'
            left_antenna.is_antenna = True
        
        if right_candidates:
            # Sort by motion (highest first)
            right_candidates.sort(key=lambda c: c.avg_motion, reverse=True)
            right_antenna = right_candidates[0]
            right_antenna.side = 'right'
            right_antenna.is_antenna = True
        
        if left_antenna is not None and right_antenna is not None:
            self.detection_complete[bee_id] = True
        
        return left_antenna, right_antenna
    
    @staticmethod
    def _point_in_triangle(point, triangle):
        """Check if point is inside triangle"""
        px, py = point
        p1, p2, p3 = triangle
        
        def sign(px, py, ax, ay, bx, by):
            return (px - bx) * (ay - by) - (ax - bx) * (py - by)
        
        d1 = sign(px, py, p1[0], p1[1], p2[0], p2[1])
        d2 = sign(px, py, p2[0], p2[1], p3[0], p3[1])
        d3 = sign(px, py, p3[0], p3[1], p1[0], p1[1])
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        return not (has_neg and has_pos)    
    def reset_bee(self, bee_id):
        """Reset detection for a bee (for re-detection)"""
        if bee_id in self.candidates:
            del self.candidates[bee_id]
        if bee_id in self.frame_count:
            del self.frame_count[bee_id]
        if bee_id in self.detection_complete:
            del self.detection_complete[bee_id]
class OpticalFlowAntennaTracker:
    """Tracks antenna keypoints using sparse optical flow"""
    
    def __init__(self):
        self.tracked_points = {}  # {bee_id: {'left_joint': (x,y), 'left_tip': (x,y), ...}}
        self.prev_frame = None
        self.confidence = {}  # {bee_id: float}
        self.frames_tracked = {}  # {bee_id: int}
        self.last_valid = {}  # {bee_id: dict} — last successfully validated positions
        self.consecutive_failures = {}  # {bee_id: int} — frames since last valid track

        # Optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
    
    def initialize_tracking(self, bee_id, left_joint, left_tip, right_joint, right_tip, frame_gray):
        """Initialize optical flow tracking for a bee"""
        self.tracked_points[bee_id] = {
            'left_joint': left_joint,
            'left_tip': left_tip,
            'right_joint': right_joint,
            'right_tip': right_tip
        }
        self.confidence[bee_id] = 1.0
        self.frames_tracked[bee_id] = 0
        self.last_valid[bee_id] = {
            'left_joint': left_joint,
            'left_tip': left_tip,
            'right_joint': right_joint,
            'right_tip': right_tip,
            'confidence': 1.0
        }
        self.consecutive_failures[bee_id] = 0
        self.prev_frame = frame_gray.copy()
    
    def track_frame(self, bee_id, current_frame_gray, head_position):
        """Track antenna keypoints in current frame

        Returns:
            dict: {'left_joint': (x,y), 'left_tip': (x,y), 'right_joint': (x,y), 'right_tip': (x,y), 'confidence': float}
            or None if tracking has failed for too many consecutive frames (grace period exhausted)
        """
        # Grace period: hold last valid positions for up to GRACE_FRAMES before giving up
        GRACE_FRAMES = 12  # ~0.4 s @ 30 fps — dots stay visible during brief re-detection gaps

        if bee_id not in self.tracked_points or self.prev_frame is None:
            return None

        # Prepare points for tracking
        pts = self.tracked_points[bee_id]
        old_points = np.array([
            [pts['left_joint'][0], pts['left_joint'][1]],
            [pts['left_tip'][0], pts['left_tip'][1]],
            [pts['right_joint'][0], pts['right_joint'][1]],
            [pts['right_tip'][0], pts['right_tip'][1]]
        ], dtype=np.float32).reshape(-1, 1, 2)

        # Calculate optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_frame,
            current_frame_gray,
            old_points,
            None,
            **self.lk_params
        )

        tracking_valid = False

        if new_points is not None:
            # Extract new positions
            new_left_joint  = (float(new_points[0][0][0]), float(new_points[0][0][1]))
            new_left_tip    = (float(new_points[1][0][0]), float(new_points[1][0][1]))
            new_right_joint = (float(new_points[2][0][0]), float(new_points[2][0][1]))
            new_right_tip   = (float(new_points[3][0][0]), float(new_points[3][0][1]))

            # Validate tracking
            tracking_valid = self._validate_tracking(
                current_frame_gray,
                head_position,
                new_left_joint, new_left_tip,
                new_right_joint, new_right_tip,
                pts,
                status
            )

            if tracking_valid:
                # Successful frame — update state
                self.tracked_points[bee_id] = {
                    'left_joint':  new_left_joint,
                    'left_tip':    new_left_tip,
                    'right_joint': new_right_joint,
                    'right_tip':   new_right_tip
                }
                self.confidence[bee_id] = min(1.0, self.confidence[bee_id] + 0.02)
                self.frames_tracked[bee_id] += 1
                self.consecutive_failures[bee_id] = 0

                result = {
                    'left_joint':  new_left_joint,
                    'left_tip':    new_left_tip,
                    'right_joint': new_right_joint,
                    'right_tip':   new_right_tip,
                    'confidence':  self.confidence[bee_id]
                }
                self.last_valid[bee_id] = result.copy()
                return result

        # Tracking failed this frame — use grace-period fallback
        fail_count = self.consecutive_failures.get(bee_id, 0) + 1
        self.consecutive_failures[bee_id] = fail_count

        # Slow confidence decay during grace period (0.08 per frame instead of 0.3)
        self.confidence[bee_id] = max(0.0, self.confidence.get(bee_id, 1.0) - 0.08)

        if fail_count <= GRACE_FRAMES and bee_id in self.last_valid:
            # Return last known positions so dots remain visible
            held = dict(self.last_valid[bee_id])
            held['confidence'] = self.confidence[bee_id]
            return held

        # Grace period exhausted — signal caller to trigger re-detection
        return None
    
    def _validate_tracking(self, frame, head_pos, left_joint, left_tip, right_joint, right_tip, old_pts, status):
        """Validate tracked positions"""
        h, w = frame.shape

        # Check optical flow status — allow up to 1 failure out of 4 (was: ALL must pass)
        bad_status = int(np.sum(status.flatten() == 0))
        if bad_status >= 2:
            return False

        # Check bounds
        all_points = [left_joint, left_tip, right_joint, right_tip]
        for pt in all_points:
            if not (0 <= pt[0] < w and 0 <= pt[1] < h):
                return False

        # Check darkness — relaxed from 30 → 15 so moving antennae on lighter
        # background sections don't instantly fail
        for pt in all_points:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= y < h and 0 <= x < w:
                darkness = 255 - frame[y, x]
                if darkness < 15:
                    return False

        # Check geometry (distances from head) — widened bounds to handle scale/angle variation
        dist_left_joint  = np.sqrt((left_joint[0]  - head_pos[0])**2 + (left_joint[1]  - head_pos[1])**2)
        dist_left_tip    = np.sqrt((left_tip[0]    - head_pos[0])**2 + (left_tip[1]    - head_pos[1])**2)
        dist_right_joint = np.sqrt((right_joint[0] - head_pos[0])**2 + (right_joint[1] - head_pos[1])**2)
        dist_right_tip   = np.sqrt((right_tip[0]   - head_pos[0])**2 + (right_tip[1]   - head_pos[1])**2)

        # Joint: 5–80 px from head (was 10–60), tip: 20–150 px (was 40–120)
        if not (5 < dist_left_joint < 80 and 20 < dist_left_tip < 150):
            return False
        if not (5 < dist_right_joint < 80 and 20 < dist_right_tip < 150):
            return False

        # Joint should be closer than tip
        if dist_left_joint >= dist_left_tip or dist_right_joint >= dist_right_tip:
            return False

        # Check movement — unchanged (already relaxed)
        old_left_joint = old_pts['left_joint']
        old_left_tip   = old_pts['left_tip']

        movement_left_joint  = np.sqrt((left_joint[0]  - old_left_joint[0])**2 + (left_joint[1]  - old_left_joint[1])**2)
        movement_left_tip    = np.sqrt((left_tip[0]    - old_left_tip[0])**2    + (left_tip[1]    - old_left_tip[1])**2)
        movement_right_joint = np.sqrt((right_joint[0] - old_pts['right_joint'][0])**2 + (right_joint[1] - old_pts['right_joint'][1])**2)
        movement_right_tip   = np.sqrt((right_tip[0]   - old_pts['right_tip'][0])**2   + (right_tip[1]   - old_pts['right_tip'][1])**2)

        # Allow up to 50 px joint, 80 px tip movement
        if movement_left_joint > 50 or movement_left_tip > 80:
            return False
        if movement_right_joint > 50 or movement_right_tip > 80:
            return False

        return True
    
    def update_prev_frame(self, frame_gray):
        """Update previous frame for next iteration"""
        self.prev_frame = frame_gray.copy()
    
    def should_redetect(self, bee_id):
        """Check if re-detection is needed"""
        if bee_id not in self.confidence:
            return True
        
        # Re-detect if confidence drops too low
        if self.confidence[bee_id] < 0.4:
            return True
        
        # Periodic re-detection every 4 seconds (120 frames @ 30fps)
        if bee_id in self.frames_tracked and self.frames_tracked[bee_id] > 120:
            return True
        
        # NEW: Force re-detection if tracking same position too long (antenna frozen)
        # This detects when optical flow gets "stuck"
        if bee_id in self.tracked_points:
            pts = self.tracked_points[bee_id]
            
            # Check if antenna tips haven't moved in last 30 frames
            # (This would be unnatural - antennae are always moving)
            if bee_id in self.frames_tracked and self.frames_tracked[bee_id] > 30:
                # Trigger re-detection to "unfreeze" tracking
                # This is a safety mechanism
                pass  # Could add position history check here
        
        return False
    
    def reset_bee(self, bee_id):
        """Reset tracking for a bee"""
        if bee_id in self.tracked_points:
            del self.tracked_points[bee_id]
        self.last_valid.pop(bee_id, None)
        self.consecutive_failures.pop(bee_id, None)
        if bee_id in self.confidence:
            del self.confidence[bee_id]
        if bee_id in self.frames_tracked:
            del self.frames_tracked[bee_id]
class AntennaContactAnalyzer:
    """Unified antenna contact detection"""
    
    def __init__(self, threshold=10):
        self.threshold = threshold
    
    def check_antenna_intersection(self, kp_a, kp_b):
        """Check antenna contacts between two bees
        
        Returns:
            tuple: (contacts_dict, distances_dict)
                where contacts_dict has keys 'RR', 'RL', 'LR', 'LL'
        """
        # Extract antenna positions
        right_scape_a, right_flag_a, left_scape_a, left_flag_a = AntennaGeometry.extract_antenna_keypoints(kp_a)
        right_scape_b, right_flag_b, left_scape_b, left_flag_b = AntennaGeometry.extract_antenna_keypoints(kp_b)
        
        contacts = {'RR': False, 'RL': False, 'LR': False, 'LL': False}
        distances = {}
        
        # RIGHT-RIGHT
        dist_rr = min(
            np.linalg.norm(right_flag_a - right_scape_b),
            np.linalg.norm(right_flag_a - right_flag_b),
            np.linalg.norm(right_scape_a - right_flag_b)
        )
        distances['RR'] = dist_rr
        contacts['RR'] = dist_rr < self.threshold
        
        # RIGHT-LEFT
        dist_rl = min(
            np.linalg.norm(right_flag_a - left_scape_b),
            np.linalg.norm(right_flag_a - left_flag_b),
            np.linalg.norm(right_scape_a - left_flag_b)
        )
        distances['RL'] = dist_rl
        contacts['RL'] = dist_rl < self.threshold
        
        # LEFT-RIGHT
        dist_lr = min(
            np.linalg.norm(left_flag_a - right_scape_b),
            np.linalg.norm(left_flag_a - right_flag_b),
            np.linalg.norm(left_scape_a - right_flag_b)
        )
        distances['LR'] = dist_lr
        contacts['LR'] = dist_lr < self.threshold
        
        # LEFT-LEFT
        dist_ll = min(
            np.linalg.norm(left_flag_a - left_scape_b),
            np.linalg.norm(left_flag_a - left_flag_b),
            np.linalg.norm(left_scape_a - left_flag_b)
        )
        distances['LL'] = dist_ll
        contacts['LL'] = dist_ll < self.threshold
        
        return contacts, distances

# ============================================================================
# END SHARED UTILITIES
# ============================================================================
@jit(nopython=True, parallel=True)
def points_in_triangle_numba(points_x, points_y, p1x, p1y, p2x, p2y, p3x, p3y):
    """Vectorized point-in-triangle test using Numba JIT compilation."""
    n = len(points_x)
    result = np.zeros(n, dtype=np.bool_)
    
    for i in prange(n):
        px, py = points_x[i], points_y[i]
        d1 = (px - p2x) * (p1y - p2y) - (p1x - p2x) * (py - p2y)
        d2 = (px - p3x) * (p2y - p3y) - (p2x - p3x) * (py - p3y)
        d3 = (px - p1x) * (p3y - p1y) - (p3x - p1x) * (py - p3y)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        result[i] = not (has_neg and has_pos)
    
    return result

DARKEST_SEARCH_RADIUS = 80
ANTENNA_CONF_THRESHOLD = 0.15
ANTENNA_SEARCH_RADIUS = 40
ANTENNA_OVERRIDE_CONFIDENCE = 0.85
LEFT_ANGLE_OFFSET = 90.0
RIGHT_ANGLE_OFFSET = -90.0
# ADD THESE NEW LINES:
HEAD_DISTANCE_MIN = 0      # pixels
HEAD_DISTANCE_MAX = 150     # pixels
@dataclass
class AntennaContactMetrics:
    """Stores antenna contact classification for a bee pair"""
    rr_count: int = 0  # Right-Right contacts
    rl_count: int = 0  # Right-Left contacts
    lr_count: int = 0  # Left-Right contacts
    ll_count: int = 0  # Left-Left contacts
    
    @property
    def total_contacts(self):
        return self.rr_count + self.rl_count + self.lr_count + self.ll_count
    
    @property
    def rr_ratio(self):
        total = self.total_contacts
        return self.rr_count / total if total > 0 else 0.0
    
    @property
    def ll_ratio(self):
        total = self.total_contacts
        return self.ll_count / total if total > 0 else 0.0
    
    @property
    def cross_ratio(self):
        total = self.total_contacts
        cross = self.rl_count + self.lr_count
        return cross / total if total > 0 else 0.0
    
    @property
    def lateralization_index(self):
        """LI = (RR - LL) / Total. Range: -1.0 (left) to +1.0 (right)"""
        total = self.total_contacts
        if total == 0:
            return 0.0
        return (self.rr_count - self.ll_count) / total
    
    @property
    def symmetry_ratio(self):
        """RL / LR ratio. Ideal near 1.0, indicates injury if extreme"""
        if self.lr_count == 0:
            return self.rl_count if self.rl_count > 0 else 1.0
        return self.rl_count / self.lr_count
@dataclass
class RegionContactData:
    """Data for a single anatomical region"""
    right_contacts: int = 0
    left_contacts: int = 0
    right_distances: list = None
    left_distances: list = None
    
    def __post_init__(self):
        if self.right_distances is None:
            self.right_distances = []
        if self.left_distances is None:
            self.left_distances = []
    
    @property
    def total_contacts(self):
        return self.right_contacts + self.left_contacts
    
    @property
    def right_ratio(self):
        total = self.total_contacts
        return self.right_contacts / total if total > 0 else 0.0
    
    @property
    def avg_right_distance(self):
        return np.mean(self.right_distances) if self.right_distances else 999.0
    
    @property
    def avg_left_distance(self):
        return np.mean(self.left_distances) if self.left_distances else 999.0


class FourRegionAntennaTracker:
    """Track antenna contacts across 4 anatomical body regions"""
    
    REGIONS = {
        'PROTHORAX': 'Head → Line 1 (Neural Control)',
        'MESOTHORAX': 'Line 1 → Line 2 (Flight/Fitness)',
        'METATHORAX': 'Line 2 → Line 3 (Foraging/Pollen)',
        'ABDOMEN': 'Line 3 → Tip (Nutrition/Trophallaxis)'
    }
    
    def __init__(self):
        # Frame-by-frame tracking
        self.frame_data = {}  # {(bee_a, bee_b): {region: RegionContactData}}
        
        # Historical tracking (10-second windows)
        self.history = defaultdict(lambda: {
            'PROTHORAX': [],
            'MESOTHORAX': [],
            'METATHORAX': [],
            'ABDOMEN': []
        })
        
        # Time-series data (for graphs)
        self.timeseries = {
            'timestamps': [],
            'PROTHORAX': [],
            'MESOTHORAX': [],
            'METATHORAX': [],
            'ABDOMEN': [],
            'right_ratio': []
        }
        
        # Baseline for anomaly detection
        self.baseline = None
        self.baseline_frames = 9000  # First 5 minutes @ 30fps
        self.baseline_data = {
            'PROTHORAX': [],
            'MESOTHORAX': [],
            'METATHORAX': [],
            'ABDOMEN': []
        }
        
        # Settings
        self.contact_threshold = 15  # pixels
        self.window_frames = 300  # 10 seconds @ 30fps
        self.anomaly_threshold = 1.5  # 150% change
        
        self.frame_counter = 0
        self.start_time = time.time()
    
    def track_contacts(self, all_bee_data, frame_number):
        """Main tracking function - call every frame"""
        self.frame_data = {}
        self.frame_counter = frame_number
        
        bee_ids = list(all_bee_data.keys())
        
        for i, bee_a_id in enumerate(bee_ids):
            for bee_b_id in bee_ids[i+1:]:
                kp_a, box_a = all_bee_data[bee_a_id]
                kp_b, box_b = all_bee_data[bee_b_id]
                
                if kp_a is None or kp_b is None:
                    continue
                
                event_key = tuple(sorted([bee_a_id, bee_b_id]))
                
                # Calculate contacts for all 4 regions
                region_data = self._calculate_region_contacts(kp_a, kp_b)
                
                self.frame_data[event_key] = region_data
                
                # Store in history
                for region, data in region_data.items():
                    self.history[event_key][region].append({
                        'frame': frame_number,
                        'right_contact': data.right_contacts > 0,
                        'left_contact': data.left_contacts > 0,
                        'right_distance': data.avg_right_distance,
                        'left_distance': data.avg_left_distance
                    })
                    
                    # Keep history bounded
                    if len(self.history[event_key][region]) > self.window_frames:
                        self.history[event_key][region].pop(0)
        
        # Update timeseries every 10 frames (reduce data size)
        if frame_number % 10 == 0:
            self._update_timeseries()
        
        # Build baseline
        if frame_number < self.baseline_frames:
            self._accumulate_baseline()
        elif frame_number == self.baseline_frames:
            self._finalize_baseline()
    
    def _calculate_region_contacts(self, kp_a, kp_b):
        """Calculate antenna contacts for each body region"""
        regions = {
            'PROTHORAX': RegionContactData(),
            'MESOTHORAX': RegionContactData(),
            'METATHORAX': RegionContactData(),
            'ABDOMEN': RegionContactData()
        }
        
        # Get antenna positions
        right_antenna_a = np.array([kp_a[5][0], kp_a[5][1]])  # Right flagellum
        left_antenna_a = np.array([kp_a[7][0], kp_a[7][1]])   # Left flagellum
        
        # Define bee B's body regions (using keypoints 0,1,2,3,4)
        bee_b_regions = {
            'PROTHORAX': self._get_region_points(kp_b, 0, 1),
            'MESOTHORAX': self._get_region_points(kp_b, 1, 2),
            'METATHORAX': self._get_region_points(kp_b, 2, 3),
            'ABDOMEN': self._get_region_points(kp_b, 3, 4)
        }
        
        # Check each region
        for region_name, region_points in bee_b_regions.items():
            if region_points is None:
                continue
            
            # Calculate distances to region
            right_dist = self._distance_to_segment(right_antenna_a, region_points[0], region_points[1])
            left_dist = self._distance_to_segment(left_antenna_a, region_points[0], region_points[1])
            
            # Store distances
            regions[region_name].right_distances.append(right_dist)
            regions[region_name].left_distances.append(left_dist)
            
            # Check contact
            if right_dist < self.contact_threshold:
                regions[region_name].right_contacts = 1
            
            if left_dist < self.contact_threshold:
                regions[region_name].left_contacts = 1
        
        return regions
    
    def _get_region_points(self, keypoints, start_idx, end_idx):
        """Get two points defining a body region"""
        if start_idx >= len(keypoints) or end_idx >= len(keypoints):
            return None
        
        start_pt = np.array([keypoints[start_idx][0], keypoints[start_idx][1]])
        end_pt = np.array([keypoints[end_idx][0], keypoints[end_idx][1]])
        
        return (start_pt, end_pt)
    
    def _distance_to_segment(self, point, seg_start, seg_end):
        """Calculate minimum distance from point to line segment"""
        # Vector from seg_start to seg_end
        line_vec = seg_end - seg_start
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1e-6:
            return np.linalg.norm(point - seg_start)
        
        # Normalize
        line_unit = line_vec / line_len
        
        # Project point onto line
        point_vec = point - seg_start
        proj_length = np.dot(point_vec, line_unit)
        
        # Clamp to segment
        proj_length = max(0, min(line_len, proj_length))
        
        # Find closest point on segment
        closest = seg_start + proj_length * line_unit
        
        # Return distance
        return np.linalg.norm(point - closest)
    
    def _update_timeseries(self):
        """Aggregate current frame data into timeseries"""
        elapsed = time.time() - self.start_time
        
        # Count contacts across all pairs
        counts = {
            'PROTHORAX': 0,
            'MESOTHORAX': 0,
            'METATHORAX': 0,
            'ABDOMEN': 0
        }
        
        total_right = 0
        total_left = 0
        
        for event_key, region_data in self.frame_data.items():
            for region, data in region_data.items():
                counts[region] += data.total_contacts
                total_right += data.right_contacts
                total_left += data.left_contacts
        
        right_ratio = total_right / (total_right + total_left) if (total_right + total_left) > 0 else 0.0
        
        self.timeseries['timestamps'].append(elapsed)
        self.timeseries['PROTHORAX'].append(counts['PROTHORAX'])
        self.timeseries['MESOTHORAX'].append(counts['MESOTHORAX'])
        self.timeseries['METATHORAX'].append(counts['METATHORAX'])
        self.timeseries['ABDOMEN'].append(counts['ABDOMEN'])
        self.timeseries['right_ratio'].append(right_ratio)
        
        # Keep only last 1000 points (reduce memory)
        if len(self.timeseries['timestamps']) > 1000:
            for key in self.timeseries.keys():
                self.timeseries[key] = self.timeseries[key][-1000:]
    
    def _accumulate_baseline(self):
        """Accumulate data for baseline calculation"""
        for event_key, region_data in self.frame_data.items():
            for region, data in region_data.items():
                if data.total_contacts > 0:
                    self.baseline_data[region].append(data.total_contacts)
    
    def _finalize_baseline(self):
        """Calculate baseline statistics"""
        self.baseline = {}
        for region, data in self.baseline_data.items():
            if len(data) > 0:
                self.baseline[region] = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'median': np.median(data)
                }
            else:
                self.baseline[region] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'median': 0.0
                }
        
        print(f"[4-REGION] Baseline established: {self.baseline}")
    
    def get_current_window_stats(self):
        """Get statistics for current 10-second window"""
        stats = {
            'PROTHORAX': {'right': 0, 'left': 0, 'total': 0},
            'MESOTHORAX': {'right': 0, 'left': 0, 'total': 0},
            'METATHORAX': {'right': 0, 'left': 0, 'total': 0},
            'ABDOMEN': {'right': 0, 'left': 0, 'total': 0}
        }
        
        for event_key, history_dict in self.history.items():
            for region, frames in history_dict.items():
                for frame_data in frames:
                    if frame_data['right_contact']:
                        stats[region]['right'] += 1
                    if frame_data['left_contact']:
                        stats[region]['left'] += 1
                    
                    stats[region]['total'] = stats[region]['right'] + stats[region]['left']
        
        return stats
    
    def detect_anomalies(self):
        """Detect anomalies vs baseline"""
        if self.baseline is None:
            return []
        
        current = self.get_current_window_stats()
        anomalies = []
        
        for region in self.REGIONS.keys():
            baseline_mean = self.baseline[region]['mean']
            current_count = current[region]['total']
            
            if baseline_mean == 0:
                continue
            
            change_ratio = current_count / (baseline_mean + 1e-6)
            
            if change_ratio > self.anomaly_threshold or change_ratio < (1.0 / self.anomaly_threshold):
                anomalies.append({
                    'region': region,
                    'baseline': baseline_mean,
                    'current': current_count,
                    'change_pct': (change_ratio - 1.0) * 100,
                    'severity': '🔴' if abs(change_ratio - 1.0) > 1.0 else '🟡'
                })
        
        return anomalies
    
    def get_region_health_assessment(self):
        """Interpret what each region's activity means"""
        stats = self.get_current_window_stats()
        
        assessments = {}
        
        # PROTHORAX (should be low)
        proto_total = stats['PROTHORAX']['total']
        if proto_total < 15:  # <5% of 300 frames
            assessments['PROTHORAX'] = {
                'status': '🟢 NORMAL',
                'meaning': 'No head/neural concerns detected'
            }
        else:
            assessments['PROTHORAX'] = {
                'status': '🟡 ELEVATED',
                'meaning': 'Unusual head region activity - check for parasites'
            }
        
        # MESOTHORAX (should be high)
        meso_total = stats['MESOTHORAX']['total']
        if meso_total > 100:  # >33% of 300 frames
            assessments['MESOTHORAX'] = {
                'status': '🟢 EXCELLENT',
                'meaning': 'Intense fitness evaluation - healthy vigilance'
            }
        elif meso_total > 60:
            assessments['MESOTHORAX'] = {
                'status': '🟢 GOOD',
                'meaning': 'Normal fitness assessment activity'
            }
        else:
            assessments['MESOTHORAX'] = {
                'status': '🟡 LOW',
                'meaning': 'Reduced fitness checks - possible lethargy'
            }
        
        # METATHORAX (moderate)
        meta_total = stats['METATHORAX']['total']
        if meta_total > 50:
            assessments['METATHORAX'] = {
                'status': '🟢 ACTIVE',
                'meaning': 'Good foraging coordination'
            }
        else:
            assessments['METATHORAX'] = {
                'status': '🟡 QUIET',
                'meaning': 'Low foraging activity - check resources'
            }
        
        # ABDOMEN (should be high)
        abd_total = stats['ABDOMEN']['total']
        if abd_total > 90:
            assessments['ABDOMEN'] = {
                'status': '🟢 EXCELLENT',
                'meaning': 'Very active food sharing - strong cohesion'
            }
        elif abd_total > 50:
            assessments['ABDOMEN'] = {
                'status': '🟢 GOOD',
                'meaning': 'Normal nutritional distribution'
            }
        else:
            assessments['ABDOMEN'] = {
                'status': '🔴 CRITICAL',
                'meaning': 'Low trophallaxis - possible starvation or disease'
            }
        
        return assessments
class IndependentAntennaTracker:
    """Track antenna contacts for ALL bee pairs independently of trophallaxis"""
    
    def __init__(self):
        self.frame_antenna_data = {}  # {(bee_a, bee_b): {'contacts': {...}, 'distance': ...}}
        self.antenna_history = defaultdict(list)  # {(bee_a, bee_b): [frame_data]}
        self.max_history = 300  # Keep last 10 seconds at 30fps
        self.antenna_contact_threshold = 10  # pixels
        
        # Use shared antenna analyzer
        self.antenna_analyzer = AntennaContactAnalyzer(threshold=self.antenna_contact_threshold)
    
    def track_antenna_contacts(self, all_bee_data):
        """
        Analyze antenna contacts for ALL bee pairs in current frame.
        Called EVERY frame, independent of trophallaxis detection.
        """
        self.frame_antenna_data = {}
        bee_ids = list(all_bee_data.keys())
        
        for i, bee_a_id in enumerate(bee_ids):
            for bee_b_id in bee_ids[i+1:]:
                kp_a, _ = all_bee_data[bee_a_id]
                kp_b, _ = all_bee_data[bee_b_id]
                
                if kp_a is None or kp_b is None:
                    continue
                
                # Calculate antenna contacts
                contacts, distances = self._check_antenna_intersection(kp_a, kp_b)
                
                event_key = tuple(sorted([bee_a_id, bee_b_id]))
                self.frame_antenna_data[event_key] = {
                    'contacts': contacts,
                    'distances': distances,
                    'bee_a': bee_a_id,
                    'bee_b': bee_b_id,
                    'closest_distance': min(distances.values()) if distances else 999
                }
                
                # Store in history
                self.antenna_history[event_key].append(self.frame_antenna_data[event_key])
                
                # Keep history bounded
                if len(self.antenna_history[event_key]) > self.max_history:
                    self.antenna_history[event_key].pop(0)
    
    def _check_antenna_intersection(self, kp_a, kp_b):
        """Check antenna contacts between two bees (delegated to shared analyzer)"""
        analyzer = AntennaContactAnalyzer(threshold=self.antenna_contact_threshold)
        return analyzer.check_antenna_intersection(kp_a, kp_b)
    
    def get_aggregate_antenna_metrics(self):
        """Calculate aggregate antenna metrics from ALL frames of history"""
        total_rr = 0
        total_rl = 0
        total_lr = 0
        total_ll = 0
        total_frames_with_contact = 0
        
        for event_key, history in self.antenna_history.items():
            for frame_data in history:
                contacts = frame_data['contacts']
                if contacts['RR']:
                    total_rr += 1
                if contacts['RL']:
                    total_rl += 1
                if contacts['LR']:
                    total_lr += 1
                if contacts['LL']:
                    total_ll += 1
                
                if any(contacts.values()):
                    total_frames_with_contact += 1
        
        total = total_rr + total_rl + total_lr + total_ll
        
        if total == 0:
            return {
                'rr_ratio': 0.0,
                'll_ratio': 0.0,
                'cross_ratio': 0.0,
                'rr_count': 0,
                'rl_count': 0,
                'lr_count': 0,
                'll_count': 0,
                'total_contacts': 0,
                'frames_with_contact': 0
            }
        
        return {
            'rr_ratio': total_rr / total,
            'll_ratio': total_ll / total,
            'rl_count': total_rl,
            'lr_count': total_lr,
            'cross_ratio': (total_rl + total_lr) / total,
            'rr_count': total_rr,
            'll_count': total_ll,
            'total_contacts': total,
            'frames_with_contact': total_frames_with_contact
        }

class AntennaInterpretationEngine:
    """Interprets antenna dominance patterns and colony health"""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)  # Track history
        self.shift_threshold = 0.15
        self.variance_threshold = 0.25
        self.persistence_threshold = 3600  # 2 minutes @ 30fps
    
    def classify_rr_ratio(self, rr_ratio):
        """Classify RR ratio into behavioral mode"""
        if rr_ratio >= 0.60:
            return {
                'classification': 'STRONG_RIGHT_BIAS',
                'emoji': '🟢',
                'meaning': 'Strong Right Bias',
                'behavior': 'Affiliative (food sharing, grooming)',
                'health': 'Calm, cooperative, high trophallaxis',
                'risk': 'LOW',
                'description': f'{rr_ratio*100:.1f}% - Positive affiliative interactions'
            }
        elif rr_ratio >= 0.40:
            return {
                'classification': 'BALANCED',
                'emoji': '🟢',
                'meaning': 'Balanced Usage',
                'behavior': 'Routine communication',
                'health': 'Healthy baseline',
                'risk': 'LOW',
                'description': f'{rr_ratio*100:.1f}% - Normal routine behavior'
            }
        else:
            return {
                'classification': 'STRONG_LEFT_BIAS',
                'emoji': '🟡',
                'meaning': 'Strong Left Bias',
                'behavior': 'Defensive/aggressive',
                'health': 'Perceived threat or stress',
                'risk': 'MEDIUM-HIGH',
                'description': f'{rr_ratio*100:.1f}% - Defensive mode activated'
            }
    
    def classify_ll_ratio(self, ll_ratio):
        """Classify LL ratio (threat awareness)"""
        if ll_ratio >= 0.40:
            return {
                'classification': 'CRITICAL_DEFENSIVE',
                'emoji': '🔴',
                'status': 'Over-Extended Defense',
                'meaning': 'Excessive defensive response'
            }
        elif ll_ratio >= 0.35:
            return {
                'classification': 'HIGH_DEFENSIVE',
                'emoji': '🟡',
                'status': 'Threat-Aware & Cautious',
                'meaning': 'Active defensive posture'
            }
        else:
            return {
                'classification': 'HEALTHY_DEFENSIVE',
                'emoji': '🟢',
                'status': 'Good Threat Capability',
                'meaning': 'Balanced defensive response'
            }
    
    def classify_lateralization_index(self, li):
        """Classify LI (balance score)"""
        if -0.30 <= li <= 0.35:
            return {
                'classification': 'BALANCED',
                'emoji': '🟢',
                'status': 'IDEAL',
                'meaning': 'Perfect hemispheric balance'
            }
        elif -0.50 <= li < -0.30 or 0.35 < li <= 0.50:
            return {
                'classification': 'SLIGHTLY_BIASED',
                'emoji': '🟡',
                'status': 'IMBALANCED',
                'meaning': 'Minor bias detected'
            }
        else:
            return {
                'classification': 'SEVERELY_BIASED',
                'emoji': '🔴',
                'status': 'SEVERELY IMBALANCED',
                'meaning': 'Severe hemispheric imbalance'
            }
    
    def classify_cross_ratio(self, cross_ratio):
        """Classify cross-pattern usage (stress indicator)"""
        if cross_ratio < 0.20:
            return {
                'classification': 'LOW_STRESS',
                'emoji': '🟢',
                'status': 'LOW',
                'meaning': 'Minimal conflict/confusion'
            }
        elif cross_ratio < 0.35:
            return {
                'classification': 'MODERATE_STRESS',
                'emoji': '🟡',
                'status': 'MODERATE',
                'meaning': 'Some stress responses'
            }
        else:
            return {
                'classification': 'HIGH_STRESS',
                'emoji': '🔴',
                'status': 'HIGH',
                'meaning': 'Significant stress/confusion'
            }
    
    def classify_symmetry(self, sym_ratio):
        """Classify directional symmetry (injury check)"""
        if 0.70 <= sym_ratio <= 1.43:
            return {
                'classification': 'SYMMETRIC',
                'emoji': '🟢',
                'status': 'GOOD',
                'meaning': 'Balanced directional use'
            }
        elif 0.50 <= sym_ratio < 0.70 or 1.43 < sym_ratio <= 2.0:
            return {
                'classification': 'ASYMMETRIC',
                'emoji': '🟡',
                'status': 'POSSIBLE INJURY',
                'meaning': 'Possible injury or lateralization'
            }
        else:
            return {
                'classification': 'SEVERELY_ASYMMETRIC',
                'emoji': '🔴',
                'status': 'COMPROMISED',
                'meaning': 'Severe injury likely'
            }
    
    def generate_colony_interpretation(self, metrics, total_events):
        """Generate comprehensive colony health interpretation"""
        rr = metrics['rr_ratio']
        ll = metrics['ll_ratio']
        li = metrics['lateralization_index']
        cross = metrics['cross_ratio']
        sym = metrics['symmetry_ratio']
        
        # Determine overall status
        if rr >= 0.50 and ll < 0.35 and -0.30 <= li <= 0.35 and cross < 0.20:
            status = '🟢 EXCELLENT'
            assessment = 'EXCELLENT'
            survival = 96
            mode = 'HEALTHY & BALANCED'
        elif rr >= 0.40 and ll < 0.40 and -0.50 <= li <= 0.50 and cross < 0.35:
            status = '🟢 HEALTHY'
            assessment = 'GOOD'
            survival = 88
            mode = 'HEALTHY'
        elif rr >= 0.30 or ll < 0.50:
            status = '🟡 CAUTION'
            assessment = 'FAIR'
            survival = 75
            mode = 'MODERATE STRESS'
        else:
            status = '🔴 CRITICAL'
            assessment = 'POOR'
            survival = 50
            mode = 'SEVERE ABNORMALITY'
        
        return {
            'status': status,
            'assessment': assessment,
            'survival_prediction': survival,
            'mode': mode,
            'total_events': total_events
        }
class InferenceWorker(QObject):
    finished = pyqtSignal(str)
    frame_processed = pyqtSignal(object)
    darkest_visualization = pyqtSignal(object)
    darkest_mask_visualization = pyqtSignal(object)
    darkest_mask_no_keypoints = pyqtSignal(object)
    body_keypoints_roi_vectors = pyqtSignal(object)  # NEW: Body keypoints + ROI direction vectors
    roi_bbox_body = pyqtSignal(object)  # NEW: ROI triangles + bounding box + body keypoints
    grayscale_body_roi = pyqtSignal(object)  # NEW: Grayscale with body keypoints and ROI angles
    full_frame_darkest = pyqtSignal(object)
    research_visualization = pyqtSignal(dict)  # NEW: For research pipeline visualization
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    metrics_updated = pyqtSignal(dict)    
    def __init__(self, model_path, video_path, thresholds=None, input_mode='video'):
        super().__init__()
        self.model_path = model_path
        self.video_path = video_path        # path to video file OR image folder
        self.input_mode = input_mode        # 'video' | 'images'
        # Images are fed at a fixed virtual rate of 60 FPS.
        # For real video files the actual FPS is read from the container header.
        self.input_fps = ImageSequenceCapture.IMAGE_SEQUENCE_FPS if input_mode == 'images' else None
        self._is_running = True
        
        # Store detection thresholds (with defaults if not provided)
        # DEFAULT VALUES optimized for THIN WORKER BEE antennae
        # For yellowjackets/thicker antennae: increase min_area to 20, kernel_size to 15, min_aspect_ratio to 4.0
        if thresholds is None:
            thresholds = {
                'motion_threshold': 2.0,      # More sensitive to subtle motion (was 3.0)
                'min_area': 5,                # Worker bee antennae are thin = fewer pixels (was 15)
                'max_area': 400,              # Don't need as large (was 600)
                'min_aspect_ratio': 2.5,      # More tolerant of curved antennae (was 4.0)
                'kernel_size': 5,             # Smaller kernel for thinner structures (was 15)
                'smoothing_frames': 3
            }
        self.detection_thresholds = thresholds
        
        # ROI Triangle Thickness (controllable via slider)
        self.roi_thickness = 1

        # ROI Percentile (0.0 = auto/Otsu, >0 = keep top N% brightest within each ROI)
        self.binary_threshold = 0.0

        # Pixel darkness (0.0 = original brightness, 1.0 = fully black)
        # Applied to every raw frame BEFORE any detection method runs.
        self.pixel_darkness = thresholds.get('pixel_darkness', 0.0)

        # Antenna EMA smoothing alpha (Layer 2 temporal stabilisation)
        # 1.0 = raw per-frame, 0.1 = heavy smoothing.  Default 0.6.
        self.ema_alpha = 0.6
        
        self.keypoint_history = {}
        self.history_size = thresholds.get('smoothing_frames', 3)  # Use slider value
        self.confidence_threshold = 0.1
        
        self.direction_history = {}
        self.direction_history_size = 7
        
        self.bee_class = {}
        self.show_full_frame_darkest = False
        
        self.cached_darkness_mask = None
        self.cached_darkness_frame_id = -1
        self.frame_counter = 0
        
        self.frames_processed = 0
        self.use_cuda = TORCH_CUDA_AVAILABLE
        
        # NEW: Track video dimensions for validation
        self.last_frame_shape = None
        self.frame_dimension_mismatch_count = 0
        self.max_dimension_mismatches = 5
        
        # OPTIMIZATION: Frame-level caches for morphology operations
        self.cached_antenna_maps = None
        self.cached_antenna_maps_frame_id = -1
        self.cached_frame_components = None
        self.cached_components_frame_id = -1
        self.cached_components_binary_hash = None
        
        # NEW: Antenna tracking state - stores last known antenna positions
        self.antenna_tracked_positions = {}  # {bee_id: {'left': (x, y, conf), 'right': (x, y, conf)}}
        self.antenna_lock_frames = {}  # {bee_id: frame_count} - how long antenna has been locked
        self.antenna_lost_threshold = 15  # frames before re-detection triggers
        self.antenna_tracking_radius = 25  # search radius around last position for updates
        # Trophallaxis Detection
        self.trophallaxis_detector = TrophallaxisDetector()
        
        # NEW: Independent Antenna Tracking (works regardless of trophallaxis)
        self.antenna_tracker = IndependentAntennaTracker()
        
        # NEW: 4-Region Anatomical Tracking
        self.region_tracker = FourRegionAntennaTracker()
        
        # Memory management and tracking safety
        self.last_valid_detections = {}
        self.last_valid_tips = {}  # {bee_id: {'left_tip': (x,y), 'right_tip': (x,y), 'left_joint': (x,y), 'right_joint': (x,y)}}
        
        # Motion-based antenna confirmation
        # Tracks bbox centroids and antenna positions across frames.
        # If bee body is stationary but antenna tips move → confirmed real antennae.
        # If bee is stationary and antennae are also still → that's OK, don't penalize.
        self.bbox_centroid_history = {}   # {bee_id: deque of (cx, cy), maxlen=5}
        self.antenna_pos_history = {}    # {bee_id: deque of ((lx,ly),(rx,ry)), maxlen=5}
        self.antenna_motion_confirmed = {}  # {bee_id: bool} — once confirmed, stays confirmed
        
        self.track_id_history = set()
        self.max_bees_per_frame = 50
        
        # Frame processing control
        self.process_every_n_frames = 1
        self.frame_skip_counter = 0
        
        # Antenna contact threshold (must be set BEFORE antenna_analyzer initialization)
        self.antenna_contact_threshold = 10  # pixels
        
        # NEW: Motion-based detection and optical flow tracking
        self.motion_detector = MotionBasedAntennaDetector(observation_frames=6, motion_threshold=0.6)
        self.optical_flow_tracker = OpticalFlowAntennaTracker()
        self.antenna_state = {}  # {bee_id: 'DETECTING' or 'TRACKING'}
        
        # NEW: Research visualization processor for morphological pipeline
        kernel_size = self.detection_thresholds.get('kernel_size', 15)
        self.research_processor = MorphologicalPipelineProcessor(kernel_size=kernel_size)
        
        # NEW: Antenna processing pipeline for research visualization
        self.antenna_pipeline = AntennaProcessingPipeline()
        
    def cleanup_stale_data(self, current_bee_ids, max_history_age=300):
        """Clean up old tracking data to prevent memory leaks"""
        # Clean keypoint history for lost bees
        stale_keypoint_ids = [bee_id for bee_id in self.keypoint_history.keys() 
                              if bee_id not in current_bee_ids]
        for bee_id in stale_keypoint_ids:
            if len(self.keypoint_history.get(bee_id, [])) > 0:
                self.last_valid_detections[bee_id] = list(self.keypoint_history[bee_id])[-1]
            self.keypoint_history.pop(bee_id, None)
        
        # Clean direction history
        stale_direction_ids = [bee_id for bee_id in self.direction_history.keys() 
                               if bee_id not in current_bee_ids]
        for bee_id in stale_direction_ids:
            self.direction_history.pop(bee_id, None)
        
        # Clean antenna tracking
        self.cleanup_lost_bees(current_bee_ids)
        
        # Limit history size if too large - OPTIMIZED
        if len(self.last_valid_detections) > 100:
            sorted_ids = sorted(self.last_valid_detections.keys())[-100:]  # Keep last 100, not first 50
            self.last_valid_detections = {k: self.last_valid_detections[k] 
                                          for k in sorted_ids}
        # Clean antenna tracker history for lost bee pairs
        lost_pair_keys = [key for key in self.antenna_tracker.antenna_history.keys()
                          if key[0] not in current_bee_ids or key[1] not in current_bee_ids]
        for key in lost_pair_keys:
            self.antenna_tracker.antenna_history.pop(key, None)        
        # OPTIMIZATION: Limit track_id_history size
        if len(self.track_id_history) > 500:
            recent_ids = list(self.track_id_history)[-500:]
            self.track_id_history = set(recent_ids)
    
    def stop(self):
        self._is_running = False
    def _clear_dimension_sensitive_cache(self):
        """Clear all cached data that depends on frame dimensions"""
        self.cached_darkness_mask = None
        self.cached_darkness_frame_id = -1
        self.cached_antenna_maps = None
        self.cached_antenna_maps_frame_id = -1
        self.cached_frame_components = None
        self.cached_components_frame_id = -1
        self.cached_components_binary_hash = None
        print("[CACHE] Cleared dimension-sensitive caches")
    def refine_antenna_keypoints_unified(self, frame_gray, keypoints, box, bee_id, frame_id):
        """
        UNIFIED antenna refinement with motion validation and optical flow tracking.
        
        States:
        - DETECTING: Frames 1-6, motion-based validation
        - TRACKING: Frame 7+, optical flow tracking
        """
        if keypoints is None or box is None:
            return keypoints
        
        # Initialize state for new bee
        if bee_id not in self.antenna_state:
            self.antenna_state[bee_id] = 'DETECTING'
        
        # Check if re-detection needed
        if self.antenna_state[bee_id] == 'TRACKING':
            if self.optical_flow_tracker.should_redetect(bee_id):
                # Confidence dropped or periodic verification needed
                self.antenna_state[bee_id] = 'DETECTING'
                self.motion_detector.reset_bee(bee_id)
                self.optical_flow_tracker.reset_bee(bee_id)
        
        # STATE: DETECTING (Frames 1-6)
        if self.antenna_state[bee_id] == 'DETECTING':
            return self._detect_antennae_with_motion(frame_gray, keypoints, box, bee_id, frame_id)
        
        # STATE: TRACKING (Frame 7+)
        elif self.antenna_state[bee_id] == 'TRACKING':
            return self._track_antennae_with_optical_flow(frame_gray, keypoints, box, bee_id)
        
        return keypoints
    
    def _detect_antennae_with_motion(self, frame_gray, keypoints, box, bee_id, frame_id):
        """Motion-based detection (frames 1-6) - TRIANGLE-FIRST with ROI CROPPING"""
        
        h, w = frame_gray.shape
        
        # Step 1: Get ROI triangles FIRST (before any processing)
        head_pos = (int(keypoints[0][0]), int(keypoints[0][1]))
        head_in_roi, center_end, left_end, right_end, *_roi = self.get_angled_roi_lines(keypoints, box, bee_id)
        
        left_triangle = [head_in_roi, center_end, left_end]
        right_triangle = [head_in_roi, center_end, right_end]
        
        # Step 2: Get darkness maps (frame-level cache)
        maps = self.precompute_antenna_detection_maps(frame_gray, frame_id)
        if maps is None:
            return keypoints
        
        dark_binary = maps['binary']
        
        # ═══════════════════════════════════════════════════════════════
        # FIX: Use ROI CROPPING instead of full-frame masking
        # This guarantees structures are physically inside triangles!
        # ═══════════════════════════════════════════════════════════════

        # LEFT TRIANGLE: Get bounding box and crop
        left_pts  = np.array(left_triangle, dtype=np.int32)
        left_x_min = max(0, int(left_pts[:, 0].min()))
        left_x_max = min(w, int(left_pts[:, 0].max()))
        left_y_min = max(0, int(left_pts[:, 1].min()))
        left_y_max = min(h, int(left_pts[:, 1].max()))

        left_antenna_region = None
        left_triangle_local = []
        if left_x_max > left_x_min + 2 and left_y_max > left_y_min + 2:
            left_roi = np.ascontiguousarray(
                dark_binary[left_y_min:left_y_max, left_x_min:left_x_max])
            left_triangle_local = [(pt[0] - left_x_min, pt[1] - left_y_min)
                                   for pt in left_triangle]
            left_mask_local = np.zeros(left_roi.shape, dtype=np.uint8)
            cv2.fillConvexPoly(left_mask_local,
                               np.array(left_triangle_local, dtype=np.int32), 255)
            left_antenna_region = cv2.bitwise_and(left_roi, left_mask_local)

        # RIGHT TRIANGLE: Get bounding box and crop
        right_pts  = np.array(right_triangle, dtype=np.int32)
        right_x_min = max(0, int(right_pts[:, 0].min()))
        right_x_max = min(w, int(right_pts[:, 0].max()))
        right_y_min = max(0, int(right_pts[:, 1].min()))
        right_y_max = min(h, int(right_pts[:, 1].max()))

        right_antenna_region = None
        right_triangle_local = []
        if right_x_max > right_x_min + 2 and right_y_max > right_y_min + 2:
            right_roi = np.ascontiguousarray(
                dark_binary[right_y_min:right_y_max, right_x_min:right_x_max])
            right_triangle_local = [(pt[0] - right_x_min, pt[1] - right_y_min)
                                    for pt in right_triangle]
            right_mask_local = np.zeros(right_roi.shape, dtype=np.uint8)
            cv2.fillConvexPoly(right_mask_local,
                               np.array(right_triangle_local, dtype=np.int32), 255)
            right_antenna_region = cv2.bitwise_and(right_roi, right_mask_local)

        
        # Step 3: Detect structures in CROPPED triangle regions
        left_structures = self._detect_structures_in_triangle_cropped(
            left_antenna_region, left_triangle_local, head_pos,
            offset=(left_x_min, left_y_min), frame_shape=(h, w), side='left'
        ) if left_antenna_region is not None else []

        right_structures = self._detect_structures_in_triangle_cropped(
            right_antenna_region, right_triangle_local, head_pos,
            offset=(right_x_min, right_y_min), frame_shape=(h, w), side='right'
        ) if right_antenna_region is not None else []
        
        # Step 4: Add to motion detector
        all_detected_structures = left_structures + right_structures
        
        if not all_detected_structures:
            return keypoints
        
        self.motion_detector.add_observation(bee_id, all_detected_structures)
        
        # Step 5: Check if observation period complete
        if self.motion_detector.is_detecting(bee_id):
            # Still observing, return YOLO keypoints
            return keypoints
        
        # Step 6: Select antennae with ROI filtering
        left_antenna, right_antenna = self.motion_detector.select_antennae_with_roi(
            bee_id, head_pos, left_triangle, right_triangle
        )
        
        if left_antenna is None or right_antenna is None:
            # Selection failed, use fallback
            return self.place_antenna_keypoints_on_darkest_lines(frame_gray, keypoints, box, bee_id, frame_id)
        
        # Step 7: Place keypoints using skeleton method with STRICT validation
        refined_keypoints = keypoints.copy()
        
        # LEFT ANTENNA
        left_blob = left_antenna.get_latest_blob()
        
        # Pass triangle for validation
        left_joint, left_tip = self.place_keypoints_with_skeleton(
            left_blob, head_pos, roi_triangle=left_triangle
        )
        
        if left_joint is not None and left_tip is not None:
            refined_keypoints[5][0] = float(left_joint[0])
            refined_keypoints[5][1] = float(left_joint[1])
            refined_keypoints[5][2] = 0.95
            
            refined_keypoints[6][0] = float(left_tip[0])
            refined_keypoints[6][1] = float(left_tip[1])
            refined_keypoints[6][2] = 0.95
        
        # RIGHT ANTENNA
        right_blob = right_antenna.get_latest_blob()
        
        # Pass triangle for validation
        right_joint, right_tip = self.place_keypoints_with_skeleton(
            right_blob, head_pos, roi_triangle=right_triangle
        )
        
        if right_joint is not None and right_tip is not None:
            refined_keypoints[7][0] = float(right_joint[0])
            refined_keypoints[7][1] = float(right_joint[1])
            refined_keypoints[7][2] = 0.95
            
            refined_keypoints[8][0] = float(right_tip[0])
            refined_keypoints[8][1] = float(right_tip[1])
            refined_keypoints[8][2] = 0.95
        
        # Step 8: Initialize optical flow tracking
        self.optical_flow_tracker.initialize_tracking(
            bee_id,
            (refined_keypoints[5][0], refined_keypoints[5][1]),
            (refined_keypoints[6][0], refined_keypoints[6][1]),
            (refined_keypoints[7][0], refined_keypoints[7][1]),
            (refined_keypoints[8][0], refined_keypoints[8][1]),
            frame_gray
        )
        
        # Switch to tracking state
        self.antenna_state[bee_id] = 'TRACKING'
        
        return refined_keypoints
    def _detect_structures_in_triangle_cropped(self, triangle_region, triangle_points_local, 
                                               head_pos, offset, frame_shape, side):
        """
        Detect antenna-like structures in a CROPPED triangle region.
        Guarantees all detected structures are inside triangle by using local coordinates.
        
        Args:
            triangle_region: Cropped binary image (only triangle bounding box)
            triangle_points_local: Triangle vertices in LOCAL (cropped) coordinates
            head_pos: Head position in GLOBAL coordinates
            offset: (x_offset, y_offset) to convert back to global coordinates
            frame_shape: (height, width) of original frame
            side: 'left' or 'right'
        
        Returns:
            list: Detected structures with 'position', 'blob', 'score', 'side' in GLOBAL coordinates
        """
        # Find connected components in LOCAL (cropped) space
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            triangle_region, connectivity=8
        )
        
        detected_structures = []
        x_offset, y_offset = offset
        h_full, w_full = frame_shape
        
        # ========== USE THRESHOLDS FROM SLIDERS ==========
        min_area = self.detection_thresholds.get('min_area', 15)
        max_area = self.detection_thresholds.get('max_area', 600)
        min_aspect = self.detection_thresholds.get('min_aspect_ratio', 4.0)
        # =================================================
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
            
            # Filter for antenna-like structures - NOW USES SLIDER VALUES
            if aspect_ratio > min_aspect and min_area < area < max_area:
                # Extract blob in LOCAL space
                component_mask_local = (labels == i).astype(np.uint8)
                
                # Get centroid in LOCAL space
                centroid_local = centroids[i]
                
                # Convert centroid to GLOBAL coordinates
                centroid_global = (int(centroid_local[0] + x_offset), 
                                 int(centroid_local[1] + y_offset))
                
                # Create FULL-FRAME blob for skeleton processing
                component_mask_global = np.zeros((h_full, w_full), dtype=np.uint8)
                
                # Place LOCAL blob into GLOBAL position
                roi_h, roi_w = triangle_region.shape
                component_mask_global[y_offset:y_offset+roi_h,
                                    x_offset:x_offset+roi_w] = component_mask_local
                
                # Calculate score
                score = aspect_ratio * area
                
                detected_structures.append({
                    'position': centroid_global,
                    'blob': component_mask_global,  # Now in GLOBAL coordinates but constrained to triangle
                    'score': score,
                    'side': side
                })
        
        return detected_structures    
    def _track_antennae_with_optical_flow(self, frame_gray, keypoints, box, bee_id):
        """Optical flow tracking (frame 7+)"""
        
        head_pos = (keypoints[0][0], keypoints[0][1])
        
        # Track with optical flow
        tracked = self.optical_flow_tracker.track_frame(bee_id, frame_gray, head_pos)
        
        if tracked is None:
            # Tracking failed, trigger re-detection
            self.antenna_state[bee_id] = 'DETECTING'
            self.motion_detector.reset_bee(bee_id)
            self.optical_flow_tracker.reset_bee(bee_id)
            return keypoints
        
        # Update keypoints with tracked positions
        refined_keypoints = keypoints.copy()
        
        refined_keypoints[5][0] = float(tracked['left_joint'][0])
        refined_keypoints[5][1] = float(tracked['left_joint'][1])
        refined_keypoints[5][2] = tracked['confidence']
        
        refined_keypoints[6][0] = float(tracked['left_tip'][0])
        refined_keypoints[6][1] = float(tracked['left_tip'][1])
        refined_keypoints[6][2] = tracked['confidence']
        
        refined_keypoints[7][0] = float(tracked['right_joint'][0])
        refined_keypoints[7][1] = float(tracked['right_joint'][1])
        refined_keypoints[7][2] = tracked['confidence']
        
        refined_keypoints[8][0] = float(tracked['right_tip'][0])
        refined_keypoints[8][1] = float(tracked['right_tip'][1])
        refined_keypoints[8][2] = tracked['confidence']
        
        return refined_keypoints
    def cleanup_lost_bees(self, current_bee_ids):
        """Remove tracking data for bees that are no longer detected."""
        lost_bees = [bee_id for bee_id in self.antenna_tracked_positions.keys() 
                    if bee_id not in current_bee_ids]
        
        for bee_id in lost_bees:
            self.antenna_tracked_positions.pop(bee_id, None)
            self.antenna_lock_frames.pop(bee_id, None)
            self.bbox_centroid_history.pop(bee_id, None)
            self.antenna_pos_history.pop(bee_id, None)
            self.antenna_motion_confirmed.pop(bee_id, None)
    def safe_process_detections(self, results):
        """Safely extract detection data with error handling"""
        try:
            if not results or len(results) == 0:
                return None, None, None, None
            
            result = results[0]
            
            if result.keypoints is None or result.keypoints.data.shape[0] == 0:
                return None, None, None, None
            
            all_kpts = result.keypoints.data.cpu().numpy()
            
            if len(all_kpts) > self.max_bees_per_frame:
                print(f"[WARNING] Too many detections ({len(all_kpts)}), limiting to {self.max_bees_per_frame}")
                all_kpts = all_kpts[:self.max_bees_per_frame]
            
            if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                track_ids = track_ids[:len(all_kpts)]
            else:
                track_ids = list(range(len(all_kpts)))
            
            if hasattr(result.boxes, 'xyxy') and result.boxes.xyxy is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                boxes = boxes[:len(all_kpts)]
            else:
                boxes = []
            
            if hasattr(result.boxes, 'cls') and result.boxes.cls is not None:
                class_indices = result.boxes.cls.cpu().numpy().astype(int)
                class_indices = class_indices[:len(all_kpts)]
            else:
                class_indices = [0] * len(track_ids)
            
            return all_kpts, track_ids, boxes, class_indices
            
        except Exception as e:
            print(f"[ERROR] Failed to process detections: {e}")
            return None, None, None, None
        
    def precompute_antenna_detection_maps(self, frame_gray, frame_id):
        """
        Compute morphology operations ONCE per frame instead of per-bee.
        ZERO ACCURACY IMPACT - just caching expensive computations.
        NOW PASSES kernel_size FROM SLIDERS TO MorphologyFactory.
        """
        # Validate frame dimensions
        if frame_gray is None or frame_gray.size == 0:
            print(f"[ERROR] Invalid frame in precompute_antenna_detection_maps")
            return None
        
        h, w = frame_gray.shape
        
        if h < 100 or w < 100:
            print(f"[WARNING] Frame too small: {h}x{w}")
            return None
        
        # Return cached result if already computed for this frame
        if self.cached_antenna_maps_frame_id == frame_id and self.cached_antenna_maps is not None:
            if self.cached_antenna_maps.get('shape') == (h, w):
                return self.cached_antenna_maps
        
        # ========== GET KERNEL SIZE FROM SLIDERS ==========
        kernel_size = self.detection_thresholds.get('kernel_size', 15)
        # ==================================================
        
        # Use factory for multi-directional morphology - NOW WITH KERNEL_SIZE
        darkest = 255 - frame_gray
        combined = MorphologyFactory.compute_darkness_map(frame_gray, mode='multi', kernel_size=kernel_size)
        
        # Apply thresholding and post-processing
        blurred = cv2.GaussianBlur(combined, (3, 3), 0.5)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Light dilation
        kernel_dilate = MorphologyFactory.get_kernel('ellipse', (3, 3))
        binary_dilated = cv2.dilate(binary, kernel_dilate, iterations=1)
        
        # Cache everything with frame dimensions
        self.cached_antenna_maps = {
            'darkest': darkest,
            'combined': combined,
            'binary': binary_dilated,
            'blurred': blurred,
            'shape': (h, w)
        }
        self.cached_antenna_maps_frame_id = frame_id
        
        return self.cached_antenna_maps
    
    def get_frame_connected_components(self, binary_image, frame_id):
        """
        Compute connected components ONCE per frame.
        ZERO ACCURACY IMPACT - exact same algorithm, just cached.
        """
        # NEW: Validate input
        if binary_image is None or binary_image.size == 0:
            print("[ERROR] Invalid binary image in get_frame_connected_components")
            return (0, None, None, None)
        
        # NEW: Validate binary image is 2D
        if len(binary_image.shape) != 2:
            print(f"[ERROR] Binary image must be 2D, got shape {binary_image.shape}")
            return (0, None, None, None)
        
        # Create hash of binary image to detect changes
        try:
            img_hash = hash(binary_image.tobytes())
        except Exception as e:
            print(f"[ERROR] Failed to hash binary image: {e}")
            return (0, None, None, None)
        
        if (self.cached_components_frame_id == frame_id and 
            self.cached_components_binary_hash == img_hash and
            self.cached_frame_components is not None):
            return self.cached_frame_components
        
        # Compute once with error handling
        try:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_image, connectivity=8
            )
        except Exception as e:
            print(f"[ERROR] Connected components failed: {e}")
            return (0, None, None, None)
        
        self.cached_frame_components = (num_labels, labels, stats, centroids)
        self.cached_components_frame_id = frame_id
        self.cached_components_binary_hash = img_hash
        
        return self.cached_frame_components
    def track_antenna_with_direction_constraint(self, frame_gray, keypoints, box, bee_id):
        """
        Track antenna along its direction vector (Approach 2):
        1. Uses last frame's direction (head → tip)
        2. Only checks points along this direction line
        3. Validates darkness at each point
        4. Updates keypoints only if validation passes
        """
        tracked = keypoints.copy()
        
        # Lightweight: Single tophat operation
        darkest_map = 255 - frame_gray
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        darkest_map = cv2.morphologyEx(darkest_map, cv2.MORPH_TOPHAT, kernel)
        
        # Get head position (keypoint 0)
        head_x = int(keypoints[0][0])
        head_y = int(keypoints[0][1])
        
        # ===== TRACK LEFT ANTENNA (keypoints 5, 6) =====
        last_left_x = int(keypoints[5][0])
        last_left_y = int(keypoints[5][1])
        last_left_conf = keypoints[5][2]
        
        # Direction vector: from head to last antenna tip
        dir_x = last_left_x - head_x
        dir_y = last_left_y - head_y
        
        # Normalize direction
        dir_x, dir_y = AntennaGeometry.normalize_vector(dir_x, dir_y)
        
        # Search along direction line: 11 points - OPTIMIZED with vectorization
        search_offsets = np.array([0, 3, 8, 15, 25, 35, -3, -8, -15, -25, -35], dtype=np.float32)
        search_x = (last_left_x + dir_x * search_offsets).astype(np.int32)
        search_y = (last_left_y + dir_y * search_offsets).astype(np.int32)
        
        # Vectorized bounds check
        valid_mask = ((search_y >= 0) & (search_y < darkest_map.shape[0]) & 
                      (search_x >= 0) & (search_x < darkest_map.shape[1]))
        
        # Vectorized darkness lookup
        valid_indices = np.where(valid_mask)[0]
        best_x, best_y, best_darkness = None, None, 0
        
        if len(valid_indices) > 0:
            darkness_values = darkest_map[search_y[valid_indices], search_x[valid_indices]]
            best_local_idx = np.argmax(darkness_values)
            best_idx = valid_indices[best_local_idx]
            best_darkness = darkness_values[best_local_idx]
            best_x = int(search_x[best_idx])
            best_y = int(search_y[best_idx])
        
        # VALIDATION: Only update if darkness is significant
        if best_darkness > 30:
            tracked[5][0] = float(best_x)
            tracked[5][1] = float(best_y)
            tracked[5][2] = min(0.99, last_left_conf + 0.03)
            
            tracked[6][0] = float(best_x)
            tracked[6][1] = float(best_y)
            tracked[6][2] = min(0.99, last_left_conf + 0.02)
        else:
            # Darkness too low = drifted off antenna, decay confidence FAST
            tracked[5][2] = last_left_conf * 0.70
            tracked[6][2] = last_left_conf * 0.65
        
        # ===== TRACK RIGHT ANTENNA (keypoints 7, 8) =====
        last_right_x = int(keypoints[7][0])
        last_right_y = int(keypoints[7][1])
        last_right_conf = keypoints[7][2]
        
        # Direction vector: from head to last antenna tip
        dir_x = last_right_x - head_x
        dir_y = last_right_y - head_y
        
        # Normalize direction
        dir_len = np.sqrt(dir_x**2 + dir_y**2)
        if dir_len > 0:
            dir_x /= dir_len
            dir_y /= dir_len
        else:
            dir_x, dir_y = 1.0, 0.0
        
        # Search along direction line: 11 points - OPTIMIZED with vectorization
        search_offsets = np.array([0, 3, 8, 15, 25, 35, -3, -8, -15, -25, -35], dtype=np.float32)
        search_x = (last_right_x + dir_x * search_offsets).astype(np.int32)
        search_y = (last_right_y + dir_y * search_offsets).astype(np.int32)
        
        # Vectorized bounds check
        valid_mask = ((search_y >= 0) & (search_y < darkest_map.shape[0]) & 
                      (search_x >= 0) & (search_x < darkest_map.shape[1]))
        
        # Vectorized darkness lookup
        valid_indices = np.where(valid_mask)[0]
        best_x, best_y, best_darkness = None, None, 0
        
        if len(valid_indices) > 0:
            darkness_values = darkest_map[search_y[valid_indices], search_x[valid_indices]]
            best_local_idx = np.argmax(darkness_values)
            best_idx = valid_indices[best_local_idx]
            best_darkness = darkness_values[best_local_idx]
            best_x = int(search_x[best_idx])
            best_y = int(search_y[best_idx])
        
        # VALIDATION: Only update if darkness is significant
        if best_darkness > 30:
            tracked[7][0] = float(best_x)
            tracked[7][1] = float(best_y)
            tracked[7][2] = min(0.99, last_right_conf + 0.03)
            
            tracked[8][0] = float(best_x)
            tracked[8][1] = float(best_y)
            tracked[8][2] = min(0.99, last_right_conf + 0.02)
        else:
            # Darkness too low = drifted off antenna, decay confidence FAST
            tracked[7][2] = last_right_conf * 0.70
            tracked[8][2] = last_right_conf * 0.65
        
        return tracked    
    def smooth_direction_vector(self, bee_id, dir_x, dir_y):
        if bee_id not in self.direction_history:
            self.direction_history[bee_id] = deque(maxlen=self.direction_history_size)
        
        # Normalize input vector
        dir_x, dir_y = AntennaGeometry.normalize_vector(dir_x, dir_y)
        
        self.direction_history[bee_id].append((dir_x, dir_y))
        
        history = list(self.direction_history[bee_id])
        avg_x = np.mean([h[0] for h in history])
        avg_y = np.mean([h[1] for h in history])
        
        # Normalize averaged vector
        avg_x, avg_y = AntennaGeometry.normalize_vector(avg_x, avg_y)
        
        return avg_x, avg_y
    
    def smooth_keypoints(self, bee_id, keypoints):
        if bee_id not in self.keypoint_history:
            self.keypoint_history[bee_id] = deque(maxlen=self.history_size)
        
        self.keypoint_history[bee_id].append(keypoints.copy())
        
        smoothed = np.zeros_like(keypoints)
        history_list = list(self.keypoint_history[bee_id])
        
        for kp_idx in range(len(keypoints)):
            valid_positions = []
            valid_confidences = []
            
            for hist_kpts in history_list:
                if hist_kpts[kp_idx][2] > self.confidence_threshold:
                    valid_positions.append(hist_kpts[kp_idx][:2])
                    valid_confidences.append(hist_kpts[kp_idx][2])
            
            if valid_positions:
                valid_positions = np.array(valid_positions)
                valid_confidences = np.array(valid_confidences)
                weights = valid_confidences / valid_confidences.sum()
                
                smoothed[kp_idx][0] = np.average(valid_positions[:, 0], weights=weights)
                smoothed[kp_idx][1] = np.average(valid_positions[:, 1], weights=weights)
                smoothed[kp_idx][2] = np.mean(valid_confidences)
            else:
                smoothed[kp_idx] = keypoints[kp_idx]
        
        return smoothed
    
    def validate_tip_keypoints(self, bee_id, keypoints, box):
        """
        Validate antenna tip keypoints (6, 8) are within ROI triangles.
        If a tip is outside its ROI triangle, revert to last known good position.
        This prevents the visual 'jump' artifact where tips fly to random positions.
        
        Keypoint indices:
            0: head, 1-3: thorax, 4: abdomen tip
            5: left joint (scape), 6: left tip (flagellum)
            7: right joint (scape), 8: right tip (flagellum)
        """
        if box is None or keypoints[0][2] < 0.1:
            return keypoints
        
        try:
            # Get the ROI triangles for this bee
            head_pos, center_end, left_end, right_end, *_roi = self.get_angled_roi_lines(keypoints, box, bee_id)
            left_triangle = [head_pos, center_end, left_end]
            right_triangle = [head_pos, center_end, right_end]
            
            # Also compute a max reasonable distance (bbox diagonal * 1.5)
            bbox_diag = np.sqrt((box[2] - box[0])**2 + (box[3] - box[1])**2)
            max_dist = bbox_diag * 1.5
            head_pt = np.array([keypoints[0][0], keypoints[0][1]])
            
            validated = keypoints.copy()
            updated_tips = {}
            
            # Check LEFT tip (index 6) against LEFT triangle
            if keypoints[6][2] > 0.1:
                tip_pt = (int(keypoints[6][0]), int(keypoints[6][1]))
                dist = np.linalg.norm(np.array([keypoints[6][0], keypoints[6][1]]) - head_pt)
                in_triangle = self.point_in_triangle(tip_pt, left_triangle)
                
                if in_triangle and dist < max_dist:
                    # Valid - store as last good position
                    updated_tips['left_tip'] = tip_pt
                else:
                    # Invalid - revert to last known good
                    if bee_id in self.last_valid_tips and 'left_tip' in self.last_valid_tips[bee_id]:
                        prev = self.last_valid_tips[bee_id]['left_tip']
                        validated[6][0] = float(prev[0])
                        validated[6][1] = float(prev[1])
                        validated[6][2] = keypoints[6][2] * 0.7  # lower confidence
                    # else keep YOLO's prediction as-is (first frame fallback)
            
            # Check LEFT joint (index 5) against LEFT triangle
            if keypoints[5][2] > 0.1:
                joint_pt = (int(keypoints[5][0]), int(keypoints[5][1]))
                dist = np.linalg.norm(np.array([keypoints[5][0], keypoints[5][1]]) - head_pt)
                in_triangle = self.point_in_triangle(joint_pt, left_triangle)
                
                if in_triangle and dist < max_dist:
                    updated_tips['left_joint'] = joint_pt
                else:
                    if bee_id in self.last_valid_tips and 'left_joint' in self.last_valid_tips[bee_id]:
                        prev = self.last_valid_tips[bee_id]['left_joint']
                        validated[5][0] = float(prev[0])
                        validated[5][1] = float(prev[1])
                        validated[5][2] = keypoints[5][2] * 0.7
            
            # Check RIGHT tip (index 8) against RIGHT triangle
            if keypoints[8][2] > 0.1:
                tip_pt = (int(keypoints[8][0]), int(keypoints[8][1]))
                dist = np.linalg.norm(np.array([keypoints[8][0], keypoints[8][1]]) - head_pt)
                in_triangle = self.point_in_triangle(tip_pt, right_triangle)
                
                if in_triangle and dist < max_dist:
                    updated_tips['right_tip'] = tip_pt
                else:
                    if bee_id in self.last_valid_tips and 'right_tip' in self.last_valid_tips[bee_id]:
                        prev = self.last_valid_tips[bee_id]['right_tip']
                        validated[8][0] = float(prev[0])
                        validated[8][1] = float(prev[1])
                        validated[8][2] = keypoints[8][2] * 0.7
            
            # Check RIGHT joint (index 7) against RIGHT triangle
            if keypoints[7][2] > 0.1:
                joint_pt = (int(keypoints[7][0]), int(keypoints[7][1]))
                dist = np.linalg.norm(np.array([keypoints[7][0], keypoints[7][1]]) - head_pt)
                in_triangle = self.point_in_triangle(joint_pt, right_triangle)
                
                if in_triangle and dist < max_dist:
                    updated_tips['right_joint'] = joint_pt
                else:
                    if bee_id in self.last_valid_tips and 'right_joint' in self.last_valid_tips[bee_id]:
                        prev = self.last_valid_tips[bee_id]['right_joint']
                        validated[7][0] = float(prev[0])
                        validated[7][1] = float(prev[1])
                        validated[7][2] = keypoints[7][2] * 0.7
            
            # Update stored valid positions
            if bee_id not in self.last_valid_tips:
                self.last_valid_tips[bee_id] = {}
            self.last_valid_tips[bee_id].update(updated_tips)
            
            return validated
            
        except Exception as e:
            # If validation fails for any reason, return original keypoints
            return keypoints
    
    def update_antenna_motion_tracking(self, bee_id, keypoints, box):
        """
        Track bbox centroid and antenna tip positions across frames.
        
        Motion-based confirmation logic:
        - If bee body is stationary (bbox centroid ~still) but antenna tips
          are moving → confirmed real antennae (boost confidence).
        - If bee is stationary and antennae are also still → OK, don't penalize.
          Stationary bees can have motionless antennae.
        - Once confirmed, stays confirmed for that bee_id.
        
        Returns:
            confidence_boost: float 0.0-0.3 to add to antenna keypoint confidence
        """
        if box is None:
            return 0.0
        
        # Bbox centroid
        bcx = (box[0] + box[2]) / 2.0
        bcy = (box[1] + box[3]) / 2.0
        
        if bee_id not in self.bbox_centroid_history:
            self.bbox_centroid_history[bee_id] = deque(maxlen=5)
            self.antenna_pos_history[bee_id] = deque(maxlen=5)
        
        self.bbox_centroid_history[bee_id].append((bcx, bcy))
        
        # Antenna tip positions (left_tip=6, right_tip=8)
        lt = (keypoints[6][0], keypoints[6][1]) if keypoints[6][2] > 0.1 else None
        rt = (keypoints[8][0], keypoints[8][1]) if keypoints[8][2] > 0.1 else None
        self.antenna_pos_history[bee_id].append((lt, rt))
        
        # Already confirmed? Keep the boost
        if self.antenna_motion_confirmed.get(bee_id, False):
            return 0.15
        
        # Need at least 3 frames of history
        bh = self.bbox_centroid_history[bee_id]
        ah = self.antenna_pos_history[bee_id]
        if len(bh) < 3:
            return 0.0
        
        # Compute body motion (max displacement over recent frames)
        body_displacements = []
        for i in range(1, len(bh)):
            d = np.sqrt((bh[i][0] - bh[i-1][0])**2 + (bh[i][1] - bh[i-1][1])**2)
            body_displacements.append(d)
        max_body_motion = max(body_displacements) if body_displacements else 0
        
        # Compute antenna motion (sum of tip displacements)
        antenna_displacements = []
        for i in range(1, len(ah)):
            for side in range(2):  # 0=left, 1=right
                prev = ah[i-1][side]
                curr = ah[i][side]
                if prev is not None and curr is not None:
                    d = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
                    antenna_displacements.append(d)
        max_antenna_motion = max(antenna_displacements) if antenna_displacements else 0
        
        # Decision logic
        body_stationary = max_body_motion < 4.0  # pixels per frame
        antenna_moving = max_antenna_motion > 3.0  # pixels per frame
        
        if body_stationary and antenna_moving:
            # Body still, antennae moving → CONFIRMED real antennae
            self.antenna_motion_confirmed[bee_id] = True
            return 0.2  # significant confidence boost
        
        # Body stationary, antennae also stationary → OK, no penalty, small boost
        if body_stationary and not antenna_moving:
            return 0.05
        
        return 0.0

    def point_in_triangle(self, point, triangle):
        px, py = point
        p1, p2, p3 = triangle
        
        def sign(px, py, ax, ay, bx, by):
            return (px - bx) * (ay - by) - (ax - bx) * (py - by)
        
        d1 = sign(px, py, p1[0], p1[1], p2[0], p2[1])
        d2 = sign(px, py, p2[0], p2[1], p3[0], p3[1])
        d3 = sign(px, py, p3[0], p3[1], p1[0], p1[1])
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        return not (has_neg and has_pos)
    def _point_away_from_triangle_edges(self, point, triangle, margin=5):
        """
        Check if point is at least 'margin' pixels away from triangle edges.
        This prevents detecting artificial endpoints at triangle boundaries.
        
        Args:
            point: (x, y) tuple
            triangle: [(x1,y1), (x2,y2), (x3,y3)]
            margin: minimum distance from edges in pixels
        
        Returns:
            bool: True if point is far enough from all edges
        """
        px, py = point
        p1, p2, p3 = triangle
        
        # Calculate distance to each edge
        def point_to_line_distance(px, py, x1, y1, x2, y2):
            """Distance from point to line segment"""
            # Line segment vector
            dx = x2 - x1
            dy = y2 - y1
            
            # If line segment is degenerate (point)
            if dx == 0 and dy == 0:
                return np.sqrt((px - x1)**2 + (py - y1)**2)
            
            # Parameter t for projection
            t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))
            
            # Closest point on line segment
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            
            # Distance to closest point
            return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
        
        # Check distance to all three edges
        dist1 = point_to_line_distance(px, py, p1[0], p1[1], p2[0], p2[1])
        dist2 = point_to_line_distance(px, py, p2[0], p2[1], p3[0], p3[1])
        dist3 = point_to_line_distance(px, py, p3[0], p3[1], p1[0], p1[1])
        
        # Point must be at least 'margin' pixels from ALL edges
        return dist1 >= margin and dist2 >= margin and dist3 >= margin
    def point_outside_both_triangles(self, point, left_triangle, right_triangle):
        """
        Check if a point is OUTSIDE both the left and right ROI triangles.
        
        Args:
            point: (x, y) tuple
            left_triangle: [(x1, y1), (x2, y2), (x3, y3)]
            right_triangle: [(x1, y1), (x2, y2), (x3, y3)]
        
        Returns:
            True if point is outside BOTH triangles, False otherwise
        """
        in_left = self.point_in_triangle(point, left_triangle)
        in_right = self.point_in_triangle(point, right_triangle)
        
        return not (in_left or in_right)    
    def compute_and_cache_darkness_mask(self, frame_gray, frame_id):
        """Compute darkness mask emphasizing thin elongated structures"""
        if self.cached_darkness_mask is not None and self.cached_darkness_frame_id == frame_id:
            return self.cached_darkness_mask
        
        # Use factory for thin darkness map
        darkest_map = MorphologyFactory.compute_darkness_map(frame_gray, mode='thin')
        
        # Adaptive threshold
        _, darkest_map = cv2.threshold(darkest_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove small noise while preserving thin lines
        kernel_clean = MorphologyFactory.get_kernel('ellipse', (3, 3))
        darkest_map = cv2.morphologyEx(darkest_map, cv2.MORPH_OPEN, kernel_clean)
        
        self.cached_darkness_mask = darkest_map
        self.cached_darkness_frame_id = frame_id
        
        return darkest_map
    
    def find_darkest_pixel_in_triangle_fast(self, frame_gray, triangle, antenna_keypoint, search_radius=DARKEST_SEARCH_RADIUS):
        """OPTIMIZED: Vectorized darkest pixel search using Numba JIT."""
        if frame_gray is None:
            return antenna_keypoint
        
        xs = [p[0] for p in triangle]
        ys = [p[1] for p in triangle]
        
        y1, y2, x1, x2 = AntennaGeometry.get_safe_roi(
            (min(xs) + max(xs)) / 2, 
            (min(ys) + max(ys)) / 2, 
            max(max(xs) - min(xs), max(ys) - min(ys)), 
            frame_gray.shape
        )
        
        if x2 <= x1 or y2 <= y1:
            return antenna_keypoint
        
        roi = frame_gray[int(y1):int(y2), int(x1):int(x2)]
        roi_h, roi_w = roi.shape
        
        y_coords, x_coords = np.meshgrid(np.arange(roi_h), np.arange(roi_w), indexing='ij')
        
        x_coords_global = x_coords + int(x1)
        y_coords_global = y_coords + int(y1)
        
        try:
            if NUMBA_AVAILABLE:
                p1, p2, p3 = triangle
                valid_mask = points_in_triangle_numba(
                    x_coords_global.ravel().astype(np.float64),
                    y_coords_global.ravel().astype(np.float64),
                    float(p1[0]), float(p1[1]),
                    float(p2[0]), float(p2[1]),
                    float(p3[0]), float(p3[1])
                ).reshape(roi.shape)
            else:
                valid_mask = np.zeros(roi.shape, dtype=bool)
                for y in range(roi_h):
                    for x in range(roi_w):
                        if self.point_in_triangle((x + int(x1), y + int(y1)), triangle):
                            valid_mask[y, x] = True
        except Exception:
            valid_mask = np.zeros(roi.shape, dtype=bool)
            for y in range(roi_h):
                for x in range(roi_w):
                    if self.point_in_triangle((x + int(x1), y + int(y1)), triangle):
                        valid_mask[y, x] = True
        
        if not valid_mask.any():
            return antenna_keypoint
        
        roi_masked = roi.copy()
        roi_masked[~valid_mask] = 255
        
        min_pos = np.unravel_index(np.argmin(roi_masked), roi_masked.shape)
        dark_y, dark_x = min_pos
        
        dark_x_frame = dark_x + int(x1)
        dark_y_frame = dark_y + int(y1)
        
        antenna_x, antenna_y = int(antenna_keypoint[0]), int(antenna_keypoint[1])
        dist = np.sqrt((dark_x_frame - antenna_x)**2 + (dark_y_frame - antenna_y)**2)
        
        if dist < search_radius:
            return (float(dark_x_frame), float(dark_y_frame))
        
        return antenna_keypoint
    
    def get_angled_roi_lines(self, keypoints, box, bee_id):
        """
        Compute ROI lines and polygon regions for antenna detection.
        
        Three lines extend from head to bounding box edges:
        - Center (green): body direction
        - Left (red): center + LEFT_ANGLE_OFFSET
        - Right (blue): center + RIGHT_ANGLE_OFFSET
        
        ROI polygons trace from head → line endpoint → along bbox perimeter → other line endpoint → head.
        Shapes are variable (triangle, quad, pentagon) depending on how lines hit the bbox.
        
        Returns:
            (head_x, head_y), center_end, left_end, right_end, left_roi_polygon, right_roi_polygon
        """
        x1, y1, x2, y2 = self.get_expanded_roi_bounds(box, expand_percent=25)
        head_x, head_y = int(keypoints[0][0]), int(keypoints[0][1])
        
        # Clamp head inside bbox (head must be inside for ray casting)
        head_x = max(x1, min(x2, head_x))
        head_y = max(y1, min(y2, head_y))
        
        kp0_x, kp0_y = keypoints[0][0], keypoints[0][1]
        if len(keypoints) > 1 and keypoints[1][2] > 0.1:
            kp1_x, kp1_y = keypoints[1][0], keypoints[1][1]
        else:
            kp1_x, kp1_y = (x1 + x2) / 2, (y1 + y2) / 2
        
        dir_x = kp0_x - kp1_x
        dir_y = kp0_y - kp1_y
        dir_x, dir_y = self.smooth_direction_vector(bee_id, dir_x, dir_y)
        
        center_angle_rad = np.arctan2(dir_y, dir_x)
        center_angle_deg = np.degrees(center_angle_rad)
        left_angle_rad = np.radians(center_angle_deg + LEFT_ANGLE_OFFSET)
        right_angle_rad = np.radians(center_angle_deg + RIGHT_ANGLE_OFFSET)
        
        # Ray-cast each line to bbox edges
        center_end = self._ray_bbox_hit(head_x, head_y, center_angle_rad, x1, y1, x2, y2)
        left_end = self._ray_bbox_hit(head_x, head_y, left_angle_rad, x1, y1, x2, y2)
        right_end = self._ray_bbox_hit(head_x, head_y, right_angle_rad, x1, y1, x2, y2)
        
        # Build polygon ROIs by tracing bbox perimeter between line endpoints
        head = (head_x, head_y)
        bbox_corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]  # TL, TR, BR, BL clockwise
        
        # Left ROI: between center line and left line (CCW sector from center_angle to left_angle)
        left_roi = self._build_roi_polygon(
            head, center_end, left_end, center_angle_rad, left_angle_rad, bbox_corners, x1, y1, x2, y2
        )
        # Right ROI: between right line and center line (CCW sector from right_angle to center_angle)
        right_roi = self._build_roi_polygon(
            head, right_end, center_end, right_angle_rad, center_angle_rad, bbox_corners, x1, y1, x2, y2
        )
        
        return head, center_end, left_end, right_end, left_roi, right_roi
    
    def _ray_bbox_hit(self, ox, oy, angle_rad, x1, y1, x2, y2):
        """Find where a ray from (ox,oy) at angle_rad first hits the bbox boundary."""
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        
        best_t = float('inf')
        hit = None
        
        # Check 4 bbox edges
        # Right edge x=x2
        if abs(dx) > 1e-10:
            t = (x2 - ox) / dx
            if t > 1e-6:
                py = oy + t * dy
                if y1 - 1 <= py <= y2 + 1 and t < best_t:
                    best_t = t
                    hit = (int(x2), int(np.clip(py, y1, y2)))
        # Left edge x=x1
        if abs(dx) > 1e-10:
            t = (x1 - ox) / dx
            if t > 1e-6:
                py = oy + t * dy
                if y1 - 1 <= py <= y2 + 1 and t < best_t:
                    best_t = t
                    hit = (int(x1), int(np.clip(py, y1, y2)))
        # Bottom edge y=y2
        if abs(dy) > 1e-10:
            t = (y2 - oy) / dy
            if t > 1e-6:
                px = ox + t * dx
                if x1 - 1 <= px <= x2 + 1 and t < best_t:
                    best_t = t
                    hit = (int(np.clip(px, x1, x2)), int(y2))
        # Top edge y=y1
        if abs(dy) > 1e-10:
            t = (y1 - oy) / dy
            if t > 1e-6:
                px = ox + t * dx
                if x1 - 1 <= px <= x2 + 1 and t < best_t:
                    best_t = t
                    hit = (int(np.clip(px, x1, x2)), int(y1))
        
        if hit is not None:
            return hit
        # Fallback: project forward and clamp
        fallback_len = max(x2 - x1, y2 - y1)
        return (int(np.clip(ox + dx * fallback_len, x1, x2)),
                int(np.clip(oy + dy * fallback_len, y1, y2)))
    
    def _build_roi_polygon(self, head, end1, end2, angle1, angle2, bbox_corners, x1, y1, x2, y2):
        """
        Build ROI polygon: head → end1 → bbox corners in sector → end2 → head.
        Includes bbox corners whose angle from head falls in the CCW arc from angle1 to angle2.
        """
        polygon = [head, end1]
        
        # Collect corners whose angle from head is in the CCW sector [angle1, angle2]
        corners_in_sector = []
        for corner in bbox_corners:
            corner_angle = np.arctan2(corner[1] - head[1], corner[0] - head[0])
            if self._angle_in_ccw_sector(corner_angle, angle1, angle2):
                # Sort key: angular distance from angle1 going CCW
                sort_key = (corner_angle - angle1) % (2 * np.pi)
                corners_in_sector.append((sort_key, corner))
        
        # Sort by angular distance from first line
        corners_in_sector.sort(key=lambda x: x[0])
        
        for _, corner in corners_in_sector:
            polygon.append(corner)
        
        polygon.append(end2)
        return polygon
    
    def _angle_in_ccw_sector(self, angle, start, end):
        """Check if angle lies in the counter-clockwise arc from start to end."""
        # Normalize all to [0, 2π)
        a = angle % (2 * np.pi)
        s = start % (2 * np.pi)
        e = end % (2 * np.pi)
        
        if s <= e:
            return s <= a <= e
        else:  # wraps around 0
            return a >= s or a <= e
    
    def point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon (any shape). Replaces point_in_triangle."""
        contour = np.array(polygon, dtype=np.float32)
        result = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
        return result >= 0
    
    
    def find_longest_thin_lines(self, frame_gray, keypoints, box):
        """ENHANCED: Extract longest thin line structures with better filtering"""
        if frame_gray is None or box is None:
            return None, None
        
        # Expand bounding box by 25%
        x1, y1, x2, y2 = self.get_expanded_roi_bounds(box, expand_percent=25)
        kp0_x, kp0_y = int(keypoints[0][0]), int(keypoints[0][1])
        
        # ========== GET KERNEL SIZE FROM SLIDERS ==========
        kernel_size = self.detection_thresholds.get('kernel_size', 15)
        # ==================================================
        
        # Use factory for multi-directional darkness map - NOW WITH KERNEL_SIZE
        darkest_map = MorphologyFactory.compute_darkness_map(frame_gray, mode='multi', kernel_size=kernel_size)
        
        # Adaptive threshold with noise reduction
        darkest_map = cv2.GaussianBlur(darkest_map, (3, 3), 0.5)
        _, darkest_map = cv2.threshold(darkest_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove tiny noise while preserving thin lines
        kernel_open = MorphologyFactory.get_kernel('ellipse', (2, 2))
        darkest_map = cv2.morphologyEx(darkest_map, cv2.MORPH_OPEN, kernel_open)
        
        # Extract ROI
        roi = darkest_map[y1:y2, x1:x2]
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi, connectivity=8)
        
        if num_labels <= 1:
            return None, None
        
        # ENHANCED: Better filtering for antenna-like structures
        # ========== USE THRESHOLDS FROM SLIDERS ==========
        min_area = self.detection_thresholds.get('min_area', 15)
        max_area = self.detection_thresholds.get('max_area', 600)
        min_aspect = self.detection_thresholds.get('min_aspect_ratio', 4.0)
        # =================================================
        
        valid_lines = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
            
            # ENHANCED: Filtering for long, thin structures - NOW USES SLIDER VALUES
            if aspect_ratio > min_aspect and min_area < area < max_area:
                component_mask = (labels == i).astype(np.uint8)
                points = np.column_stack(np.where(component_mask > 0))
                
                if len(points) < 3:
                    continue
                
                # ENHANCED: More efficient distance calculation using convex hull endpoints
                # Find convex hull for more accurate endpoints
                hull_points = cv2.convexHull(points.astype(np.float32))
                hull_points = hull_points.reshape(-1, 2).astype(np.int32)
                
                max_dist = 0
                endpoint1, endpoint2 = None, None
                
                # Only check hull points (much faster, more accurate)
                for j in range(len(hull_points)):
                    for k in range(j + 1, len(hull_points)):
                        dist = np.sqrt((hull_points[j][0] - hull_points[k][0])**2 + 
                                     (hull_points[j][1] - hull_points[k][1])**2)
                        if dist > max_dist:
                            max_dist = dist
                            endpoint1 = hull_points[j]
                            endpoint2 = hull_points[k]
                
                # ENHANCED: Higher minimum length threshold for cleaner results
                if endpoint1 is not None and max_dist > 20:
                    ep1_frame = (endpoint1[1] + x1, endpoint1[0] + y1)
                    ep2_frame = (endpoint2[1] + x1, endpoint2[0] + y1)
                    
                    dist_from_head = min(
                        np.sqrt((ep1_frame[0] - kp0_x)**2 + (ep1_frame[1] - kp0_y)**2),
                        np.sqrt((ep2_frame[0] - kp0_x)**2 + (ep2_frame[1] - kp0_y)**2)
                    )
                    
                    # ENHANCED: Composite score favoring longer lines
                    score = max_dist * aspect_ratio / (dist_from_head + 1)
                    
                    valid_lines.append({
                        'length': max_dist,
                        'endpoints': (ep1_frame, ep2_frame),
                        'dist_from_head': dist_from_head,
                        'aspect_ratio': aspect_ratio,
                        'score': score,
                        'area': area
                    })
        
        if len(valid_lines) == 0:
            return None, None
        
        # ENHANCED: Sort by composite score (length * aspect_ratio / distance)
        valid_lines.sort(key=lambda x: x['score'], reverse=True)
        
        # Return the farthest endpoint from head for each of the top 2 lines
        point1, point2 = None, None
        
        if len(valid_lines) > 0:
            ep1, ep2 = valid_lines[0]['endpoints']
            dist1 = np.sqrt((ep1[0] - kp0_x)**2 + (ep1[1] - kp0_y)**2)
            dist2 = np.sqrt((ep2[0] - kp0_x)**2 + (ep2[1] - kp0_y)**2)
            point1 = ep1 if dist1 > dist2 else ep2
        
        if len(valid_lines) > 1:
            ep1, ep2 = valid_lines[1]['endpoints']
            dist1 = np.sqrt((ep1[0] - kp0_x)**2 + (ep1[1] - kp0_y)**2)
            dist2 = np.sqrt((ep2[0] - kp0_x)**2 + (ep2[1] - kp0_y)**2)
            point2 = ep1 if dist1 > dist2 else ep2
        
        return point1, point2
    
    def draw_darkest_pixels_visualization(self, frame_gray, all_bee_data):
        if frame_gray is None:
            return None
        
        viz_frame = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        darkest_mask = 255 - frame_gray
        darkest_mask = cv2.morphologyEx(darkest_mask, cv2.MORPH_TOPHAT, kernel)
        darkest_mask = cv2.GaussianBlur(darkest_mask, (5, 5), 1.0)
        
        darkest_enhanced = cv2.applyColorMap(darkest_mask, cv2.COLORMAP_TURBO)
        
        edges = cv2.Canny(frame_gray, 20, 80)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_colored[edges > 0] = [255, 255, 0]
        
        viz_frame = cv2.addWeighted(viz_frame, 0.4, darkest_enhanced, 0.4, 0)
        viz_frame = cv2.addWeighted(viz_frame, 1.0, edges_colored, 0.3, 0)
        
        for bee_id, (keypoints, box) in all_bee_data.items():
            if box is None:
                continue
            
            hue = (bee_id * 40) % 180
            color_base = cv2.cvtColor(np.uint8([[[hue, 255, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
            color_tuple = (int(color_base[0]), int(color_base[1]), int(color_base[2]))
            
            # Expand bounding box by 25% for display
            x1, y1, x2, y2 = self.get_expanded_roi_bounds(box, expand_percent=25)
            cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color_tuple, 2)
            
            head_pos, center_end, left_end, right_end, *_roi = self.get_angled_roi_lines(keypoints, box, bee_id)
            
            left_triangle = np.array([head_pos, center_end, left_end], dtype=np.int32)
            right_triangle = np.array([head_pos, center_end, right_end], dtype=np.int32)
            
            cv2.polylines(viz_frame, [left_triangle], True, (255, 200, 100), 2)
            cv2.polylines(viz_frame, [right_triangle], True, (100, 200, 255), 2)
            
            label_text = f"ID: {bee_id}"
            cv2.putText(viz_frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_tuple, 2)
        
        return viz_frame
    
    def draw_darkest_mask_visualization(self, frame_gray, all_bee_data):
        """IMPROVED: Dark lines only with intensity-based coloring - STABLE KEYPOINTS"""
        if frame_gray is None:
            return None
        
        h, w = frame_gray.shape
        mask_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Use factory for multi-directional darkness map
        darkest_map = MorphologyFactory.compute_darkness_map(frame_gray, mode='multi')
        
        # Adaptive threshold
        _, dark_binary = cv2.threshold(darkest_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply intensity-based coloring: darker pixels get more intense colors
        # Normalize darkness map to 0-255 range
        darkest_map_normalized = cv2.normalize(darkest_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply Viridis colormap with intensity
        darkest_colored = cv2.applyColorMap(darkest_map_normalized, cv2.COLORMAP_VIRIDIS)
        
        # Create mask for only dark line pixels
        dark_line_mask = dark_binary > 0
        
        # Initialize output with black background
        mask_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Copy only dark line pixels to output
        mask_frame[dark_line_mask] = darkest_colored[dark_line_mask]
        
        for bee_id, (keypoints, box) in all_bee_data.items():
            if box is None:
                continue
            
            head_pos, center_end, left_end, right_end, *_roi = self.get_angled_roi_lines(keypoints, box, bee_id)
            
            left_triangle = np.array([head_pos, center_end, left_end], dtype=np.int32)
            right_triangle = np.array([head_pos, center_end, right_end], dtype=np.int32)
            
            # Draw triangle outlines
            cv2.polylines(mask_frame, [left_triangle], True, (255, 200, 100), 2)
            cv2.polylines(mask_frame, [right_triangle], True, (100, 200, 255), 2)
            
            # Yellow and cyan antenna blob visualization removed
            
            # ENHANCED: Stable keypoint plotting with reduced flickering
            valid_keypoints = {}
            for idx, (x, y, conf) in enumerate(keypoints):
                # Use 0.05 confidence threshold (lower = more stable, less flicker)
                if conf > 0.05:
                    x_int, y_int = int(x), int(y)
                    
                    # Bounds check
                    if 0 <= y_int < h and 0 <= x_int < w:
                        valid_keypoints[idx] = (x_int, y_int)
                        
                        # Get intensity value at keypoint location
                        intensity = darkest_map_normalized[y_int, x_int]
                        
                        # Get color from colormap based on intensity
                        color = darkest_colored[y_int, x_int]
                        color_tuple = tuple(int(c) for c in color)
                        
                        # Adaptive sizing based on confidence for stability
                        base_radius = 5
                        outline_width = 2
                        
                        # Slightly boost confidence for visual stability
                        visual_conf = min(1.0, conf * 1.2)
                        alpha_blend = int(200 * visual_conf + 55)
                        
                        # Draw keypoint with intensity-based color and adaptive size
                        cv2.circle(mask_frame, (x_int, y_int), base_radius, color_tuple, -1)
                        cv2.circle(mask_frame, (x_int, y_int), base_radius + 1, (255, 255, 255), outline_width)
                        
                        # Add subtle confidence glow (only for high confidence)
                        if conf > 0.3:
                            cv2.circle(mask_frame, (x_int, y_int), base_radius + 3, color_tuple, 1)
            
            # Draw skeleton connections (only between stable keypoints)
            skeleton_connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 7), (7, 8), (0, 5), (5, 6)]
            for start_idx, end_idx in skeleton_connections:
                if start_idx in valid_keypoints and end_idx in valid_keypoints:
                    pt1 = valid_keypoints[start_idx]
                    pt2 = valid_keypoints[end_idx]
                    
                    start_conf = keypoints[start_idx][2]
                    end_conf = keypoints[end_idx][2]
                    
                    # Only draw connections if both points are reasonably confident
                    if start_conf > 0.1 and end_conf > 0.1:
                        cv2.line(mask_frame, pt1, pt2, (180, 180, 180), 1)
        
        return mask_frame
    
    def draw_darkest_mask_no_keypoints(self, frame_gray, all_bee_data):
        """Same as darkest mask visualization but WITHOUT keypoints - only ROI triangles"""
        if frame_gray is None:
            return None
        
        h, w = frame_gray.shape
        mask_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Use factory for multi-directional darkness map
        darkest_map = MorphologyFactory.compute_darkness_map(frame_gray, mode='multi')
        
        # Adaptive threshold
        _, dark_binary = cv2.threshold(darkest_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply intensity-based coloring: darker pixels get more intense colors
        # Normalize darkness map to 0-255 range
        darkest_map_normalized = cv2.normalize(darkest_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply Viridis colormap with intensity
        darkest_colored = cv2.applyColorMap(darkest_map_normalized, cv2.COLORMAP_VIRIDIS)
        
        # Create mask for only dark line pixels
        dark_line_mask = dark_binary > 0
        
        # Initialize output with black background
        mask_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Copy only dark line pixels to output
        mask_frame[dark_line_mask] = darkest_colored[dark_line_mask]
        
        for bee_id, (keypoints, box) in all_bee_data.items():
            if box is None:
                continue
            
            # Skip drawing if thickness is 0
            if self.roi_thickness == 0:
                continue
            
            head_pos, center_end, left_end, right_end, *_roi = self.get_angled_roi_lines(keypoints, box, bee_id)
            
            left_triangle = np.array([head_pos, center_end, left_end], dtype=np.int32)
            right_triangle = np.array([head_pos, center_end, right_end], dtype=np.int32)
            
            # Draw triangle outlines with dynamic thickness from slider
            cv2.polylines(mask_frame, [left_triangle], True, (255, 200, 100), self.roi_thickness)
            cv2.polylines(mask_frame, [right_triangle], True, (100, 200, 255), self.roi_thickness)
        
        return mask_frame
    
    def draw_body_keypoints_roi_vectors(self, frame_gray, all_bee_data):
        """
        Visualization showing:
        1. Body keypoints only (head, thorax points, abdomen) - NO antenna keypoints
        2. ROI direction vector from head showing the antenna search direction
        """
        if frame_gray is None:
            return None
        
        h, w = frame_gray.shape
        
        # Start with grayscale frame converted to BGR
        display_frame = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        
        for bee_id, (keypoints, box) in all_bee_data.items():
            if keypoints is None or box is None:
                continue
            
            # Define colors for body keypoints (NOT antenna)
            BODY_KEYPOINT_COLORS = [
                (255, 0, 0),    # Blue - Head (kp 0)
                (0, 255, 0),    # Green - Thorax point 1 (kp 1)
                (255, 255, 0),  # Cyan - Thorax point 2 (kp 2)
                (255, 0, 255),  # Magenta - Thorax point 3 (kp 3)
                (128, 255, 255),# Light cyan - Abdomen (kp 4)
            ]
            
            # Draw ONLY body keypoints (indices 0-4: head, thorax x3, abdomen)
            for idx in range(5):  # Only first 5 keypoints (body)
                if idx < len(keypoints):
                    kp = keypoints[idx]
                    if len(kp) >= 2 and kp[2] > 0.05:  # Confidence check
                        x, y = int(kp[0]), int(kp[1])
                        if 0 <= x < w and 0 <= y < h:
                            color = BODY_KEYPOINT_COLORS[idx]
                            # Draw filled circle
                            cv2.circle(display_frame, (x, y), 5, color, -1)
                            # Draw white outline
                            cv2.circle(display_frame, (x, y), 6, (255, 255, 255), 1)
            
            # Draw body skeleton connections (NO antenna connections)
            body_connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Body centerline only
            ]
            for start_idx, end_idx in body_connections:
                if start_idx < len(keypoints) and end_idx < len(keypoints):
                    start_kp = keypoints[start_idx]
                    end_kp = keypoints[end_idx]
                    if len(start_kp) >= 2 and len(end_kp) >= 2:
                        if start_kp[2] > 0.05 and end_kp[2] > 0.05:
                            pt1 = (int(start_kp[0]), int(start_kp[1]))
                            pt2 = (int(end_kp[0]), int(end_kp[1]))
                            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                                cv2.line(display_frame, pt1, pt2, (200, 200, 200), 2)
            
            # Get ROI direction vector
            head_pos, center_end, left_end, right_end, *_roi = self.get_angled_roi_lines(keypoints, box, bee_id)
            
            # Draw ROI triangles
            left_triangle = np.array([head_pos, center_end, left_end], dtype=np.int32)
            right_triangle = np.array([head_pos, center_end, right_end], dtype=np.int32)
            
            # Draw triangle outlines with dynamic thickness from slider
            if self.roi_thickness > 0:
                cv2.polylines(display_frame, [left_triangle], True, (255, 200, 100), self.roi_thickness)
                cv2.polylines(display_frame, [right_triangle], True, (100, 200, 255), self.roi_thickness)
            
            # Draw the main ROI direction vector (from head to center_end)
            # This shows the antenna search direction
            cv2.arrowedLine(display_frame, head_pos, center_end, (0, 255, 255), 3, tipLength=0.15)  # Cyan arrow
            
            # Draw the left and right ROI boundary vectors (thinner)
            cv2.arrowedLine(display_frame, head_pos, left_end, (255, 200, 100), 2, tipLength=0.2)  # Orange for left
            cv2.arrowedLine(display_frame, head_pos, right_end, (100, 200, 255), 2, tipLength=0.2)  # Blue for right
        
        return display_frame
    
    def draw_roi_bbox_body(self, frame_gray, all_bee_data):
        """
        Visualization showing:
        1. ROI triangles (orange for left, cyan for right)
        2. Bounding box around bee
        3. Body keypoints only (no antenna keypoints)
        """
        if frame_gray is None:
            return None
        
        h, w = frame_gray.shape
        
        # Start with grayscale frame converted to BGR
        display_frame = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        
        for bee_id, (keypoints, box) in all_bee_data.items():
            if keypoints is None or box is None:
                continue
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Cyan bounding box
            
            # Draw extended bounding box (25% expansion)
            ex1, ey1, ex2, ey2 = self.get_expanded_roi_bounds(box, expand_percent=25)
            cv2.rectangle(display_frame, (ex1, ey1), (ex2, ey2), (255, 255, 0), 1)  # Yellow extended bbox (thinner)
            
            # Draw bee ID label
            label = f"ID:{bee_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(display_frame, 
                         (x1, y1 - label_size[1] - 4), 
                         (x1 + label_size[0], y1), 
                         (0, 255, 255), -1)
            cv2.putText(display_frame, label, (x1, y1 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Get ROI triangles
            head_pos, center_end, left_end, right_end, *_roi = self.get_angled_roi_lines(keypoints, box, bee_id)
            
            left_triangle = np.array([head_pos, center_end, left_end], dtype=np.int32)
            right_triangle = np.array([head_pos, center_end, right_end], dtype=np.int32)
            
            # Draw ROI triangles (always visible in this view, use slider thickness or default to 1)
            thickness = max(1, self.roi_thickness)  # Ensure at least 1 pixel thick
            cv2.polylines(display_frame, [left_triangle], True, (255, 200, 100), thickness)  # Orange
            cv2.polylines(display_frame, [right_triangle], True, (100, 200, 255), thickness)  # Cyan
            
            # Define colors for body keypoints
            BODY_KEYPOINT_COLORS = [
                (255, 0, 0),    # Blue - Head (kp 0)
                (0, 255, 0),    # Green - Thorax point 1 (kp 1)
                (255, 255, 0),  # Cyan - Thorax point 2 (kp 2)
                (255, 0, 255),  # Magenta - Thorax point 3 (kp 3)
                (128, 255, 255),# Light cyan - Abdomen (kp 4)
            ]
            
            # Draw ONLY body keypoints (indices 0-4)
            for idx in range(5):
                if idx < len(keypoints):
                    kp = keypoints[idx]
                    if len(kp) >= 2 and kp[2] > 0.05:
                        x, y = int(kp[0]), int(kp[1])
                        if 0 <= x < w and 0 <= y < h:
                            color = BODY_KEYPOINT_COLORS[idx]
                            cv2.circle(display_frame, (x, y), 5, color, -1)
                            cv2.circle(display_frame, (x, y), 6, (255, 255, 255), 1)
            
            # Draw body skeleton connections
            body_connections = [(0, 1), (1, 2), (2, 3), (3, 4)]
            for start_idx, end_idx in body_connections:
                if start_idx < len(keypoints) and end_idx < len(keypoints):
                    start_kp = keypoints[start_idx]
                    end_kp = keypoints[end_idx]
                    if len(start_kp) >= 2 and len(end_kp) >= 2:
                        if start_kp[2] > 0.05 and end_kp[2] > 0.05:
                            pt1 = (int(start_kp[0]), int(start_kp[1]))
                            pt2 = (int(end_kp[0]), int(end_kp[1]))
                            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                                cv2.line(display_frame, pt1, pt2, (200, 200, 200), 2)
        
        return display_frame
    
    def draw_grayscale_body_roi(self, frame_gray, all_bee_data):
        """
        Grayscale visualization with body keypoints and ROI angle lines.
        Shows clean grayscale image with minimal overlays - body keypoints and ROI triangles only.
        """
        if frame_gray is None:
            return None
        
        h, w = frame_gray.shape
        
        # Start with grayscale frame converted to BGR
        display_frame = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        
        for bee_id, (keypoints, box) in all_bee_data.items():
            if keypoints is None or box is None:
                continue
            
            # Define colors for body keypoints
            BODY_KEYPOINT_COLORS = [
                (255, 0, 0),    # Blue - Head (kp 0)
                (0, 255, 0),    # Green - Thorax point 1 (kp 1)
                (255, 255, 0),  # Cyan - Thorax point 2 (kp 2)
                (255, 0, 255),  # Magenta - Thorax point 3 (kp 3)
                (128, 255, 255),# Light cyan - Abdomen (kp 4)
            ]
            
            # Draw ONLY body keypoints (indices 0-4)
            for idx in range(5):
                if idx < len(keypoints):
                    kp = keypoints[idx]
                    if len(kp) >= 2 and kp[2] > 0.05:
                        x, y = int(kp[0]), int(kp[1])
                        if 0 <= x < w and 0 <= y < h:
                            color = BODY_KEYPOINT_COLORS[idx]
                            cv2.circle(display_frame, (x, y), 4, color, -1)
                            cv2.circle(display_frame, (x, y), 5, (255, 255, 255), 1)
            
            # Draw body skeleton connections
            body_connections = [(0, 1), (1, 2), (2, 3), (3, 4)]
            for start_idx, end_idx in body_connections:
                if start_idx < len(keypoints) and end_idx < len(keypoints):
                    start_kp = keypoints[start_idx]
                    end_kp = keypoints[end_idx]
                    if len(start_kp) >= 2 and len(end_kp) >= 2:
                        if start_kp[2] > 0.05 and end_kp[2] > 0.05:
                            pt1 = (int(start_kp[0]), int(start_kp[1]))
                            pt2 = (int(end_kp[0]), int(end_kp[1]))
                            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                                cv2.line(display_frame, pt1, pt2, (180, 180, 180), 1)
            
            # Get ROI lines
            head_pos, center_end, left_end, right_end, *_roi = self.get_angled_roi_lines(keypoints, box, bee_id)
            
            # Draw ROI triangles
            left_triangle = np.array([head_pos, center_end, left_end], dtype=np.int32)
            right_triangle = np.array([head_pos, center_end, right_end], dtype=np.int32)
            
            if self.roi_thickness > 0:
                cv2.polylines(display_frame, [left_triangle], True, (255, 200, 100), self.roi_thickness)
                cv2.polylines(display_frame, [right_triangle], True, (100, 200, 255), self.roi_thickness)
        
        return display_frame
    
    def draw_full_frame_darkest(self, frame_gray, all_bee_data):
        """IMPROVED: Full frame with only dark lines and intensity-based coloring - STABLE KEYPOINTS"""
        if frame_gray is None:
            return None
        
        h, w = frame_gray.shape
        
        # Use factory for multi-directional darkness map
        darkest_map = MorphologyFactory.compute_darkness_map(frame_gray, mode='multi')
        
        # Adaptive threshold
        _, dark_binary = cv2.threshold(darkest_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Normalize and apply colormap
        darkest_map_normalized = cv2.normalize(darkest_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        darkest_colored = cv2.applyColorMap(darkest_map_normalized, cv2.COLORMAP_VIRIDIS)
        
        # Create mask for dark line pixels only
        dark_line_mask = dark_binary > 0
        
        # Initialize output
        result_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Copy only dark pixels to output
        result_frame[dark_line_mask] = darkest_colored[dark_line_mask]
        
        for bee_id, (keypoints, box) in all_bee_data.items():
            hue = (bee_id * 40) % 180
            color_base = cv2.cvtColor(np.uint8([[[hue, 255, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
            color_tuple = (int(color_base[0]), int(color_base[1]), int(color_base[2]))
            
            if box is not None:
                # Expand bounding box by 25% for display
                x1, y1, x2, y2 = self.get_expanded_roi_bounds(box, expand_percent=25)
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color_tuple, 2)
            
            # ENHANCED: Stable keypoint plotting with reduced flickering
            valid_keypoints = {}
            for idx, (x, y, conf) in enumerate(keypoints):
                # Lower threshold for stability
                if conf > 0.05:
                    x_int, y_int = int(x), int(y)
                    
                    # Bounds check
                    if 0 <= y_int < h and 0 <= x_int < w:
                        valid_keypoints[idx] = (x_int, y_int)
                        
                        # Get intensity at keypoint
                        intensity = darkest_map_normalized[y_int, x_int]
                        
                        # Get color from colormap
                        kp_color = darkest_colored[y_int, x_int]
                        kp_color_tuple = tuple(int(c) for c in kp_color)
                        
                        # Adaptive sizing
                        base_radius = 4
                        
                        # Draw keypoint
                        cv2.circle(result_frame, (x_int, y_int), base_radius, kp_color_tuple, -1)
                        cv2.circle(result_frame, (x_int, y_int), base_radius + 1, (255, 255, 255), 2)
                        
                        # Subtle glow for high confidence points
                        if conf > 0.3:
                            cv2.circle(result_frame, (x_int, y_int), base_radius + 2, kp_color_tuple, 1)
            
            # Draw skeleton with stability check
            skeleton_connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 7), (7, 8), (0, 5), (5, 6)]
            for start_idx, end_idx in skeleton_connections:
                if start_idx in valid_keypoints and end_idx in valid_keypoints:
                    pt1 = valid_keypoints[start_idx]
                    pt2 = valid_keypoints[end_idx]
                    
                    start_conf = keypoints[start_idx][2]
                    end_conf = keypoints[end_idx][2]
                    
                    # Only draw if both reasonably confident
                    if start_conf > 0.1 and end_conf > 0.1:
                        cv2.line(result_frame, pt1, pt2, color_tuple, 2)
            # Draw body segments with perpendicular lines
            if box is not None:
                result_frame = self.draw_body_segments(result_frame, keypoints, box, bee_id)            
            # Draw bee label
            if box is not None:
                x1, y1 = int(box[0]), int(box[1])
                class_name = "Worker"
                if bee_id in self.bee_class:
                    class_name = self.bee_class[bee_id]
                label_text = f"{class_name}; {bee_id}"
                cv2.putText(result_frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_tuple, 2)
        
        return result_frame

    def place_antenna_keypoints_on_darkest_lines(self, frame_gray, keypoints, box, bee_id, frame_id):
        """
        IMPROVED: Aggressively place antenna keypoints (5,6,7,8) at the tips of darkest thin lines.
        Now with frame dimension safety checks.
        """
        if frame_gray is None or box is None:
            return keypoints
        
        h, w = frame_gray.shape
        
        # NEW: Validate box is within frame bounds
        box = np.clip(box, [0, 0, 0, 0], [w, h, w, h])
        
        placed = keypoints.copy()
        
        # Expand bounding box by 25% WITH FRAME BOUNDS
        expanded_box = self.expand_bbox_by_percentage(box, expand_percent=25, frame_width=w, frame_height=h)
        x1, y1, x2, y2 = (int(expanded_box[0]), int(expanded_box[1]), 
                          int(expanded_box[2]), int(expanded_box[3]))
        
        # NEW: Ensure bounds are valid
        x1 = max(0, min(x1, w-1))
        x2 = max(x1+1, min(x2, w))
        y1 = max(0, min(y1, h-1))
        y2 = max(y1+1, min(y2, h))
        
        kp0_x, kp0_y = int(keypoints[0][0]), int(keypoints[0][1])
        
        # OPTIMIZATION: Use precomputed maps instead of recomputing per bee
        maps = self.precompute_antenna_detection_maps(frame_gray, frame_id)
        
        if maps is None:
            return placed
        
        dark_binary = maps['binary']  # Already computed once for entire frame
        
        # NEW: Validate extracted ROI
        roi = dark_binary[y1:y2, x1:x2]
        if roi.size == 0:
            return placed
        
        # Find connected components - OPTIMIZED with caching
        num_labels, labels, stats, centroids = self.get_frame_connected_components(roi, frame_id)
        
        if num_labels <= 1:
            return placed
        
        # Find all valid antenna-like structures
        antenna_candidates = []
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Filters for thin, elongated structures
            aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
            
            # USE THRESHOLD SLIDERS instead of hardcoded values
            min_aspect = self.detection_thresholds.get('min_aspect_ratio', 1.5)
            min_area = self.detection_thresholds.get('min_area', 3)
            max_area = self.detection_thresholds.get('max_area', 1200)
            
            if aspect_ratio > min_aspect and min_area < area < max_area:
                component_mask = (labels == i).astype(np.uint8)
                points = np.column_stack(np.where(component_mask > 0))
                
                if len(points) < 3:
                    continue
                
                # Get convex hull for reliable endpoint detection
                hull_points = cv2.convexHull(points.astype(np.float32))
                hull_points = hull_points.reshape(-1, 2).astype(np.int32)
                
                # Find the two endpoints - OPTIMIZED with vectorization
                max_dist = 0
                endpoint1, endpoint2 = None, None
                
                if len(hull_points) > 1:
                    # Vectorized distance computation (much faster than nested loops)
                    dists = squareform(pdist(hull_points))
                    max_flat_idx = np.argmax(dists)
                    endpoint1_idx, endpoint2_idx = np.unravel_index(max_flat_idx, dists.shape)
                    endpoint1 = hull_points[endpoint1_idx]
                    endpoint2 = hull_points[endpoint2_idx]
                    max_dist = dists[endpoint1_idx, endpoint2_idx]
                
                if endpoint1 is not None and max_dist > 15:
                    # Convert to frame coordinates
                    ep1_frame = (endpoint1[1] + x1, endpoint1[0] + y1)
                    ep2_frame = (endpoint2[1] + x1, endpoint2[0] + y1)
                    
                    # Distance from head to endpoints
                    dist1_head = np.sqrt((ep1_frame[0] - kp0_x)**2 + (ep1_frame[1] - kp0_y)**2)
                    dist2_head = np.sqrt((ep2_frame[0] - kp0_x)**2 + (ep2_frame[1] - kp0_y)**2)
                    
                    # The tip is the endpoint farthest from head
                    tip = ep1_frame if dist1_head > dist2_head else ep2_frame
                    tip_dist = max(dist1_head, dist2_head)
                    
                    score = max_dist * aspect_ratio / (tip_dist + 1)
                    
                    antenna_candidates.append({
                        'tip': tip,
                        'length': max_dist,
                        'aspect_ratio': aspect_ratio,
                        'dist_from_head': tip_dist,
                        'score': score,
                        'area': area
                    })
        
        if not antenna_candidates:
            return placed
        
        # Sort by composite score
        antenna_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Get the two best antennas
        left_antenna = None
        right_antenna = None
        
        if len(antenna_candidates) >= 1:
            left_antenna = antenna_candidates[0]
        
        if len(antenna_candidates) >= 2:
            right_antenna = antenna_candidates[1]
        
        # If only one antenna found, check if it's on left or right of head
        if len(antenna_candidates) == 1:
            tip_x, tip_y = antenna_candidates[0]['tip']
            if tip_x < kp0_x:
                left_antenna = antenna_candidates[0]
            else:
                right_antenna = antenna_candidates[0]
        
        # Get ROI triangles for validation (antenna must be outside them)
        head_pos, center_end, left_end, right_end, *_roi = self.get_angled_roi_lines(keypoints, box, bee_id)
        left_triangle = [head_pos, center_end, left_end]
        right_triangle = [head_pos, center_end, right_end]
        
        # Validate antenna positions are outside triangles
        if left_antenna:
            tip_x, tip_y = left_antenna['tip']
            if not self.point_outside_both_triangles((tip_x, tip_y), left_triangle, right_triangle):
                # Antenna is inside triangle, discard it
                left_antenna = None
        
        if right_antenna:
            tip_x, tip_y = right_antenna['tip']
            if not self.point_outside_both_triangles((tip_x, tip_y), left_triangle, right_triangle):
                # Antenna is inside triangle, discard it
                right_antenna = None
        
        # Place keypoints at antenna tips
        if left_antenna:
            tip_x, tip_y = left_antenna['tip']
            # Place both left antenna keypoints (5, 6) at the same tip
            placed[5][0] = float(tip_x)
            placed[5][1] = float(tip_y)
            placed[5][2] = 0.95
            
            placed[6][0] = float(tip_x)
            placed[6][1] = float(tip_y)
            placed[6][2] = 0.90
        
        if right_antenna:
            tip_x, tip_y = right_antenna['tip']
            # Place both right antenna keypoints (7, 8) at the same tip
            placed[7][0] = float(tip_x)
            placed[7][1] = float(tip_y)
            placed[7][2] = 0.95
            
            placed[8][0] = float(tip_x)
            placed[8][1] = float(tip_y)
            placed[8][2] = 0.90
        
        return placed


    def _refine_antenna_position_in_radius(self, frame_gray, darkest_map, last_position, search_radius):
        """
        Search around last antenna position for darkest pixel within search radius.
        Returns refined position or None if not found.
        """
        seed_x, seed_y = int(last_position[0]), int(last_position[1])
        conf = last_position[2]
        
        # Bounds check
        h, w = frame_gray.shape
        seed_x = max(0, min(w - 1, seed_x))
        seed_y = max(0, min(h - 1, seed_y))
        
        # Define search region
        y_min, y_max, x_min, x_max = AntennaGeometry.get_safe_roi(seed_x, seed_y, search_radius, (h, w))
        
        # Extract ROI
        roi = darkest_map[y_min:y_max, x_min:x_max]
        
        if roi.size == 0:
            return None
        
        # Find darkest pixel in ROI
        max_pos = np.unravel_index(np.argmax(roi), roi.shape)
        max_darkness = roi[max_pos]
        
        # Only accept if darkness is significant (not just noise)
        if max_darkness < 20:
            return None
        
        refined_x = max_pos[1] + x_min
        refined_y = max_pos[0] + y_min
        
        # Slightly decay confidence over time (optional, maintains uncertainty)
        new_conf = min(0.95, conf * 0.98 + 0.02)
        
        return (float(refined_x), float(refined_y), new_conf)    
    def refine_antenna_keypoints_iterative(self, frame_gray, keypoints, box, bee_id, search_depth=3):
        """
        BONUS: Iteratively refine antenna keypoints by searching outward from initial placement.
        Handles cases where initial detection is slightly off.
        """
        if frame_gray is None or box is None:
            return keypoints
        
        refined = keypoints.copy()
        
        # Compute darkness map
        darkest_map = 255 - frame_gray
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        darkest_map = cv2.morphologyEx(darkest_map, cv2.MORPH_TOPHAT, kernel)
        darkest_map = cv2.GaussianBlur(darkest_map, (3, 3), 0.5)
        
        antenna_indices = [5, 6, 7, 8]
        
        for idx in antenna_indices:
            if refined[idx][2] < 0.5:
                continue
            
            seed_x, seed_y = int(refined[idx][0]), int(refined[idx][1])
            
            # Ensure within bounds
            seed_y = max(0, min(frame_gray.shape[0] - 1, seed_y))
            seed_x = max(0, min(frame_gray.shape[1] - 1, seed_x))
            
            # Search in expanding squares
            best_point = (seed_x, seed_y)
            best_darkness = darkest_map[seed_y, seed_x]
            
            for depth in range(1, search_depth + 1):
                search_radius = depth * 3
                
                y_min, y_max, x_min, x_max = AntennaGeometry.get_safe_roi(
                    seed_x, seed_y, search_radius, frame_gray.shape
                )
                
                roi = darkest_map[y_min:y_max, x_min:x_max]
                
                if roi.size == 0:
                    continue
                
                max_pos = np.unravel_index(np.argmax(roi), roi.shape)
                max_darkness = roi[max_pos]
                
                if max_darkness > best_darkness:
                    best_darkness = max_darkness
                    best_point = (max_pos[1] + x_min, max_pos[0] + y_min)
            
            refined[idx][0] = float(best_point[0])
            refined[idx][1] = float(best_point[1])
            refined[idx][2] = min(1.0, refined[idx][2] + 0.1)
        
        return refined    
    def place_keypoints_with_skeleton(self, antenna_blob, head_position, roi_triangle=None):
        """Place antenna keypoints using skeleton tracing with gap handling
        
        Args:
            antenna_blob: Binary image of antenna structure
            head_position: (x, y) tuple of head keypoint
            roi_triangle: Optional [(x1,y1), (x2,y2), (x3,y3)] to mask antenna blob to ROI
                        If None, assumes blob is ALREADY triangle-constrained
        
        Returns:
            tuple: (joint_position, tip_position) or (None, None) if failed
        """
        
        # Step 0: Apply ROI mask if provided
        # NEW: If roi_triangle is None, blob is already constrained - skip masking
        if roi_triangle is not None:
            # Create ROI mask
            h, w = antenna_blob.shape
            triangle_pts = np.array(roi_triangle, dtype=np.int32)
            # Guard: skip masking if triangle is degenerate (all points outside frame)
            t_xmin, t_xmax = int(triangle_pts[:,0].min()), int(triangle_pts[:,0].max())
            t_ymin, t_ymax = int(triangle_pts[:,1].min()), int(triangle_pts[:,1].max())
            if t_xmax > t_xmin + 1 and t_ymax > t_ymin + 1 and h > 0 and w > 0:
                roi_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(roi_mask, triangle_pts, 255)
                antenna_blob = cv2.bitwise_and(antenna_blob, roi_mask)
        
        # Step 1: Apply morphological closing to prevent gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        antenna_closed = cv2.morphologyEx(antenna_blob, cv2.MORPH_CLOSE, kernel)
        
        # Step 2: Extract skeleton
        skeleton = SkeletonProcessor.morphological_skeleton(antenna_closed)
        
        if skeleton is None or cv2.countNonZero(skeleton) < 10:
            return None, None
        
        # Step 3: Find endpoints
        endpoints = SkeletonProcessor.find_skeleton_endpoints(skeleton)
        
        if len(endpoints) < 2:
            # Failed to find endpoints
            return None, None
        
        # Step 4: Determine base and tip
        # Base = closest to head, Tip = farthest from head
        distances = [np.sqrt((ep[0] - head_position[0])**2 + (ep[1] - head_position[1])**2) 
                    for ep in endpoints]
        
        base_idx = np.argmin(distances)
        tip_idx = np.argmax(distances)
        
        base = endpoints[base_idx]
        tip = endpoints[tip_idx]
        
        # Step 5: Trace path or create virtual line
        if len(endpoints) == 2:
            # Good: clean skeleton, trace path
            skeleton_path = SkeletonProcessor.trace_skeleton_path(skeleton, base, tip)
        else:
            # Broken: multiple gaps, use straight line
            skeleton_path = SkeletonProcessor.interpolate_line(base, tip, 100)
        
        if len(skeleton_path) < 3:
            return None, None
        
        # Step 6: Place keypoints
        # Joint at 30% from base
        joint_idx = int(0.30 * len(skeleton_path))
        joint_position = skeleton_path[joint_idx]
        
        # Tip at end
        tip_position = skeleton_path[-1]
        
        return joint_position, tip_position
    def calculate_event_quality_score(self, event):
        """Calculate quality score 0-100 based on all metrics"""
        score = 50
        
        avg_straightness = (event['straightness_a'] + event['straightness_b']) / 2
        score += (avg_straightness - 0.85) * 100
        
        angle_diff = event['angle_diff']
        if 175 <= angle_diff <= 185:
            score += 15
        elif 170 <= angle_diff <= 190:
            score += 10
        elif 150 <= angle_diff <= 210:
            score += 5
        
        rr_dom = event['rr_dominance']
        score += rr_dom * 20
        
        if event['duration_seconds'] >= 10.0:
            score += 10
        
        return min(100, max(0, score))
    
    def get_aggregate_interpretation(self):
        """Wrapper that delegates to TrophallaxisDetector's new dynamic method"""
        return self.trophallaxis_detector.get_aggregate_colony_interpretation(fps=30)
    
    def get_colony_health_status(self):
        """Determine overall colony health status - SIMPLIFIED"""
        
        if len(self.trophallaxis_detector.completed_events) == 0:
            return "🟢 HEALTHY", "MONITORING"
        
        confirmed = [e for e in self.trophallaxis_detector.completed_events 
                    if e.get('confirmed', False)]
        
        if not confirmed:
            return "🟢 HEALTHY", "MONITORING"
        
        # Get metrics from detector's method
        total_contacts = sum(sum(e['antenna_contacts'].values()) for e in confirmed)
        total_rr = sum(e['antenna_contacts']['RR'] for e in confirmed)
        rr_ratio = total_rr / total_contacts if total_contacts > 0 else 0.0
        
        avg_quality = np.mean([self.trophallaxis_detector.calculate_event_quality_score(e) 
                              for e in confirmed])
        continuous_events = sum(1 for e in confirmed if e['frames_active'] > 290)
        continuous_ratio = continuous_events / len(confirmed) if confirmed else 0.0
        
        # Status determination
        if rr_ratio >= 0.85 and avg_quality >= 92 and continuous_ratio >= 0.90:
            return "🟢 HEALTHY", "EXCELLENT"
        elif rr_ratio >= 0.75 and avg_quality >= 85 and continuous_ratio >= 0.80:
            return "🟢 HEALTHY", "GOOD"
        elif rr_ratio >= 0.60 and avg_quality >= 75:
            return "🟡 CAUTION", "FAIR"
        else:
            return "🔴 CRITICAL", "POOR"
    def _calculate_antenna_dominance_metrics(self):
        """Calculate aggregate antenna dominance metrics from all completed events"""
        if not self.trophallaxis_detector.completed_events:
            return {
                'rr_ratio': 0.0,
                'll_ratio': 0.0,
                'cross_ratio': 0.0,
                'lateralization_index': 0.0,
                'symmetry_ratio': 1.0,
                'total_contacts': 0,
                'classification': 'NO_DATA'
            }
        
        total_rr = 0
        total_rl = 0
        total_lr = 0
        total_ll = 0
        
        for event in self.trophallaxis_detector.completed_events:
            if 'antenna_metrics' in event:
                metrics = event['antenna_metrics']
                total_rr += metrics['rr_count']
                total_rl += metrics['rl_count']
                total_lr += metrics['lr_count']
                total_ll += metrics['ll_count']
        
        total = total_rr + total_rl + total_lr + total_ll
        
        if total == 0:
            return {
                'rr_ratio': 0.0,
                'll_ratio': 0.0,
                'cross_ratio': 0.0,
                'lateralization_index': 0.0,
                'symmetry_ratio': 1.0,
                'total_contacts': 0,
                'classification': 'NO_DATA'
            }
        
        rr_ratio = total_rr / total
        ll_ratio = total_ll / total
        cross_ratio = (total_rl + total_lr) / total
        li = (total_rr - total_ll) / total if total > 0 else 0.0
        sym = total_rl / total_lr if total_lr > 0 else (total_rl if total_rl > 0 else 1.0)
        
        return {
            'rr_ratio': rr_ratio,
            'll_ratio': ll_ratio,
            'cross_ratio': cross_ratio,
            'lateralization_index': li,
            'symmetry_ratio': sym,
            'rr_count': total_rr,
            'rl_count': total_rl,
            'lr_count': total_lr,
            'll_count': total_ll,
            'total_contacts': total,
            'classification': self.trophallaxis_detector.interpretation_engine.classify_rr_ratio(rr_ratio)['classification']
        }
    def get_expanded_roi_bounds(self, box, expand_percent=25):
        """
        Convenience wrapper for expanded ROI extraction.
        
        Returns:
            tuple: (x1, y1, x2, y2) integers
        """
        if box is None:
            return None
        
        expanded = self.expand_bbox_by_percentage(box, expand_percent)
        return (int(expanded[0]), int(expanded[1]), int(expanded[2]), int(expanded[3]))
    def expand_bbox_by_percentage(self, box, expand_percent=25, frame_width=None, frame_height=None):
        """
        Expand bounding box by given percentage from center.
        Now with frame dimension bounds checking.
        
        Args:
            box: [x1, y1, x2, y2] format
            expand_percent: percentage to expand (default 25%)
            frame_width: frame width for bounds checking (optional)
            frame_height: frame height for bounds checking (optional)
        
        Returns:
            expanded_box: [x1_new, y1_new, x2_new, y2_new]
        """
        if box is None:
            return None
        
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        
        # NEW: Validate box coordinates
        if x1 >= x2 or y1 >= y2:
            print(f"[WARNING] Invalid box coordinates: {box}")
            return box
        
        # Calculate center and dimensions
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        # Expand dimensions
        expand_factor = 1.0 + (expand_percent / 100.0)
        new_width = width * expand_factor
        new_height = height * expand_factor
        
        # Calculate new coordinates from center
        new_x1 = center_x - new_width / 2.0
        new_y1 = center_y - new_height / 2.0
        new_x2 = center_x + new_width / 2.0
        new_y2 = center_y + new_height / 2.0
        
        # NEW: Apply frame bounds if provided
        if frame_width is not None and frame_height is not None:
            new_x1 = max(0, min(new_x1, frame_width))
            new_y1 = max(0, min(new_y1, frame_height))
            new_x2 = max(0, min(new_x2, frame_width))
            new_y2 = max(0, min(new_y2, frame_height))
        
        return np.array([new_x1, new_y1, new_x2, new_y2])
    def draw_body_segments(self, frame, keypoints, box, bee_id):
        """Draw segmented body rectangle with perpendicular lines at keypoints"""
        if frame is None or box is None:
            return frame
        
        # Get head and abdomen positions
        head = np.array([keypoints[0][0], keypoints[0][1]])
        abdomen_tip = np.array([keypoints[4][0], keypoints[4][1]])
        
        # Calculate body vector (orientation)
        body_vector = abdomen_tip - head
        body_length = np.linalg.norm(body_vector)
        
        if body_length < 5:  # Degenerate case
            return frame
        
        # Normalize body vector
        body_unit = body_vector / body_length
        
        # Calculate perpendicular vector (rotated 90°)
        perp_unit = np.array([-body_unit[1], body_unit[0]])
        
        # Rectangle dimensions
        bbox_width = box[2] - box[0]
        rect_width = bbox_width * 0.4  # 60% of bbox
        rect_height = body_length * 1.4
        
        # Center of rectangle (middle of bbox)
        rect_center = np.array([
            (box[0] + box[2]) / 2.0,
            (box[1] + box[3]) / 2.0
        ])
        
        # Calculate rectangle corners
        half_width = rect_width / 2.0
        half_height = rect_height / 2.0
        
        corners = np.array([
            rect_center - half_width * perp_unit - half_height * body_unit,
            rect_center + half_width * perp_unit - half_height * body_unit,
            rect_center + half_width * perp_unit + half_height * body_unit,
            rect_center - half_width * perp_unit + half_height * body_unit
        ], dtype=np.int32)
        
        # Get bee color
        hue = (bee_id * 40) % 180
        color_base = cv2.cvtColor(np.uint8([[[hue, 255, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
        color = (int(color_base[0]), int(color_base[1]), int(color_base[2]))
        
        # Draw body rectangle
        cv2.polylines(frame, [corners], True, color, 2)
        
        # Draw perpendicular lines at each keypoint (1, 2, 3)
        segment_keypoints = [1, 2, 3]
        segment_names = ['Thorax Top', 'Thorax Bot', 'Abdomen Start']
        
        for kp_idx, kp_name in zip(segment_keypoints, segment_names):
            kp = np.array([keypoints[kp_idx][0], keypoints[kp_idx][1]])
            
            # Project keypoint onto body line to find perpendicular points
            to_kp = kp - head
            proj_len = np.dot(to_kp, body_unit)
            proj_point = head + proj_len * body_unit
            
            # Perpendicular extent (half rectangle width)
            left_point = proj_point - half_width * perp_unit
            right_point = proj_point + half_width * perp_unit
            
            # Draw perpendicular line
            cv2.line(frame, tuple(left_point.astype(int)), 
                    tuple(right_point.astype(int)), color, 2)
            
            # Draw small circle at keypoint
            cv2.circle(frame, tuple(kp.astype(int)), 4, color, -1)
            cv2.circle(frame, tuple(kp.astype(int)), 5, (255, 255, 255), 1)
        
        # Antenna segment visualization disabled
        
        return frame
    
    def _draw_antenna_segments(self, frame, keypoints, bee_id):
        """Draw antenna structure - visualization disabled"""
        # Antenna segment visualization removed to reduce clutter
        pass

    def draw_keypoints_with_labels(self, image, all_bee_data):
        """Draw keypoints with skeleton and keypoint numbers - antenna keypoints labeled"""
        overlay = image.copy()
        
        for bee_id, (keypoints, box) in all_bee_data.items():
            hue = (bee_id * 40) % 180
            color_base = cv2.cvtColor(np.uint8([[[hue, 255, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
            color_tuple = (int(color_base[0]), int(color_base[1]), int(color_base[2]))
            
            if box is not None:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color_tuple, 2)
            
            valid_keypoints = {}
            for idx, (x, y, conf) in enumerate(keypoints):
                if conf > 0.1:
                    valid_keypoints[idx] = (int(x), int(y))
                    cv2.circle(overlay, (int(x), int(y)), 3, color_tuple, -1)
                    cv2.circle(overlay, (int(x), int(y)), 4, (255, 255, 255), 1)
                    
                    # ADD KEYPOINT NUMBERS FOR ANTENNA KEYPOINTS (5, 6, 7, 8)
                    if idx in [5, 6, 7, 8]:
                        text = str(idx)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness = 2
                        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                        
                        # Position text near the keypoint
                        text_x = int(x) + 8
                        text_y = int(y) - 8
                        
                        # Add background rectangle for readability
                        cv2.rectangle(overlay, 
                                    (text_x - 2, text_y - text_size[1] - 2),
                                    (text_x + text_size[0] + 2, text_y + 2),
                                    color_tuple, -1)
                        
                        cv2.putText(overlay, text, (text_x, text_y), 
                                  font, font_scale, (255, 255, 255), thickness)
            
            skeleton_connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 7), (7, 8), (0, 5), (5, 6)]
            for start_idx, end_idx in skeleton_connections:
                if start_idx in valid_keypoints and end_idx in valid_keypoints:
                    pt1 = valid_keypoints[start_idx]
                    pt2 = valid_keypoints[end_idx]
                    cv2.line(overlay, pt1, pt2, color_tuple, 2)
            
            if len(keypoints) > 0 and keypoints[0][2] > 0.1:
                x, y = keypoints[0][:2]
                class_name = "Worker"
                if bee_id in self.bee_class:
                    class_name = self.bee_class[bee_id]
                label_text = f"{class_name}; {bee_id}"
                cv2.putText(overlay, label_text, (int(x) - 20, int(y) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_tuple, 2)
        
        alpha = 0.8
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    def get_tracker_config(self):
        """Get tracker configuration, fallback to default if custom not found"""
        import os
        custom_config = "bytetrack_robust.yaml"
        
        if os.path.exists(custom_config):
            print(f"[TRACKER] Using custom config: {custom_config}")
            return custom_config
        else:
            print(f"[TRACKER] Custom config not found, using default bytetrack.yaml")
            return "bytetrack.yaml"    
    def run(self):
            try:
                self.progress.emit(f"Loading model... (Device: {TORCH_DEVICE})")
                print(f"\n[INFERENCE WORKER] Starting inference on device: {TORCH_DEVICE}")
                print(f"[INFERENCE WORKER] CUDA Available: {self.use_cuda}")
                
                model = YOLO(self.model_path)
                
                if self.use_cuda:
                    print("[INFERENCE WORKER] Moving model to CUDA device...")
                    model.to(TORCH_DEVICE)
                
                if hasattr(model, 'names') and model.names:
                    class_names = [model.names[i] for i in sorted(model.names.keys())]
                else:
                    class_names = ["Worker", "Drone", "Yellowjacket", "Resting"]
                
                self.progress.emit(f"Starting inference on {TORCH_DEVICE}...")
                print(f"[INFERENCE WORKER] Model loaded. Starting frame processing...")
                
                # ── Source selection: image folder OR video file ──────────────────
                if self.input_mode == 'images':
                    cap = ImageSequenceCapture(self.video_path)
                    source_label = "IMAGE FOLDER (60 FPS virtual)"
                else:
                    cap = cv2.VideoCapture(self.video_path)
                    source_label = "VIDEO FILE"

                # Check if source opened successfully
                if not cap.isOpened():
                    raise Exception(
                        f"Failed to open {source_label}: {self.video_path}"
                    )

                # Get source properties
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # For image sequences the virtual FPS is always 60;
                # for real videos fall back to 30 if the header reports 0.
                if self.input_mode == 'images':
                    self.input_fps = ImageSequenceCapture.IMAGE_SEQUENCE_FPS
                else:
                    self.input_fps = fps if fps > 0 else 30.0

                print(f"[{source_label}] Total frames: {total_frames}, "
                      f"FPS: {self.input_fps}")
                
                frame_id = 0
                consecutive_read_failures = 0
                max_read_failures = 30
                
                while self._is_running:
                    # UPDATE KERNELS BASED ON CURRENT THRESHOLD VALUES (real-time adjustment)
                    current_kernel_size = self.detection_thresholds.get('kernel_size', 15)
                    if current_kernel_size != self.research_processor.kernel_size:
                        self.research_processor = MorphologicalPipelineProcessor(kernel_size=current_kernel_size)
                        # Antenna pipeline doesn't depend on kernel size, so no need to re-create
                    
                    ret, frame = cap.read()
                    
                    # Robust frame reading
                    if not ret:
                        consecutive_read_failures += 1
                        print(f"[WARNING] Frame read failed (attempt {consecutive_read_failures}/{max_read_failures})")
                        
                        if consecutive_read_failures >= max_read_failures:
                            print("[INFO] End of video or too many read failures")
                            break
                        
                        continue
                    
                    consecutive_read_failures = 0
                    
                    # NEW: Validate frame dimensions match expected video format
                    if self.last_frame_shape is None:
                        self.last_frame_shape = frame.shape
                        print(f"[VIDEO] Frame dimensions: {frame.shape}")
                    elif frame.shape != self.last_frame_shape:
                        print(f"[WARNING] Dimension mismatch! Expected {self.last_frame_shape}, got {frame.shape}")
                        self.frame_dimension_mismatch_count += 1
                        
                        if self.frame_dimension_mismatch_count >= self.max_dimension_mismatches:
                            print(f"[ERROR] Video has inconsistent frame dimensions - cannot continue")
                            self._clear_dimension_sensitive_cache()
                            break
                        
                        # Skip this malformed frame
                        frame_id += 1
                        continue
                    
                    # Reset mismatch counter on valid frame
                    self.frame_dimension_mismatch_count = 0
                    
                    # Frame skipping for performance
                    self.frame_skip_counter += 1
                    if self.frame_skip_counter % self.process_every_n_frames != 0:
                        frame_id += 1
                        continue
                    
                    # Validate frame
                    if frame is None or frame.size == 0:
                        print(f"[WARNING] Invalid frame at {frame_id}")
                        frame_id += 1
                        continue
                    
                    try:
                        # ── Selective Dark-Pixel Darkening ────────────────────────────
                        # Applied BEFORE cvtColor / YOLO / any detection method.
                        # Only dark pixels are attenuated; bright pixels are untouched.
                        # Formula per pixel p (0-255):
                        #   out = p × (1 − darkness × (1 − p/255))
                        # When p≈0  → multiplied by (1−darkness)  [maximum darkening]
                        # When p≈255 → multiplied by ≈1.0          [no change]
                        if self.pixel_darkness > 0.0:
                            lut = np.array([
                                int(round(p * (1.0 - self.pixel_darkness * (1.0 - p / 255.0))))
                                for p in range(256)
                            ], dtype=np.uint8)
                            frame = cv2.LUT(frame, lut)
                        # ──────────────────────────────────────────────────────────────
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    except Exception as e:
                        print(f"[ERROR] Failed to convert frame {frame_id}: {e}")
                        frame_id += 1
                        continue
                    
                    display_frame = frame.copy()
                    all_bee_data = {}
                    
                    # Robust tracking call with error handling
                    try:
                        results = model.track(
                            frame,
                            stream=False,
                            persist=True,
                            tracker=self.get_tracker_config(),
                            conf=0.25,
                            iou=0.5,
                            verbose=False,
                            device=0 if self.use_cuda else "cpu",
                            half=False,
                            max_det=self.max_bees_per_frame
                        )
                    except Exception as e:
                        print(f"[ERROR] Tracking failed at frame {frame_id}: {e}")
                        frame_id += 1
                        continue
                    
                    # Use safe detection processing
                    all_kpts, track_ids, boxes, class_indices = self.safe_process_detections(results)
                    
                    if all_kpts is not None and track_ids is not None:
                        # Track all seen IDs
                        for tid in track_ids:
                            self.track_id_history.add(tid)
                        
                        # Process each detected bee
                        for bee_idx, bee_id in enumerate(track_ids):
                            if bee_idx >= len(all_kpts):
                                break
                            
                            keypoints = all_kpts[bee_idx]
                            
                            if bee_idx < len(class_indices):
                                cls_idx = class_indices[bee_idx]
                                if cls_idx < len(class_names):
                                    self.bee_class[bee_id] = class_names[cls_idx]
                            
                            keypoints = self.smooth_keypoints(bee_id, keypoints)
                            
                            box = boxes[bee_idx] if len(boxes) > bee_idx else None
                            
                            if box is not None:
                                # UNIFIED antenna refinement (intelligent strategy selection)
                                if bee_id not in self.antenna_tracked_positions:
                                    # First frame: Maximum search depth
                                    keypoints = self.refine_antenna_keypoints_unified(frame_gray, keypoints, box, bee_id, frame_id)
                                    keypoints = self.refine_antenna_keypoints_iterative(frame_gray, keypoints, box, bee_id, search_depth=12)
                                    
                                    self.antenna_tracked_positions[bee_id] = {
                                        'left': (keypoints[5][0], keypoints[5][1], keypoints[5][2]),
                                        'right': (keypoints[7][0], keypoints[7][1], keypoints[7][2])
                                    }
                                    self.antenna_lock_frames[bee_id] = 0
                                else:
                                    # Subsequent frames: Unified strategy (auto switches between fast & thorough)
                                    keypoints = self.refine_antenna_keypoints_unified(frame_gray, keypoints, box, bee_id, frame_id)
                                    
                                    # AUTO RE-DETECT if confidence too low
                                    left_conf = keypoints[5][2]
                                    right_conf = keypoints[7][2]
                                    
                                    if left_conf < 0.4 or right_conf < 0.4:
                                        # Antenna lost - force thorough re-detection
                                        keypoints = self.place_antenna_keypoints_on_darkest_lines(frame_gray, keypoints, box, bee_id, frame_id)
                                        keypoints = self.refine_antenna_keypoints_iterative(frame_gray, keypoints, box, bee_id, search_depth=10)
                            
                            # VALIDATE: Reject antenna tips outside ROI triangles
                            keypoints = self.validate_tip_keypoints(bee_id, keypoints, box)
                            
                            # MOTION CONFIRMATION: Boost antenna confidence if body is
                            # stationary but antennae are independently moving
                            motion_boost = self.update_antenna_motion_tracking(bee_id, keypoints, box)
                            if motion_boost > 0:
                                for ki in [5, 6, 7, 8]:
                                    if keypoints[ki][2] > 0.1:
                                        keypoints[ki] = (keypoints[ki][0], keypoints[ki][1],
                                                         min(1.0, keypoints[ki][2] + motion_boost))
                            
                            all_bee_data[bee_id] = (keypoints, box)
                        
                        # Clean up tracking data for lost bees
                        self.cleanup_lost_bees(track_ids)
                    
                    # ── BFS pipeline FIRST (every frame): refines kp[5..8] in
                    # all_bee_data so every consumer below uses correct positions ──
                    if self.frames_processed % 2 == 0:
                        research_data = self.research_processor.compute_all_stages(
                            frame_gray, frame_id, all_bee_data, self.antenna_pipeline, worker_instance=self
                        )
                        if research_data is not None:
                            self.research_visualization.emit(research_data)

                    # Now draw the main display with BFS-corrected keypoints
                    display_frame = self.draw_keypoints_with_labels(display_frame, all_bee_data)

                    # Tracking also uses BFS-corrected positions
                    # NEW: Track antenna contacts independently (EVERY frame)
                    self.antenna_tracker.track_antenna_contacts(all_bee_data)
                    
                    # NEW: Track 4-region anatomical contacts (EVERY frame)
                    self.region_tracker.track_contacts(all_bee_data, frame_id)
                    
                    # Trophallaxis detection (separate system)
                    self.trophallaxis_detector.detect_trophallaxis(all_bee_data, frame_id, fps=self.input_fps)
                    # Update optical flow tracker with current frame
                    self.optical_flow_tracker.update_prev_frame(frame_gray)
                    print(f"[TROPHALLAXIS] Frame {frame_id}: Pending={len(self.trophallaxis_detector.pending_events)}, Completed={len(self.trophallaxis_detector.completed_events)}")
                    
                    # OPTIMIZATION: Remaining viz tabs every 2 frames
                    if self.frames_processed % 2 == 0:
                        # All downstream viz tabs use BFS-corrected keypoints
                        darkest_viz = self.draw_darkest_pixels_visualization(frame_gray, all_bee_data)
                        darkest_mask_viz = self.draw_darkest_mask_visualization(frame_gray, all_bee_data)
                        darkest_mask_no_kp_viz = self.draw_darkest_mask_no_keypoints(frame_gray, all_bee_data)
                        body_kp_roi_viz = self.draw_body_keypoints_roi_vectors(frame_gray, all_bee_data)
                        roi_bbox_body_viz = self.draw_roi_bbox_body(frame_gray, all_bee_data)
                        grayscale_body_roi_viz = self.draw_grayscale_body_roi(frame_gray, all_bee_data)
                        full_frame_darkest_viz = self.draw_full_frame_darkest(frame_gray, all_bee_data)
                        
                        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_frame.shape
                        bytes_per_line = ch * w
                        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                        self.frame_processed.emit(qt_img.copy())
                        
                        if darkest_viz is not None:
                            rgb_darkest = cv2.cvtColor(darkest_viz, cv2.COLOR_BGR2RGB)
                            h_d, w_d, ch_d = rgb_darkest.shape
                            bytes_per_line_d = ch_d * w_d
                            qt_darkest = QImage(rgb_darkest.data, w_d, h_d, bytes_per_line_d, QImage.Format.Format_RGB888)
                            self.darkest_visualization.emit(qt_darkest.copy())
                        
                        if darkest_mask_viz is not None:
                            rgb_mask = cv2.cvtColor(darkest_mask_viz, cv2.COLOR_BGR2RGB)
                            h_m, w_m, ch_m = rgb_mask.shape
                            bytes_per_line_m = ch_m * w_m
                            qt_mask = QImage(rgb_mask.data, w_m, h_m, bytes_per_line_m, QImage.Format.Format_RGB888)
                            self.darkest_mask_visualization.emit(qt_mask.copy())
                        
                        if darkest_mask_no_kp_viz is not None:
                            rgb_no_kp = cv2.cvtColor(darkest_mask_no_kp_viz, cv2.COLOR_BGR2RGB)
                            h_nk, w_nk, ch_nk = rgb_no_kp.shape
                            bytes_per_line_nk = ch_nk * w_nk
                            qt_no_kp = QImage(rgb_no_kp.data, w_nk, h_nk, bytes_per_line_nk, QImage.Format.Format_RGB888)
                            self.darkest_mask_no_keypoints.emit(qt_no_kp.copy())
                        
                        if body_kp_roi_viz is not None:
                            rgb_body_roi = cv2.cvtColor(body_kp_roi_viz, cv2.COLOR_BGR2RGB)
                            h_br, w_br, ch_br = rgb_body_roi.shape
                            bytes_per_line_br = ch_br * w_br
                            qt_body_roi = QImage(rgb_body_roi.data, w_br, h_br, bytes_per_line_br, QImage.Format.Format_RGB888)
                            self.body_keypoints_roi_vectors.emit(qt_body_roi.copy())
                        
                        if roi_bbox_body_viz is not None:
                            rgb_roi_bbox = cv2.cvtColor(roi_bbox_body_viz, cv2.COLOR_BGR2RGB)
                            h_rbb, w_rbb, ch_rbb = rgb_roi_bbox.shape
                            bytes_per_line_rbb = ch_rbb * w_rbb
                            qt_roi_bbox = QImage(rgb_roi_bbox.data, w_rbb, h_rbb, bytes_per_line_rbb, QImage.Format.Format_RGB888)
                            self.roi_bbox_body.emit(qt_roi_bbox.copy())
                        
                        if grayscale_body_roi_viz is not None:
                            rgb_gray_roi = cv2.cvtColor(grayscale_body_roi_viz, cv2.COLOR_BGR2RGB)
                            h_gr, w_gr, ch_gr = rgb_gray_roi.shape
                            bytes_per_line_gr = ch_gr * w_gr
                            qt_gray_roi = QImage(rgb_gray_roi.data, w_gr, h_gr, bytes_per_line_gr, QImage.Format.Format_RGB888)
                            self.grayscale_body_roi.emit(qt_gray_roi.copy())
                        
                        if full_frame_darkest_viz is not None:
                            rgb_full = cv2.cvtColor(full_frame_darkest_viz, cv2.COLOR_BGR2RGB)
                            h_f, w_f, ch_f = rgb_full.shape
                            bytes_per_line_f = ch_f * w_f
                            qt_full = QImage(rgb_full.data, w_f, h_f, bytes_per_line_f, QImage.Format.Format_RGB888)
                            self.full_frame_darkest.emit(qt_full.copy())
                        
                        # (research viz computed and emitted earlier in this block)
                    
                    self.frames_processed += 1
                    if self.frames_processed % 30 == 0:
                        device_status = f"[GPU: {CUDA_DEVICE_NAME}]" if self.use_cuda else "[CPU Mode]"
                        print(f"[FRAME {self.frames_processed}] {device_status}")
                    
                    # Update metrics
                    if self.frames_processed % 10 == 0:
                        # Convert pending events to list format for emission
                        pending_events_list = []
                        live_interpretations = {}  # NEW: Store live event interpretations
                        
                        print(f"[METRICS EMIT] Frame {self.frames_processed}: Emitting metrics update")
                        
                        for event_key, event_data in self.trophallaxis_detector.pending_events.items():
                            duration_frames = self.frames_processed - event_data['start_frame']
                            duration_seconds = duration_frames / self.input_fps  # fps from source
                            event_data['duration_seconds'] = duration_seconds
                            pending_events_list.append(event_data)
                            
                            # NEW: Generate live interpretation for this event
                            live_interp = self.trophallaxis_detector.get_live_event_interpretation(
                                event_key, event_data, self.frames_processed, fps=self.input_fps
                            )
                            live_interpretations[event_key] = live_interp
                        
                        # Sort by duration (longest first)
                        pending_events_list.sort(key=lambda x: x.get('duration_seconds', 0), reverse=True)
                        
                        # NEW: Get aggregate colony interpretation
                        colony_interpretation = self.trophallaxis_detector.get_aggregate_colony_interpretation(fps=self.input_fps)
                        
                        # Calculate antenna dominance metrics
                        antenna_metrics = self._calculate_antenna_dominance_metrics()
                        
                        # NEW: Get 4-region data
                        region_stats = self.region_tracker.get_current_window_stats()
                        region_health = self.region_tracker.get_region_health_assessment()
                        region_anomalies = self.region_tracker.detect_anomalies()
                        
                        metrics_data = {
                            'total_events': len(self.trophallaxis_detector.completed_events),
                            'active_events': len(self.trophallaxis_detector.pending_events),
                            'pending_events': pending_events_list,
                            'interpretation': colony_interpretation,
                            'health_status': self.get_colony_health_status(),
                            'completed_events': self.trophallaxis_detector.completed_events[-5:] if self.trophallaxis_detector.completed_events else [],
                            'live_interpretations': live_interpretations,
                            'antenna_metrics': antenna_metrics,
                            'antenna_contacts': self.antenna_tracker.get_aggregate_antenna_metrics(),  # NEW: Antenna contact data
                            'region_stats': region_stats,
                            'region_health': region_health,
                            'region_anomalies': region_anomalies,
                            'region_timeseries': self.region_tracker.timeseries,
                            'frame_number': frame_id,
                            'active_pairs': len(self.region_tracker.frame_data)
                        }
                        self.metrics_updated.emit(metrics_data)                    
                    frame_id += 1
                
                cap.release()
                print(f"\n[INFERENCE COMPLETE] Processed {self.frames_processed} frames on {TORCH_DEVICE}")
                self.progress.emit("Analysis Complete.")
                self.finished.emit("Complete")
                
            except Exception as e:
                print(f"[ERROR] {str(e)}")
                self.error.emit(str(e))
class TrophallaxisDetector:
    """Detects and analyzes trophallaxis events using 4 conditions with real-time tracking"""
    
    def __init__(self):
        self.active_events = {}  # {(bee_id_A, bee_id_B): event_data}
        self.completed_events = []
        self.pending_events = {}  # Events being monitored before 10s confirmation
        self.event_counter = 0
        
        # Thresholds
        self.min_straightness = 0.70
        self.min_angle_range = 120  # degrees
        self.max_angle_range = 240  # degrees
        self.min_duration = 10.0  # seconds
        self.antenna_contact_threshold = 10  # pixels
        self.min_event_frames = 300  # 10 seconds at 30 FPS
        
        # ADD THESE NEW LINES:
        self.head_distance_min = HEAD_DISTANCE_MIN  # pixels
        self.head_distance_max = HEAD_DISTANCE_MAX  # pixels
        
        # NEW: Forfeit system for pending events
        self.condition_violation_frames = {}  # {event_key: {'condition_X': frame_count}}
        self.violation_threshold_frames = 150  # 5 seconds at 30 FPS
        
        # NEW: Antenna dominance tracking
        self.antenna_dominance_metrics = defaultdict(lambda: AntennaContactMetrics())
        self.interpretation_engine = AntennaInterpretationEngine()
        
        # Use shared antenna analyzer
        self.antenna_analyzer = AntennaContactAnalyzer(threshold=self.antenna_contact_threshold)
    
    def calculate_straightness(self, keypoints):
        """Calculate body straightness from keypoints 0->1->2->3->4"""
        if len(keypoints) < 5:
            return 0.0
        
        head = np.array([keypoints[0][0], keypoints[0][1]])
        abdomen = np.array([keypoints[4][0], keypoints[4][1]])
        
        line_vector = abdomen - head
        line_length = np.linalg.norm(line_vector)
        
        if line_length == 0:
            return 0.0
        
        line_unit = line_vector / line_length
        total_deviation = 0
        max_possible_deviation = 0
        
        for i in range(1, 4):
            kp = np.array([keypoints[i][0], keypoints[i][1]])
            to_point = kp - head
            projection_length = np.dot(to_point, line_unit)
            projection = head + projection_length * line_unit
            deviation = np.linalg.norm(kp - projection)
            total_deviation += deviation
            max_possible_deviation += line_length * 0.15
        
        straightness = 1.0 - (total_deviation / (max_possible_deviation + 1e-6))
        return max(0.0, min(1.0, straightness))
    
    def calculate_direction_angle(self, keypoints):
        """Calculate direction angle from head (0) to abdomen (4)"""
        head = np.array([keypoints[0][0], keypoints[0][1]])
        abdomen = np.array([keypoints[4][0], keypoints[4][1]])
        
        direction = abdomen - head
        angle_rad = np.arctan2(direction[1], direction[0])
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg % 360
    
    def calculate_angular_difference(self, angle_a, angle_b):
        """Calculate angular difference (target: 180° ± 30°)"""
        diff = abs(angle_a - angle_b)
        
        if diff > 180:
            diff = 360 - diff
        
        return diff
    
    # ADD THIS NEW METHOD:
    def calculate_head_distance(self, kp_a, kp_b):
        """Calculate Euclidean distance between head keypoints (kp[0])"""
        head_a = np.array([kp_a[0][0], kp_a[0][1]])
        head_b = np.array([kp_b[0][0], kp_b[0][1]])
        
        distance = np.linalg.norm(head_a - head_b)
        return distance
    
    def check_antenna_intersection(self, kp_a, kp_b):
        """Check antenna contacts and classify into RR/RL/LR/LL types"""
        analyzer = AntennaContactAnalyzer(threshold=self.antenna_contact_threshold)
        return analyzer.check_antenna_intersection(kp_a, kp_b)
    
    def get_condition_status(self, straightness_a, straightness_b, angle_diff, contacts, duration_seconds, head_distance):
        """Get detailed condition status for monitoring"""
        status = {
            'condition_1': {
                'name': 'Body Straightness',
                'passed': straightness_a >= self.min_straightness and straightness_b >= self.min_straightness,
                'value_a': straightness_a,
                'value_b': straightness_b,
                'threshold': self.min_straightness,
                'description': f'A: {straightness_a:.2f} | B: {straightness_b:.2f} | Threshold: {self.min_straightness}'
            },
            'condition_2': {
                'name': 'Angular Alignment',
                'passed': self.min_angle_range <= angle_diff <= self.max_angle_range,
                'value': angle_diff,
                'min_threshold': self.min_angle_range,
                'max_threshold': self.max_angle_range,
                'description': f'{angle_diff:.1f}° | Range: {self.min_angle_range}-{self.max_angle_range}°'
            },
            'condition_4': {
                'name': 'Antenna Contact',
                'passed': any(contacts.values()),
                'contact_types': contacts,
                'description': f"Contacts: {', '.join([k for k,v in contacts.items() if v])}"
            },
            'condition_3': {
                'name': 'Duration Threshold',
                'passed': duration_seconds >= self.min_duration,
                'value': duration_seconds,
                'threshold': self.min_duration,
                'description': f'{duration_seconds:.1f}s / {self.min_duration}s'
            },
            # ADD THIS NEW CONDITION:
            'condition_5': {
                'name': 'Head Proximity',
                'passed': self.head_distance_min <= head_distance <= self.head_distance_max,
                'value': head_distance,
                'min_threshold': self.head_distance_min,
                'max_threshold': self.head_distance_max,
                'description': f'{head_distance:.1f}px | Range: {self.head_distance_min}-{self.head_distance_max}px'
            }
        }
        
        return status
    def get_live_event_interpretation(self, event_key, event_data, frame_number, fps=30):
        """Generate dynamic interpretation for a single live event based on current metrics"""
        
        bee_a = event_data['bee_a_id']
        bee_b = event_data['bee_b_id']
        straightness_a = event_data['straightness_a']
        straightness_b = event_data['straightness_b']
        angle_diff = event_data['angle_diff']
        contacts = event_data['antenna_contacts']
        distances = event_data['antenna_distances']
        head_distance = event_data.get('head_distance', 999)        
        duration_frames = frame_number - event_data['start_frame']
        duration_seconds = duration_frames / fps
        
        # Calculate RR dominance
        total_contacts = sum(contacts.values())
        rr_ratio = contacts['RR'] / total_contacts if total_contacts > 0 else 0.0
        
        # SCENARIO 1: ALIGNMENT INTERPRETATION
        avg_straightness = (straightness_a + straightness_b) / 2
        
        if 175 <= angle_diff <= 185 and avg_straightness > 0.87:
            alignment_text = f"🟢 PERFECT ALIGNMENT: {angle_diff:.1f}° with excellent body control ({straightness_a:.2f} & {straightness_b:.2f}). Sustained right-antenna dominance ({rr_ratio*100:.0f}% RR). Expected efficiency: >95%."
        elif 170 <= angle_diff <= 190 and avg_straightness >= 0.85:
            alignment_text = f"🟢 GOOD ALIGNMENT: {angle_diff:.1f}° with stable posture ({straightness_a:.2f} & {straightness_b:.2f}). Right-antenna usage dominant ({rr_ratio*100:.0f}% RR). Efficiency 85-90%."
        elif (150 <= angle_diff <= 160 or 200 <= angle_diff <= 210) and avg_straightness >= 0.83:
            alignment_text = f"🟡 MARGINAL ALIGNMENT: {angle_diff:.1f}° with slightly reduced straightness ({straightness_a:.2f} & {straightness_b:.2f}). Mixed antenna contacts ({rr_ratio*100:.0f}% RR). Efficiency 70-80%. Monitor for disengagement."
        else:
            alignment_text = f"🔴 POOR ALIGNMENT: {angle_diff:.1f}° outside optimal range with reduced straightness ({straightness_a:.2f} & {straightness_b:.2f}). Excessive cross-pattern usage. Efficiency <50%. MONITOR CLOSELY."
        
        # SCENARIO 2: ANTENNA CONTACT INTERPRETATION
        if total_contacts == 0:
            antenna_text = "⚠️ NO ANTENNA CONTACT detected yet. Monitoring for alignment."
        elif rr_ratio >= 0.95:
            antenna_text = f"🟢 PURE RIGHT DOMINANCE: {contacts['RR']}/{total_contacts} contacts = {rr_ratio*100:.0f}% RR. Perfect antenna communication. Maximum transfer efficiency."
        elif rr_ratio >= 0.80:
            antenna_text = f"🟢 RIGHT-DOMINANT: {rr_ratio*100:.0f}% RR with minor cross-pattern ({100-rr_ratio*100:.0f}% RL/LR). Fine-tuning alignment. Transfer quality excellent."
        elif rr_ratio >= 0.60:
            antenna_text = f"🟡 BALANCED PATTERN: {rr_ratio*100:.0f}% RR vs {100-rr_ratio*100:.0f}% cross-pattern. Stress responses detected. Transfer efficiency 60-75%."
        else:
            antenna_text = f"🔴 STRESS SIGNAL: High non-dominant usage ({100-rr_ratio*100:.0f}% left/cross). Possible food contamination or donor illness. Abort risk HIGH."
        
        # SCENARIO 3: DURATION INTERPRETATION
        if duration_seconds >= 9.5:
            duration_text = f"🟢 EXCELLENT DURATION: {duration_seconds:.1f}/10.0 sec ({duration_seconds*10:.0f}% complete). Unwavering commitment. Prediction: Will confirm as valid trophallaxis."
        elif duration_seconds >= 7.0:
            duration_text = f"🟡 IN PROGRESS: {duration_seconds:.1f}/10.0 sec. Sustained transfer, monitoring stability."
        else:
            duration_text = f"⏳ EARLY STAGE: {duration_seconds:.1f}/10.0 sec. Event establishing, conditions must hold."
        
        # SCENARIO 4: CLOSEST ANTENNA DISTANCE
        closest_dist = min(distances.values()) if distances else 999
        if closest_dist < self.antenna_contact_threshold:
            distance_text = f"📍 Antennas in contact: {closest_dist:.1f}px (threshold: {self.antenna_contact_threshold}px)"
        else:
            distance_text = f"📍 Closest antennas: {closest_dist:.1f}px (threshold: {self.antenna_contact_threshold}px) - Not yet contacting"
        
        # ADD THIS NEW SCENARIO:
        # SCENARIO 5: HEAD PROXIMITY
        if self.head_distance_min <= head_distance <= self.head_distance_max:
            head_text = f"📏 Head proximity: {head_distance:.1f}px (OPTIMAL range: {self.head_distance_min}-{self.head_distance_max}px)"
        elif head_distance > self.head_distance_max:
            head_text = f"⚠️ Heads too far: {head_distance:.1f}px (MAX: {self.head_distance_max}px) - Bees may disengage"
        else:
            head_text = f"📏 Head proximity: {head_distance:.1f}px"
        
        # Compile interpretation
        interpretation = f"""
BEE PAIR: {bee_a} ↔ {bee_b} | Duration: {duration_seconds:.1f}s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{alignment_text}

{antenna_text}

{distance_text}

{head_text}

{duration_text}
"""
        
        return interpretation.strip()
    
    def get_aggregate_colony_interpretation(self, fps=30):
        """Generate dynamic interpretation of overall colony health based on all completed events"""
        
        if len(self.completed_events) == 0:
            return "⏳ Waiting for trophallaxis events to analyze..."
        
        completed = [e for e in self.completed_events if e.get('confirmed', False)]
        
        if not completed:
            return "⏳ No confirmed trophallaxis events yet..."
        
        # Calculate aggregate metrics
        total_contacts = sum(sum(e['antenna_contacts'].values()) for e in completed)
        total_rr = sum(e['antenna_contacts']['RR'] for e in completed)
        rr_ratio = total_rr / total_contacts if total_contacts > 0 else 0.0
        
        avg_quality = np.mean([self.calculate_event_quality_score(e) for e in completed])
        avg_duration = np.mean([e['duration_seconds'] for e in completed])
        continuous_events = sum(1 for e in completed if e['frames_active'] > 290)
        continuous_ratio = continuous_events / len(completed) if completed else 0.0
        
        aborted_count = len(self.pending_events)
        abort_rate = aborted_count / (len(completed) + aborted_count) if (len(completed) + aborted_count) > 0 else 0.0
        
        # SCENARIO 4A: EXCELLENT HEALTH
        if rr_ratio >= 0.85 and avg_quality >= 92 and continuous_ratio >= 0.90:
            interpretation = (
                f"🟢 EXCELLENT COLONY HEALTH\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Right-antenna dominance: {rr_ratio*100:.1f}% (Optimal ≥85%)\n"
                f"Quality score: {avg_quality:.0f}/100 (Excellent >92)\n"
                f"Sustained alignment: {continuous_ratio*100:.1f}% (Good ≥90%)\n\n"
                f"Colony operating at peak social efficiency with minimal stress, "
                f"excellent food quality acceptance, and strong inter-bee bonding.\n"
                f"Nutritional distribution: OPTIMAL | Survival prediction: 98%+"
            )
        
        # SCENARIO 4B: HEALTHY COLONY
        elif rr_ratio >= 0.75 and avg_quality >= 85 and continuous_ratio >= 0.80:
            interpretation = (
                f"🟢 HEALTHY COLONY\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Right-antenna dominance: {rr_ratio*100:.1f}% (Good 75-85%)\n"
                f"Quality score: {avg_quality:.0f}/100 (Good 85-92)\n"
                f"Sustained alignment: {continuous_ratio*100:.1f}% (Good ≥80%)\n\n"
                f"Colony showing strong social cohesion and healthy food distribution. "
                f"Minor cross-pattern usage normal. Manageable stress levels.\n"
                f"Overall health: EXCELLENT | Survival prediction: 95%+"
            )
        
        # SCENARIO 4C: STRESSED COLONY
        elif rr_ratio >= 0.50 and avg_quality >= 70:
            interpretation = (
                f"🟡 MODERATE STRESS DETECTED\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Right-antenna dominance: {rr_ratio*100:.1f}% (Declining 50-74%)\n"
                f"Quality score: {avg_quality:.0f}/100 (Declining 70-85)\n"
                f"Sustained alignment: {continuous_ratio*100:.1f}% (Frequent breaks <80%)\n"
                f"Abort rate: {abort_rate*100:.1f}%\n\n"
                f"⚠️ Colony stress increasing. Possible causes: Dehydration, heat stress, "
                f"vibration exposure, or pathogen exposure.\n"
                f"RECOMMEND: Check water/nectar supply, inspect for Varroa/Nosema, "
                f"adjust environmental conditions.\n"
                f"Survival prediction: 85-90%"
            )
        
        # SCENARIO 4D: CRITICAL STRESS
        else:
            interpretation = (
                f"🔴 CRITICAL STRESS - IMMEDIATE ACTION REQUIRED\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Right-antenna dominance: {rr_ratio*100:.1f}% (Severely reduced <50%)\n"
                f"Quality score: {avg_quality:.0f}/100 (Poor <70)\n"
                f"Sustained alignment: {continuous_ratio*100:.1f}% (Highly interrupted <65%)\n"
                f"Abort rate: {abort_rate*100:.1f}%\n\n"
                f"🚨 Colony suffering from critical stress. Strong indicators of:\n"
                f"• Severe dehydration or starvation\n"
                f"• Serious pathogen outbreak (Varroa/Nosema)\n"
                f"• Pesticide or heat shock exposure\n\n"
                f"IMMEDIATE INTERVENTION REQUIRED:\n"
                f"✓ Assess water and food availability\n"
                f"✓ Perform health inspection (Varroa, Nosema)\n"
                f"✓ Provide shelter from environmental extremes\n"
                f"✓ Consider emergency feeding\n\n"
                f"Survival prediction without intervention: 50-70% | Mortality risk: 20-40%"
            )
        
        return interpretation
    
  
    def detect_trophallaxis(self, all_bee_data, frame_number, fps=30):
        """Main detection function - call every frame
        
        CONDITIONS FOR TROPHALLAXIS (Antenna contact now optional):
        1. Body Straightness: Both bees ≥ 0.85
        2. Angular Alignment: 120° ≤ angle_diff ≤ 240°
        3. Duration: ≥ 10 seconds (for confirmation)
        
        Antenna contact is TRACKED but NOT REQUIRED for confirmation
        
        FORFEIT SYSTEM: Any condition absent for >5 seconds causes pending event to abort
        """
        
        bee_ids = list(all_bee_data.keys())
        
        # NEW: Check for condition violation timeouts on existing pending events
        self._check_condition_violation_timeouts(frame_number)
        
        for i, bee_a_id in enumerate(bee_ids):
            for bee_b_id in bee_ids[i+1:]:
                
                kp_a, box_a = all_bee_data[bee_a_id]
                kp_b, box_b = all_bee_data[bee_b_id]
                
                if kp_a is None or kp_b is None:
                    continue
                
                # CONDITION 1: STRAIGHTNESS
                straightness_a = self.calculate_straightness(kp_a)
                straightness_b = self.calculate_straightness(kp_b)
                
                cond1_pass = (straightness_a > self.min_straightness and 
                            straightness_b > self.min_straightness)
                
                event_key = tuple(sorted([bee_a_id, bee_b_id]))
                
                if not cond1_pass:
                    # NEW: Track violation instead of immediate removal
                    self._track_condition_violation(event_key, 'condition_1', frame_number)
                    continue
                else:
                    # Condition passed - clear any violation record
                    self._clear_condition_violation(event_key, 'condition_1')
                
                # CONDITION 2: ANGULAR ALIGNMENT
                angle_a = self.calculate_direction_angle(kp_a)
                angle_b = self.calculate_direction_angle(kp_b)
                angle_diff = self.calculate_angular_difference(angle_a, angle_b)
                
                cond2_pass = (self.min_angle_range <= angle_diff <= self.max_angle_range)
                
                if not cond2_pass:
                    # NEW: Track violation instead of immediate removal
                    self._track_condition_violation(event_key, 'condition_2', frame_number)
                    continue
                else:
                    # Condition passed - clear any violation record
                    self._clear_condition_violation(event_key, 'condition_2')
                # ADD THIS NEW CONDITION CHECK:
                # CONDITION 5: HEAD PROXIMITY
                head_distance = self.calculate_head_distance(kp_a, kp_b)
                
                cond5_pass = (self.head_distance_min <= head_distance <= self.head_distance_max)
                
                if not cond5_pass:
                    # Track violation instead of immediate removal
                    self._track_condition_violation(event_key, 'condition_5', frame_number)
                    continue
                else:
                    # Condition passed - clear any violation record
                    self._clear_condition_violation(event_key, 'condition_5')                
                # OPTIONAL: ANTENNA CONTACT (tracked but not required)
                contacts, distances = self.check_antenna_intersection(kp_a, kp_b)
                
                # REMOVED: cond4_pass requirement
                # Now just track antenna contacts for health assessment
                
                # Track antenna contact types in metrics
                event_key = tuple(sorted([bee_a_id, bee_b_id]))
                if contacts['RR']:
                    self.antenna_dominance_metrics[event_key].rr_count += 1
                if contacts['RL']:
                    self.antenna_dominance_metrics[event_key].rl_count += 1
                if contacts['LR']:
                    self.antenna_dominance_metrics[event_key].lr_count += 1
                if contacts['LL']:
                    self.antenna_dominance_metrics[event_key].ll_count += 1
                
                # All required conditions passed (1, 2 only) - Create/Update pending event
                event_key = tuple(sorted([bee_a_id, bee_b_id]))
                
                # OPTIMIZED: Single dictionary lookup
                event = self.pending_events.get(event_key)
                
                if event is None:
                    # NEW PENDING EVENT
                    self.event_counter += 1
                    event = {
                        'event_id': self.event_counter,
                        'bee_a_id': bee_a_id,
                        'bee_b_id': bee_b_id,
                        'start_frame': frame_number,
                        'start_time': frame_number / fps,
                        'straightness_a': straightness_a,
                        'straightness_b': straightness_b,
                        'angle_a': angle_a,
                        'angle_b': angle_b,
                        'angle_diff': angle_diff,
                        'antenna_contacts': {'RR': 0, 'RL': 0, 'LR': 0, 'LL': 0},
                        'antenna_distances': distances,
                        'head_distance': head_distance,
                        'contact_points': [],
                        'last_contact_frame': frame_number,
                        'frames_active': 1,
                        'condition_history': []
                    }
                    self.pending_events[event_key] = event
                
                else:
                    # UPDATE PENDING EVENT - OPTIMIZED batch update
                    event.update({
                        'straightness_a': straightness_a,
                        'straightness_b': straightness_b,
                        'angle_diff': angle_diff,
                        'antenna_distances': distances,
                        'head_distance': head_distance,
                        'frames_active': event['frames_active'] + 1,
                        'last_contact_frame': frame_number
                    })
                    
                    # Track antenna contacts (for health metrics)
                    for contact_type, is_contact in contacts.items():
                        if is_contact:
                            event['antenna_contacts'][contact_type] += 1
                
                # CONDITION 3: CHECK DURATION
                duration_frames = frame_number - self.pending_events[event_key]['start_frame']
                duration_seconds = duration_frames / fps
                
                # Store condition status in history
                condition_status = self.get_condition_status(
                    straightness_a, straightness_b, angle_diff, contacts, duration_seconds, head_distance
                )
                self.pending_events[event_key]['condition_history'] = condition_status
                
                if duration_seconds >= self.min_duration:
                    # CONFIRMED - Move to active/completed
                    self.end_event(event_key, frame_number, fps, confirmed=True)
        
        # OPTIMIZATION: Limit completed_events size (prevent unbounded growth)
        if len(self.completed_events) > 200:
            self.completed_events = self.completed_events[-200:]
    def _track_condition_violation(self, event_key, condition_name, frame_number):
        """Track when a condition fails for a pending event"""
        if event_key not in self.condition_violation_frames:
            self.condition_violation_frames[event_key] = {}
        
        if condition_name not in self.condition_violation_frames[event_key]:
            self.condition_violation_frames[event_key][condition_name] = frame_number
    
    def _clear_condition_violation(self, event_key, condition_name):
        """Clear violation tracking when condition is restored"""
        if event_key in self.condition_violation_frames:
            self.condition_violation_frames[event_key].pop(condition_name, None)
            
            # If all violations cleared, remove event key from tracking
            if not self.condition_violation_frames[event_key]:
                self.condition_violation_frames.pop(event_key, None)
    
    def _check_condition_violation_timeouts(self, frame_number):
        """Remove pending events if any condition violated for >5 seconds"""
        events_to_forfeit = []
        
        for event_key, violation_dict in self.condition_violation_frames.items():
            for condition_name, violation_start_frame in violation_dict.items():
                frames_violated = frame_number - violation_start_frame
                
                if frames_violated > self.violation_threshold_frames:
                    # Violation exceeded 5 seconds - forfeit event
                    if event_key not in events_to_forfeit:
                        events_to_forfeit.append(event_key)
        
        # Remove forfeited events
        for event_key in events_to_forfeit:
            if event_key in self.pending_events:
                self.pending_events.pop(event_key, None)
            self.condition_violation_frames.pop(event_key, None)    
    def calculate_event_quality_score(self, event):
        """Calculate quality score 0-100 based on all metrics"""
        score = 50
        
        avg_straightness = (event['straightness_a'] + event['straightness_b']) / 2
        score += (avg_straightness - 0.85) * 100
        
        angle_diff = event['angle_diff']
        if 175 <= angle_diff <= 185:
            score += 15
        elif 170 <= angle_diff <= 190:
            score += 10
        elif 150 <= angle_diff <= 210:
            score += 5
        
        rr_dom = event.get('rr_dominance', 0)
        score += rr_dom * 20
        
        if event.get('duration_seconds', 0) >= 10.0:
            score += 10
        
        return min(100, max(0, score))
    
    def end_event(self, event_key, frame_number, fps=30, confirmed=False):
        """End an event and move to completed"""
        event = self.pending_events[event_key]
        
        duration_frames = frame_number - event['start_frame']
        duration_seconds = duration_frames / fps
        
        event['end_frame'] = frame_number
        event['duration_frames'] = duration_frames
        event['duration_seconds'] = duration_seconds
        event['confirmed'] = confirmed
        
        total_contacts = sum(event['antenna_contacts'].values())
        if total_contacts > 0:
            event['rr_dominance'] = event['antenna_contacts']['RR'] / total_contacts
        else:
            event['rr_dominance'] = 0.0
        
        # Store antenna metrics from tracking
        if event_key in self.antenna_dominance_metrics:
            metrics = self.antenna_dominance_metrics[event_key]
            event['antenna_metrics'] = {
                'rr_count': metrics.rr_count,
                'rl_count': metrics.rl_count,
                'lr_count': metrics.lr_count,
                'll_count': metrics.ll_count,
                'rr_ratio': metrics.rr_ratio,
                'll_ratio': metrics.ll_ratio,
                'cross_ratio': metrics.cross_ratio,
                'lateralization_index': metrics.lateralization_index,
                'symmetry_ratio': metrics.symmetry_ratio
            }
        
        self.completed_events.append(event)
        self.pending_events.pop(event_key, None)

# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

EVAL_MATCHING_THRESHOLD_PX = 50

KEYPOINT_NAMES = {
    0: "k0", 1: "k1", 2: "k2", 3: "k3", 4: "k4",
    5: "k5", 6: "k6", 7: "k7", 8: "k8"
}
KEYPOINT_LABELS = ["k0","k1","k2","k3","k4","k5","k6","k7","k8"]
KP_GROUPS = {"body": [0,1,2,3,4], "antenna": [5,6,7,8]}


def _bee_centroid_from_flat_keypoints(kp_flat):
    """Compute centroid from visible keypoints in a flat [x,y,v, x,y,v, ...] array."""
    xs, ys = [], []
    for k in range(min(9, len(kp_flat) // 3)):
        x, y, v = kp_flat[k*3], kp_flat[k*3+1], kp_flat[k*3+2]
        if v > 0 and (x > 0 or y > 0):
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    return (np.mean(xs), np.mean(ys))


def filter_predictions_to_gt_bees(gt_annotations, pred_annotations, max_dist=80):
    """
    Keep only predicted bees that spatially correspond to a GT-annotated bee.
    Uses centroid of visible keypoints. Greedy closest-first matching.
    """
    if not gt_annotations or not pred_annotations:
        return pred_annotations

    gt_centroids = [_bee_centroid_from_flat_keypoints(gt['keypoints']) for gt in gt_annotations]
    pred_centroids = [_bee_centroid_from_flat_keypoints(p['keypoints']) for p in pred_annotations]

    pairs = []
    for gi, gc in enumerate(gt_centroids):
        if gc is None:
            continue
        for pi, pc in enumerate(pred_centroids):
            if pc is None:
                continue
            dist = np.sqrt((gc[0] - pc[0])**2 + (gc[1] - pc[1])**2)
            if dist < max_dist:
                pairs.append((dist, gi, pi))

    pairs.sort(key=lambda x: x[0])
    used_gt, used_pred, matched_pred_indices = set(), set(), set()
    for dist, gi, pi in pairs:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matched_pred_indices.add(pi)

    return [pred_annotations[i] for i in sorted(matched_pred_indices)]


def extract_all_keypoints_eval(annotations_list, keypoint_type='gt'):
    all_keypoints = []
    for bee_id, ann in enumerate(annotations_list):
        kp_flat = ann['keypoints']
        if len(kp_flat) < 27:
            kp_flat = list(kp_flat) + [0.0] * (27 - len(kp_flat))
        for k_idx in range(9):
            x = kp_flat[k_idx * 3]
            y = kp_flat[k_idx * 3 + 1]
            v = kp_flat[k_idx * 3 + 2]
            if v > 0:
                all_keypoints.append({'index': k_idx, 'x': x, 'y': y, 'visibility': v,
                                       'type': keypoint_type, 'bee_id': bee_id})
    return all_keypoints


def match_keypoints_spatially_eval(gt_keypoints, pred_keypoints, threshold=EVAL_MATCHING_THRESHOLD_PX):
    matches = []
    gt_by_index = {}
    pred_by_index = {}
    for gt_kp in gt_keypoints:
        gt_by_index.setdefault(gt_kp['index'], []).append(gt_kp)
    for pred_kp in pred_keypoints:
        pred_by_index.setdefault(pred_kp['index'], []).append(pred_kp)
    used_pred_global = set()
    for k_idx in range(9):
        if k_idx not in gt_by_index or k_idx not in pred_by_index:
            continue
        for gt_kp in gt_by_index[k_idx]:
            best_pred, best_dist = None, float('inf')
            for pred_kp in pred_by_index[k_idx]:
                if id(pred_kp) in used_pred_global:
                    continue
                dist = np.sqrt((gt_kp['x']-pred_kp['x'])**2 + (gt_kp['y']-pred_kp['y'])**2)
                if dist < best_dist and dist < threshold:
                    best_dist, best_pred = dist, pred_kp
            if best_pred is not None:
                matches.append((gt_kp, best_pred, best_dist))
                used_pred_global.add(id(best_pred))
    matched_gt_ids = set(id(m[0]) for m in matches)
    matched_pred_ids = set(id(m[1]) for m in matches)
    unmatched_gt = [kp for kp in gt_keypoints if id(kp) not in matched_gt_ids]
    unmatched_pred = [kp for kp in pred_keypoints if id(kp) not in matched_pred_ids]
    return matches, unmatched_gt, unmatched_pred


def reorder_keypoints_roboflow_to_pipeline(kp_flat):
    if len(kp_flat) < 27:
        kp_flat = list(kp_flat) + [0.0] * (27 - len(kp_flat))
    robo_head    = kp_flat[0:3]
    robo_thorax  = kp_flat[3:6]
    robo_abdomen = kp_flat[6:9]
    robo_l_ant   = kp_flat[9:12]
    robo_r_ant   = kp_flat[12:15]
    robo_l_wing  = kp_flat[15:18]
    robo_r_wing  = kp_flat[18:21]
    robo_l_leg   = kp_flat[21:24]
    robo_r_leg   = kp_flat[24:27]
    return (robo_head + robo_l_ant + robo_r_ant + robo_thorax +
            robo_l_wing + robo_r_wing + robo_abdomen + robo_l_leg + robo_r_leg)


def _fig_to_qpixmap(fig, dpi=90):
    """Convert matplotlib figure to QPixmap."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    data = buf.read()
    qimg = QImage.fromData(data)
    return QPixmap.fromImage(qimg)


def _make_error_bar_chart(metrics):
    """Bar chart: mean pixel error per keypoint."""
    fig, ax = plt.subplots(figsize=(7, 4), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    errors = [metrics.get(k, {}).get('mean_error', 0) for k in range(9)]
    thresholds = [(0,10,'#90ee90'), (10,20,'#ffff99'), (20,30,'#ffa500'), (30,1e9,'#ff9999')]
    colors = []
    for e in errors:
        for lo, hi, c in thresholds:
            if lo <= e < hi:
                colors.append(c); break
        else:
            colors.append('#ff9999')
    bars = ax.bar(range(9), errors, color=colors, edgecolor='black', linewidth=0.6)
    for bar, e in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{e:.1f}', ha='center', va='bottom', fontsize=7.5, color='white')
    ax.set_xticks(range(9))
    ax.set_xticklabels(KEYPOINT_LABELS, color='white', fontsize=8)
    ax.set_xlabel('Keypoint', color='white', fontsize=9)
    ax.set_ylabel('Mean Pixel Error (px)', color='white', fontsize=9)
    ax.set_title('Mean Pixel Error by Keypoint\n(Lower is Better)', color='white', fontsize=10, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')
    ax.yaxis.grid(True, color='#444', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    legend_patches = [mpatches.Patch(color=c, label=f'{lo}-{hi if hi<1e9 else "∞"}px')
                      for lo, hi, c in thresholds]
    ax.legend(handles=legend_patches, fontsize=7, facecolor='#2d2d2d', labelcolor='white',
              edgecolor='#555', loc='upper left')
    fig.tight_layout()
    return fig


def _make_pck_chart(metrics, all_errors_by_kp, pck_thresholds=None):
    """PCK curve per distance threshold."""
    if pck_thresholds is None:
        pck_thresholds = [5, 10, 15]
    fig, ax = plt.subplots(figsize=(7, 4), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    max_thresh = 50
    thresholds = np.arange(0, max_thresh+1, 1)

    def pck_curve(indices):
        errs = []
        for k in indices:
            errs.extend(all_errors_by_kp.get(k, []))
        if not errs:
            return np.zeros(len(thresholds))
        errs = np.array(errs)
        return np.array([np.mean(errs <= t)*100 for t in thresholds])

    overall = pck_curve(range(9))
    body    = pck_curve(KP_GROUPS['body'])
    antenna = pck_curve(KP_GROUPS['antenna'])

    ax.plot(thresholds, overall, 'g-o', markersize=2, label='Overall', linewidth=1.5)
    ax.plot(thresholds, body, 'b-s', markersize=2, label='Body (k0-k4)', linewidth=1.5)
    ax.plot(thresholds, antenna, color='tomato', marker='^', markersize=2,
            label='Antenna (k5-k8)', linewidth=1.5)

    colors_pck = ['#90ee90', '#ffd700', '#ffa07a']
    for t, col in zip(pck_thresholds[:3], colors_pck):
        val = float(np.interp(t, thresholds, overall))
        ax.axvline(x=t, color=col, linestyle='--', alpha=0.6, linewidth=0.8)
        ax.text(t+0.4, val+2, f'{val:.1f}%', color=col, fontsize=7.5, fontweight='bold')
    title_pck = " | ".join([f"PCK@{t}px={float(np.interp(t,thresholds,overall)):.1f}%" for t in pck_thresholds[:3]])
    ax.set_title(f'PCK Curve\n{title_pck}', color='white', fontsize=9, fontweight='bold')

    ax.axhline(90, color='#90ee90', linestyle='--', linewidth=0.8, alpha=0.5, label='90% line')
    ax.axhline(75, color='#ffd700', linestyle='--', linewidth=0.8, alpha=0.5, label='75% line')
    ax.set_xlabel('Distance Threshold (pixels)', color='white', fontsize=9)
    ax.set_ylabel('PCK (%)', color='white', fontsize=9)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')
    ax.yaxis.grid(True, color='#444', linestyle='--', alpha=0.4)
    ax.legend(fontsize=7.5, facecolor='#2d2d2d', labelcolor='white', edgecolor='#555')
    ax.set_ylim(0, 105)
    fig.tight_layout()
    return fig


def _make_detection_chart(metrics):
    """Grouped bar chart: Precision/Recall/F1 per keypoint."""
    fig, ax = plt.subplots(figsize=(7, 4), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    x = np.arange(9)
    w = 0.25
    prec  = [metrics.get(k,{}).get('precision',0)*100 for k in range(9)]
    rec   = [metrics.get(k,{}).get('recall',0)*100    for k in range(9)]
    f1    = [metrics.get(k,{}).get('f1',0)*100        for k in range(9)]
    ax.bar(x-w, prec, w, label='Precision', color='#4caf50', edgecolor='black', linewidth=0.5)
    ax.bar(x,   rec,  w, label='Recall',    color='#2196f3', edgecolor='black', linewidth=0.5)
    ax.bar(x+w, f1,   w, label='F1-Score',  color='#ff9800', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(KEYPOINT_LABELS, color='white', fontsize=8)
    ax.set_ylabel('Percentage (%)', color='white', fontsize=9)
    ax.set_xlabel('Keypoint', color='white', fontsize=9)
    ax.set_title('Detection Metrics: Precision, Recall, and F1-Score per Keypoint', color='white', fontsize=9, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')
    ax.yaxis.grid(True, color='#444', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8, facecolor='#2d2d2d', labelcolor='white', edgecolor='#555')
    fig.tight_layout()
    return fig


def _make_correlation_chart(all_errors_by_kp):
    """Correlation matrix of errors across keypoints."""
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    n = 9
    # Build binary error vectors per frame is complex; use presence/absence approximation
    max_len = max((len(v) for v in all_errors_by_kp.values()), default=1)
    mat = np.zeros((n, max_len))
    for k in range(n):
        errs = all_errors_by_kp.get(k, [])
        mat[k, :len(errs)] = errs
    # Correlation
    valid = [k for k in range(n) if len(all_errors_by_kp.get(k,[])) > 1]
    corr = np.eye(n)
    for i in valid:
        for j in valid:
            vi = np.array(all_errors_by_kp[i][:max_len])
            vj = np.array(all_errors_by_kp[j][:max_len])
            mn = min(len(vi), len(vj))
            if mn > 1:
                c = np.corrcoef(vi[:mn], vj[:mn])[0,1]
                corr[i,j] = c if not np.isnan(c) else 0
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                    fontsize=6.5, color='black' if abs(corr[i,j]) < 0.5 else 'white', fontweight='bold')
    ax.set_xticks(range(n)); ax.set_xticklabels(KEYPOINT_LABELS, color='white', fontsize=7.5)
    ax.set_yticks(range(n)); ax.set_yticklabels(KEYPOINT_LABELS, color='white', fontsize=7.5)
    ax.set_title('Error Correlation Matrix\n(High values = errors occur together)', color='white', fontsize=9, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation Coefficient', color='white', fontsize=8)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    ax.spines[:].set_color('#444')
    fig.tight_layout()
    return fig


def _make_severity_chart(all_errors_by_kp):
    """Heatmap: error severity distribution per keypoint."""
    from matplotlib.colors import LinearSegmentedColormap as _LSC
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    bins = [(0,5,'<5px'), (5,10,'5-10px'), (10,15,'10-15px'), (15,25,'15-25px'), (25,1e9,'>25px')]
    n_bins = len(bins)
    data = np.zeros((9, n_bins))
    counts = np.zeros((9, n_bins), dtype=int)
    for k in range(9):
        errs = np.array(all_errors_by_kp.get(k, []))
        total = len(errs)
        for b_idx, (lo, hi, _) in enumerate(bins):
            mask = (errs >= lo) & (errs < hi)
            cnt = int(np.sum(mask))
            counts[k, b_idx] = cnt
            data[k, b_idx] = cnt / total * 100 if total > 0 else 0

    cmap = _LSC.from_list('sev', ['#006400','#90ee90','#ffff00','#ffa500','#ff0000'])
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect='auto')
    for i in range(9):
        for j in range(n_bins):
            pct = data[i, j]
            cnt = counts[i, j]
            if pct > 0:
                txt_color = 'white' if pct > 50 else 'black'
                ax.text(j, i, f'{pct:.0f}%\n({cnt})', ha='center', va='center',
                        fontsize=6.5, color=txt_color, fontweight='bold')
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels([b[2] for b in bins], color='white', fontsize=8)
    ax.set_yticks(range(9))
    ax.set_yticklabels(KEYPOINT_LABELS, color='white', fontsize=8)
    ax.set_xlabel('Error Range', color='white', fontsize=9)
    ax.set_ylabel('Keypoint', color='white', fontsize=9)
    ax.set_title('Error Severity Distribution Matrix\n(Percentage of keypoint errors in each range)',
                 color='white', fontsize=9, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Percentage (%)', color='white', fontsize=8)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    ax.spines[:].set_color('#444')
    fig.tight_layout()
    return fig


# ============================================================================
# CLICKABLE IMAGE LABEL — zoom on click, download button
# ============================================================================

class ClickableImageLabel(QLabel):
    """QLabel that opens a fullscreen zoom dialog on click and shows a save button."""

    def __init__(self, title="Image", border_color="#00b4d8", parent=None):
        super().__init__("—", parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            f"background:#0a0a0a; border:1px solid {border_color}; color:#444; font-size:10px;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._full_pixmap = None   # full-res pixmap stored here
        self._title       = title
        self._border_color = border_color

    def set_full_pixmap(self, pix: QPixmap):
        self._full_pixmap = pix
        # Inside a QScrollArea with widgetResizable=True the label expands
        # to fill available space — scale down to fit, never upscale past native
        self._refresh_scaled()
        self.setToolTip("Click to zoom  |  Right-click to save")

    def _refresh_scaled(self):
        if not self._full_pixmap:
            return
        # Use parent scroll area viewport size if available, else own size
        parent = self.parent()
        if parent and hasattr(parent, 'viewport'):
            available = parent.viewport().size()
        else:
            available = self.size()
        if available.width() < 10 or available.height() < 10:
            return
        scaled = self._full_pixmap.scaled(
            available,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        # Set minimum size so scroll bars appear when panel is too small
        self.setMinimumSize(1, 1)
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_scaled()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._full_pixmap:
            self._open_zoom_dialog()
        elif event.button() == Qt.MouseButton.RightButton and self._full_pixmap:
            self._save_image()
        else:
            super().mousePressEvent(event)

    def _open_zoom_dialog(self):
        from PyQt6.QtWidgets import QDialog
        dlg = QDialog(self.window())
        dlg.setWindowTitle(f"🔍  {self._title}")
        dlg.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowCloseButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint)
        dlg.setStyleSheet("background:#0a0a0a;")
        dlg.resize(1100, 750)

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Scrollable image area
        scroll = QScrollArea()
        scroll.setStyleSheet(
            "QScrollArea { background:#0a0a0a; border:none; }"
            "QScrollBar:vertical   { background:#1a1a1a; width:10px; }"
            "QScrollBar::handle:vertical   { background:#555; border-radius:5px; }"
            "QScrollBar:horizontal { background:#1a1a1a; height:10px; }"
            "QScrollBar::handle:horizontal { background:#555; border-radius:5px; }"
        )
        scroll.setWidgetResizable(True)

        img_lbl = QLabel()
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_lbl.setStyleSheet(f"background:#0a0a0a; border:2px solid {self._border_color};")
        # Show full-resolution
        screen_geom = self.screen().availableGeometry() if self.screen() else None
        max_w = (screen_geom.width() - 60)  if screen_geom else 1800
        max_h = (screen_geom.height() - 160) if screen_geom else 900
        scaled = self._full_pixmap.scaled(
            max_w, max_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        img_lbl.setPixmap(scaled)
        img_lbl.setMinimumSize(scaled.size())
        scroll.setWidget(img_lbl)
        layout.addWidget(scroll, stretch=1)

        # Bottom bar
        btn_row = QHBoxLayout()
        info_lbl = QLabel(
            f"<span style='color:#888; font-size:10px;'>"
            f"{self._full_pixmap.width()} × {self._full_pixmap.height()} px  |  "
            f"Click image to dismiss  |  Right-click on thumbnail to save</span>")
        info_lbl.setTextFormat(Qt.TextFormat.RichText)
        btn_row.addWidget(info_lbl)
        btn_row.addStretch()

        save_btn = QPushButton("⬇  Save Image")
        save_btn.setStyleSheet(
            f"QPushButton {{ background:#1a3a5c; color:{self._border_color}; "
            f"border:1px solid {self._border_color}; border-radius:4px; "
            f"padding:6px 16px; font-weight:bold; font-size:10px; }}"
            f"QPushButton:hover {{ background:#1e4a72; }}")
        save_btn.clicked.connect(lambda: self._save_image(dlg))
        btn_row.addWidget(save_btn)

        close_btn = QPushButton("✕  Close")
        close_btn.setStyleSheet(
            "QPushButton { background:#2a2a2a; color:#aaa; border:1px solid #555; "
            "border-radius:4px; padding:6px 14px; font-size:10px; }"
            "QPushButton:hover { background:#3a3a3a; }")
        close_btn.clicked.connect(dlg.close)
        btn_row.addWidget(close_btn)

        layout.addLayout(btn_row)
        dlg.exec()

    def _save_image(self, parent_widget=None):
        if not self._full_pixmap:
            return
        safe_title = self._title.replace(" ", "_").replace("/", "-")
        path, _ = QFileDialog.getSaveFileName(
            parent_widget or self.window(),
            f"Save — {self._title}",
            f"{safe_title}.png",
            "PNG Image (*.png);;JPEG Image (*.jpg);;BMP Image (*.bmp)")
        if path:
            self._full_pixmap.save(path)


# ============================================================================
# EVALUATION WORKER
# ============================================================================

class EvaluationWorker(QThread):
    progress       = pyqtSignal(int, int, str)
    frame_processed = pyqtSignal(int, dict)
    finished       = pyqtSignal(dict)
    error          = pyqtSignal(str)

    def __init__(self, model_path, image_folder, coco_data, conf_thresh, iou_thresh,
                 match_thresh, pck_thresh):
        super().__init__()
        self.model_path    = model_path
        self.image_folder  = Path(image_folder)
        self.coco_data     = coco_data
        self.conf_thresh   = conf_thresh
        self.iou_thresh    = iou_thresh
        self.match_thresh  = match_thresh
        self.pck_thresholds = pck_thresh   # list of px thresholds e.g. [5,10,15]
        self.is_running    = True

    def run(self):
        try:
            model  = YOLO(self.model_path)
            images = sorted(self.coco_data['images'], key=lambda x: x['file_name'])
            total  = len(images)

            all_metrics    = {k: {'errors': [], 'count': 0, 'tp': 0, 'fp': 0, 'fn': 0} for k in range(9)}
            frame_predictions = {}

            for idx, img_info in enumerate(images):
                if not self.is_running:
                    break
                self.progress.emit(idx+1, total, img_info['file_name'])

                img_path = self.image_folder / img_info['file_name']
                if not img_path.exists():
                    continue

                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue

                results = model(frame, verbose=False,
                                conf=self.conf_thresh, iou=self.iou_thresh)

                predictions = []
                if results and len(results[0].keypoints) > 0:
                    for det_idx in range(len(results[0].keypoints.xy)):
                        kp_xy   = results[0].keypoints.xy[det_idx].cpu().numpy()
                        kp_conf = (results[0].keypoints.conf[det_idx].cpu().numpy()
                                   if results[0].keypoints.conf is not None else np.ones(9))
                        kp_flat = []
                        for j in range(9):
                            if j < len(kp_xy):
                                kp_flat.extend([float(kp_xy[j][0]), float(kp_xy[j][1]), float(kp_conf[j])])
                            else:
                                kp_flat.extend([0.0, 0.0, 0.0])
                        predictions.append({'keypoints': kp_flat})

                gt_anns = [ann for ann in self.coco_data['annotations']
                           if ann['image_id'] == img_info['id']]
                gt_reordered = []
                for gt in gt_anns:
                    gt_copy = gt.copy()
                    gt_copy['keypoints'] = reorder_keypoints_roboflow_to_pipeline(gt['keypoints'])
                    gt_reordered.append(gt_copy)

                # FILTER: Only keep predictions that match a GT-annotated bee
                predictions = filter_predictions_to_gt_bees(gt_reordered, predictions)

                gt_kps   = extract_all_keypoints_eval(gt_reordered, 'gt')
                pred_kps = extract_all_keypoints_eval(predictions, 'pred')
                matches, unmatched_gt, unmatched_pred = match_keypoints_spatially_eval(
                    gt_kps, pred_kps, threshold=self.match_thresh)

                frame_metrics = {k: {'tp':0,'fp':0,'fn':0,'errors':[]} for k in range(9)}
                for gt_kp, pred_kp, dist in matches:
                    k = gt_kp['index']
                    frame_metrics[k]['errors'].append(dist)
                    frame_metrics[k]['tp'] += 1
                    all_metrics[k]['errors'].append(dist)
                    all_metrics[k]['tp'] += 1
                    all_metrics[k]['count'] += 1
                for gt_kp in unmatched_gt:
                    k = gt_kp['index']
                    frame_metrics[k]['fn'] += 1
                    all_metrics[k]['fn'] += 1
                for pred_kp in unmatched_pred:
                    k = pred_kp['index']
                    frame_metrics[k]['fp'] += 1
                    all_metrics[k]['fp'] += 1

                # ── Build 3 visualizations ──────────────────────────────────

                # 1. GT only (green keypoints + labels on clean frame)
                gt_img = frame.copy()
                cv2.putText(gt_img, "GROUND TRUTH (COCO)", (6, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                for gt_kp in gt_kps:
                    cx, cy = int(gt_kp['x']), int(gt_kp['y'])
                    cv2.circle(gt_img, (cx, cy), 5, (0, 255, 0), -1)
                    cv2.circle(gt_img, (cx, cy), 6, (0, 180, 0), 1)
                    cv2.putText(gt_img, str(gt_kp['index']),
                                (cx + 7, cy - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)

                # 2. Predictions only (blue keypoints + labels on clean frame)
                pred_img = frame.copy()
                cv2.putText(pred_img, "PREDICTED (.pt)", (6, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 140, 255), 1, cv2.LINE_AA)
                for pred_kp in pred_kps:
                    cx, cy = int(pred_kp['x']), int(pred_kp['y'])
                    cv2.circle(pred_img, (cx, cy), 5, (255, 80, 0), -1)
                    cv2.circle(pred_img, (cx, cy), 6, (200, 60, 0), 1)
                    cv2.putText(pred_img, str(pred_kp['index']),
                                (cx + 7, cy - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 140, 80), 1, cv2.LINE_AA)

                # 3. Overlay: GT + Pred + match lines
                vis_img = frame.copy()
                cv2.putText(vis_img, "COMPARISON (GT+Pred+Match)", (6, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                for gt_kp in gt_kps:
                    cv2.circle(vis_img, (int(gt_kp['x']), int(gt_kp['y'])), 5, (0, 255, 0), -1)
                    cv2.putText(vis_img, str(gt_kp['index']),
                                (int(gt_kp['x'])+7, int(gt_kp['y'])-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)
                for pred_kp in pred_kps:
                    cv2.circle(vis_img, (int(pred_kp['x']), int(pred_kp['y'])), 5, (255, 80, 0), -1)
                    cv2.putText(vis_img, str(pred_kp['index']),
                                (int(pred_kp['x'])+7, int(pred_kp['y'])+14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 140, 80), 1, cv2.LINE_AA)
                for gt_kp, pred_kp, dist in matches:
                    cv2.line(vis_img,
                             (int(gt_kp['x']), int(gt_kp['y'])),
                             (int(pred_kp['x']), int(pred_kp['y'])),
                             (0, 255, 255), 1)

                frame_data = {
                    'file_name': img_info['file_name'],
                    'predictions': predictions,
                    'gt_anns': gt_reordered,
                    'frame_idx': idx,
                    'gt_image':   gt_img,
                    'pred_image': pred_img,
                    'vis_image':  vis_img,
                    'matches': matches,
                    'frame_metrics': frame_metrics,
                    'gt_kps': gt_kps,
                    'pred_kps': pred_kps
                }
                frame_predictions[img_info['id']] = frame_data
                self.frame_processed.emit(idx, frame_data)

            # Final metrics
            final_metrics = {}
            all_errors_by_kp = {}
            for k in range(9):
                tp  = all_metrics[k]['tp']
                fp  = all_metrics[k]['fp']
                fn  = all_metrics[k]['fn']
                prec = tp / (tp+fp) if (tp+fp) > 0 else 0
                rec  = tp / (tp+fn) if (tp+fn) > 0 else 0
                f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
                errs = all_metrics[k]['errors']
                mean_err = float(np.mean(errs)) if errs else 0.0
                std_err  = float(np.std(errs))  if errs else 0.0
                all_errors_by_kp[k] = errs
                final_metrics[k] = {
                    'mean_error': mean_err, 'std_error': std_err,
                    'precision': prec, 'recall': rec, 'f1': f1,
                    'tp': tp, 'fp': fp, 'fn': fn,
                    'count': all_metrics[k]['count']
                }

            self.finished.emit({
                'metrics': final_metrics,
                'all_errors_by_kp': all_errors_by_kp,
                'frame_predictions': frame_predictions,
                'total_frames': total
            })
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")

    def stop(self):
        self.is_running = False


# ============================================================================
# PIPELINE EVALUATION WORKER  (Stage 1 YOLO + Stage 2 morphological refinement)
# ============================================================================

def _draw_pipeline_frame(frame, all_bee_data, inf_worker):
    """
    Render a frame with full-pipeline keypoints + ROI triangles (thin 1px lines).
    all_bee_data: {bee_id: (keypoints_list, box)}
    """
    vis = frame.copy()
    cv2.putText(vis, "FULL PIPELINE (2-Stage)", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 80, 255), 1, cv2.LINE_AA)

    for bee_id, (keypoints, box) in all_bee_data.items():
        hue = (bee_id * 40) % 180
        cb = cv2.cvtColor(np.uint8([[[hue, 230, 210]]]), cv2.COLOR_HSV2BGR)[0][0]
        color = (int(cb[0]), int(cb[1]), int(cb[2]))

        # ── ROI polygons (thin 1px) ──────────────────────────────────────────
        if box is not None and len(keypoints) > 0 and keypoints[0][2] > 0.1:
            try:
                result = inf_worker.get_angled_roi_lines(keypoints, box, bee_id)
                if len(result) == 6:
                    _, _, _, _, left_roi, right_roi = result
                    if left_roi is not None and len(left_roi) >= 3:
                        cv2.polylines(vis, [np.array(left_roi, np.int32)], True, (0, 200, 200), 1, cv2.LINE_AA)
                    if right_roi is not None and len(right_roi) >= 3:
                        cv2.polylines(vis, [np.array(right_roi, np.int32)], True, (0, 200, 200), 1, cv2.LINE_AA)
                else:
                    head_pos, center_end, left_end, right_end = result[:4]
                    pts_l = np.array([head_pos, left_end, center_end], np.int32)
                    cv2.polylines(vis, [pts_l], True, (0, 200, 200), 1, cv2.LINE_AA)
                    pts_r = np.array([head_pos, right_end, center_end], np.int32)
                    cv2.polylines(vis, [pts_r], True, (0, 200, 200), 1, cv2.LINE_AA)
            except Exception:
                pass

        # ── Skeleton connections ──────────────────────────────────────────────
        SKEL = [(0,1),(1,2),(2,3),(3,4),(0,7),(7,8),(0,5),(5,6)]
        valid = {}
        for idx, (x, y, c) in enumerate(keypoints):
            if c > 0.1:
                valid[idx] = (int(x), int(y))
        for s, e in SKEL:
            if s in valid and e in valid:
                cv2.line(vis, valid[s], valid[e], color, 1, cv2.LINE_AA)

        # ── Keypoints ─────────────────────────────────────────────────────────
        for idx, (x, y, c) in enumerate(keypoints):
            if c > 0.1:
                cx, cy = int(x), int(y)
                radius = 5 if idx in [5, 6, 7, 8] else 4
                cv2.circle(vis, (cx, cy), radius, color, -1)
                cv2.circle(vis, (cx, cy), radius+1, (255, 255, 255), 1)
                cv2.putText(vis, str(idx), (cx+6, cy-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)

    return vis


def _pipeline_keypoints_to_eval_format(all_bee_data):
    """Convert {bee_id: (keypoints_list, box)} to COCO-style flat annotation list."""
    result = []
    for bee_id, (keypoints, box) in all_bee_data.items():
        kp_flat = []
        for (x, y, c) in keypoints:
            kp_flat.extend([float(x), float(y), float(c)])
        result.append({'keypoints': kp_flat})
    return result


class PipelineEvaluationWorker(QThread):
    """
    Runs the FULL two-stage pipeline on each evaluation image:
      Stage 1: YOLO inference + tracking
      Stage 2: Morphological antenna refinement
    Emits per-frame data and cumulative metrics, parallel to EvaluationWorker.
    """
    progress        = pyqtSignal(int, int, str)
    frame_processed = pyqtSignal(int, dict)
    finished        = pyqtSignal(dict)
    error           = pyqtSignal(str)

    def __init__(self, model_path, image_folder, coco_data,
                 conf_thresh, iou_thresh, match_thresh, pck_thresh, detection_thresholds):
        super().__init__()
        self.model_path           = model_path
        self.image_folder         = Path(image_folder)
        self.coco_data            = coco_data
        self.conf_thresh          = conf_thresh
        self.iou_thresh           = iou_thresh
        self.match_thresh         = match_thresh
        self.pck_thresholds       = pck_thresh
        self.detection_thresholds = detection_thresholds
        self.is_running           = True

    def run(self):
        try:
            # Release any GPU memory held by the previous YOLO worker before loading a new model
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
            inf = InferenceWorker.__new__(InferenceWorker)
            QObject.__init__(inf)   # ← must be called or Qt segfaults on GC
            inf.keypoint_history        = {}
            inf.history_size            = self.detection_thresholds.get('smoothing_frames', 3)
            inf.confidence_threshold    = 0.1
            inf.direction_history       = {}
            inf.direction_history_size  = 7
            inf.bee_class               = {}
            inf.detection_thresholds    = self.detection_thresholds
            inf.roi_thickness           = 1
            inf.binary_threshold        = self.detection_thresholds.get('binary_threshold', 0.0)
            inf.antenna_tracked_positions = {}
            inf.antenna_lock_frames     = {}
            inf.antenna_lost_threshold  = 15
            inf.antenna_tracking_radius = 25
            inf.use_cuda                = TORCH_CUDA_AVAILABLE
            inf.cached_darkness_mask    = None
            inf.cached_darkness_frame_id = -1
            inf.cached_antenna_maps     = None
            inf.cached_antenna_maps_frame_id = -1
            inf.cached_frame_components = None
            inf.cached_components_frame_id = -1
            inf.cached_components_binary_hash = None
            inf.last_valid_detections   = {}
            inf.last_valid_tips         = {}
            inf.max_bees_per_frame      = 50
            inf.antenna_state           = {}
            inf.frame_counter           = 0
            inf.antenna_contact_threshold = 10
            # Motion / optical flow / tracking subsystems
            inf.motion_detector         = MotionBasedAntennaDetector(observation_frames=6, motion_threshold=0.6)
            inf.optical_flow_tracker    = OpticalFlowAntennaTracker()
            inf.trophallaxis_detector   = TrophallaxisDetector()
            inf.antenna_tracker         = IndependentAntennaTracker()
            inf.region_tracker          = FourRegionAntennaTracker()
            kernel_size = self.detection_thresholds.get('kernel_size', 5)
            inf.research_processor = MorphologicalPipelineProcessor(kernel_size=kernel_size)
            inf.antenna_pipeline   = AntennaProcessingPipeline()

            model  = YOLO(self.model_path)
            images = sorted(self.coco_data['images'], key=lambda x: x['file_name'])
            total  = len(images)

            all_metrics    = {k: {'errors': [], 'count': 0, 'tp': 0, 'fp': 0, 'fn': 0} for k in range(9)}
            frame_predictions = {}

            for idx, img_info in enumerate(images):
                if not self.is_running:
                    break
                self.progress.emit(idx+1, total, img_info['file_name'])

                img_path = self.image_folder / img_info['file_name']
                if not img_path.exists():
                    continue
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue
                frame      = np.ascontiguousarray(frame)
                frame_gray = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

                # ── Stage 1: YOLO inference ────────────────────────────────────
                results = model(frame, verbose=False,
                                conf=self.conf_thresh, iou=self.iou_thresh)

                all_bee_data = {}
                if results and len(results[0].keypoints) > 0:
                    kp_xy_all   = results[0].keypoints.xy
                    kp_conf_all = results[0].keypoints.conf
                    boxes_all   = results[0].boxes.xyxy if results[0].boxes is not None else None

                    for det_idx in range(len(kp_xy_all)):
                        kp_xy   = kp_xy_all[det_idx].cpu().numpy()
                        kp_conf = (kp_conf_all[det_idx].cpu().numpy()
                                   if kp_conf_all is not None else np.ones(9))
                        box = (boxes_all[det_idx].cpu().numpy()
                               if boxes_all is not None else None)

                        keypoints = []
                        for j in range(9):
                            if j < len(kp_xy):
                                keypoints.append((float(kp_xy[j][0]),
                                                  float(kp_xy[j][1]),
                                                  float(kp_conf[j])))
                            else:
                                keypoints.append((0.0, 0.0, 0.0))

                        bee_id = det_idx
                        # ── Stage 2: Morphological antenna refinement ──────────
                        try:
                            keypoints = inf.smooth_keypoints(bee_id, keypoints)
                            if box is not None:
                                if bee_id not in inf.antenna_tracked_positions:
                                    keypoints = inf.refine_antenna_keypoints_unified(
                                        frame_gray, keypoints, box, bee_id, idx)
                                    keypoints = inf.refine_antenna_keypoints_iterative(
                                        frame_gray, keypoints, box, bee_id, search_depth=12)
                                    inf.antenna_tracked_positions[bee_id] = {
                                        'left':  (keypoints[5][0], keypoints[5][1], keypoints[5][2]),
                                        'right': (keypoints[7][0], keypoints[7][1], keypoints[7][2])
                                    }
                                    inf.antenna_lock_frames[bee_id] = 0
                                else:
                                    keypoints = inf.refine_antenna_keypoints_unified(
                                        frame_gray, keypoints, box, bee_id, idx)
                                    if keypoints[5][2] < 0.4 or keypoints[7][2] < 0.4:
                                        keypoints = inf.place_antenna_keypoints_on_darkest_lines(
                                            frame_gray, keypoints, box, bee_id, idx)
                                        keypoints = inf.refine_antenna_keypoints_iterative(
                                            frame_gray, keypoints, box, bee_id, search_depth=10)
                        except Exception as stage2_err:
                            print(f"[PipelineEval] Stage 2 skipped for det {det_idx}: {stage2_err}")
                        all_bee_data[bee_id] = (keypoints, box)

                # ── GT (load FIRST, needed for filtering) ──────────────────────
                gt_anns = [a for a in self.coco_data['annotations']
                           if a['image_id'] == img_info['id']]
                gt_reordered = []
                for gt in gt_anns:
                    gt_copy = gt.copy()
                    gt_copy['keypoints'] = reorder_keypoints_roboflow_to_pipeline(gt['keypoints'])
                    gt_reordered.append(gt_copy)

                # FILTER: Remove bees that don't correspond to any GT annotation
                # This ensures visualization and metrics only include annotated bees
                if gt_reordered and all_bee_data:
                    pipe_anns_all = _pipeline_keypoints_to_eval_format(all_bee_data)
                    pipe_anns_matched = filter_predictions_to_gt_bees(gt_reordered, pipe_anns_all)
                    
                    # Rebuild all_bee_data with only GT-matched bees
                    matched_bee_ids = set()
                    bee_id_list = list(all_bee_data.keys())
                    for matched_ann in pipe_anns_matched:
                        mc = _bee_centroid_from_flat_keypoints(matched_ann['keypoints'])
                        if mc is None:
                            continue
                        best_id, best_dist = None, float('inf')
                        for bid in bee_id_list:
                            if bid in matched_bee_ids:
                                continue
                            kps, _ = all_bee_data[bid]
                            xs = [k[0] for k in kps if k[2] > 0]
                            ys = [k[1] for k in kps if k[2] > 0]
                            if not xs:
                                continue
                            pc = (np.mean(xs), np.mean(ys))
                            d = np.sqrt((mc[0]-pc[0])**2 + (mc[1]-pc[1])**2)
                            if d < best_dist:
                                best_dist, best_id = d, bid
                        if best_id is not None:
                            matched_bee_ids.add(best_id)
                    
                    all_bee_data = {bid: all_bee_data[bid] for bid in matched_bee_ids}

                # ── Build visualization (only GT-matched bees) ─────────────────
                pipe_vis = _draw_pipeline_frame(frame, all_bee_data, inf)

                # ── Convert filtered output to eval format ─────────────────────
                pipe_anns = _pipeline_keypoints_to_eval_format(all_bee_data)

                gt_kps   = extract_all_keypoints_eval(gt_reordered, 'gt')
                pipe_kps = extract_all_keypoints_eval(pipe_anns, 'pred')
                matches, unmatched_gt, unmatched_pred = match_keypoints_spatially_eval(
                    gt_kps, pipe_kps, threshold=self.match_thresh)

                frame_metrics = {k: {'tp':0,'fp':0,'fn':0,'errors':[]} for k in range(9)}
                for gt_kp, pred_kp, dist in matches:
                    k = gt_kp['index']
                    frame_metrics[k]['errors'].append(dist)
                    frame_metrics[k]['tp'] += 1
                    all_metrics[k]['errors'].append(dist)
                    all_metrics[k]['tp'] += 1
                    all_metrics[k]['count'] += 1
                for gt_kp in unmatched_gt:
                    k = gt_kp['index']
                    frame_metrics[k]['fn'] += 1
                    all_metrics[k]['fn'] += 1
                for pred_kp in unmatched_pred:
                    k = pred_kp['index']
                    frame_metrics[k]['fp'] += 1
                    all_metrics[k]['fp'] += 1

                frame_data = {
                    'file_name':     img_info['file_name'],
                    'frame_idx':     idx,
                    'pipe_image':    pipe_vis,
                    'frame_metrics': frame_metrics,
                    'gt_kps':        gt_kps,
                    'pipe_kps':      pipe_kps,
                    'matches':       matches,
                }
                frame_predictions[img_info['id']] = frame_data
                self.frame_processed.emit(idx, frame_data)

            # ── Final cumulative metrics ───────────────────────────────────────
            final_metrics    = {}
            all_errors_by_kp = {}
            for k in range(9):
                tp   = all_metrics[k]['tp']
                fp   = all_metrics[k]['fp']
                fn   = all_metrics[k]['fn']
                prec = tp / (tp+fp) if (tp+fp) > 0 else 0
                rec  = tp / (tp+fn) if (tp+fn) > 0 else 0
                f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
                errs = all_metrics[k]['errors']
                all_errors_by_kp[k] = errs
                final_metrics[k] = {
                    'mean_error': float(np.mean(errs)) if errs else 0.0,
                    'std_error':  float(np.std(errs))  if errs else 0.0,
                    'precision': prec, 'recall': rec, 'f1': f1,
                    'tp': tp, 'fp': fp, 'fn': fn,
                    'count': all_metrics[k]['count']
                }

            self.finished.emit({
                'metrics':          final_metrics,
                'all_errors_by_kp': all_errors_by_kp,
                'frame_predictions': frame_predictions,
                'total_frames':     total
            })
        except Exception as e:
            import traceback
            self.error.emit(f"Pipeline eval error: {str(e)}\n{traceback.format_exc()}")

    def stop(self):
        self.is_running = False




class EvaluationTab(QWidget):
    """Full-screen evaluation tab: left=frame comparison, right=metrics & charts."""

    _BTN_STYLE = """
        QPushButton {
            background-color: #1a3a5c; color: #00b4d8;
            border: 1px solid #00b4d8; border-radius: 4px;
            padding: 6px 12px; font-weight: bold; font-size: 10px;
        }
        QPushButton:hover { background-color: #1e4a72; }
        QPushButton:disabled { background-color: #2a2a2a; color: #555; border-color: #444; }
    """
    _RUN_STYLE = """
        QPushButton { background: #2e7d32; color: white; border: none;
                      padding: 10px; font-weight: bold; font-size: 11px; border-radius: 4px; }
        QPushButton:hover { background: #388e3c; }
        QPushButton:disabled { background: #2a2a2a; color: #555; }
    """
    _STOP_STYLE = """
        QPushButton { background: #c62828; color: white; border: none;
                      padding: 10px; font-weight: bold; font-size: 11px; border-radius: 4px; }
        QPushButton:disabled { background: #2a2a2a; color: #555; }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_gui       = parent
        self.coco_data        = None
        self.image_folder     = None
        self.eval_model_path  = ""
        self.frame_predictions = {}
        self.current_frame_idx = 0
        self.worker           = None
        self.pipe_worker      = None        # pipeline worker
        self._final_metrics   = {}
        self._all_errors_by_kp = {}
        self._pipe_metrics    = {}          # pipeline cumulative metrics
        self._pipe_errors_by_kp = {}
        self._pipe_frame_predictions = {}   # {frame_idx: frame_data}
        self._current_vis_pixmap = None

        # Threshold defaults (populated by threshold tab widgets)
        self._conf_thresh   = 0.25
        self._iou_thresh    = 0.45
        self._match_thresh  = 50
        self._pck_thresholds = [5, 10, 15]

        self._build_ui()

    # ------------------------------------------------------------------ build
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── TOP CONTROL BAR ──────────────────────────────────────────────────
        ctrl = QFrame()
        ctrl.setStyleSheet("QFrame { background:#2d2d2d; border:1px solid #444; border-radius:6px; }")
        ctrl_layout = QHBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(10, 6, 10, 6)
        ctrl_layout.setSpacing(8)

        def _lbl(text, color="#b0b0b0"):
            l = QLabel(text)
            l.setStyleSheet(f"color:{color}; font-size:10px; border:none;")
            return l

        # Model
        self.eval_model_btn = QPushButton("🤖 Model (.pt)")
        self.eval_model_btn.setStyleSheet(self._BTN_STYLE)
        self.eval_model_btn.clicked.connect(self._browse_model)
        self.eval_model_lbl = _lbl("No model")
        ctrl_layout.addWidget(self.eval_model_btn)
        ctrl_layout.addWidget(self.eval_model_lbl)
        ctrl_layout.addWidget(_lbl("│", "#555"))

        # COCO JSON
        self.coco_btn = QPushButton("📁 COCO JSON")
        self.coco_btn.setStyleSheet(self._BTN_STYLE)
        self.coco_btn.clicked.connect(self._browse_coco)
        self.coco_lbl = _lbl("No JSON")
        ctrl_layout.addWidget(self.coco_btn)
        ctrl_layout.addWidget(self.coco_lbl)
        ctrl_layout.addWidget(_lbl("│", "#555"))

        # Image folder
        self.folder_btn = QPushButton("📂 Images")
        self.folder_btn.setStyleSheet(self._BTN_STYLE)
        self.folder_btn.clicked.connect(self._browse_folder)
        self.folder_lbl = _lbl("No folder")
        ctrl_layout.addWidget(self.folder_btn)
        ctrl_layout.addWidget(self.folder_lbl)
        ctrl_layout.addWidget(_lbl("│", "#555"))

        # ── Threshold quick-status indicator ──
        self.thresh_status_lbl = _lbl("⚙ Conf:0.25 | IoU:0.45 | Match:50px | PCK:5,10,15px", "#ffd700")
        ctrl_layout.addWidget(self.thresh_status_lbl)

        ctrl_layout.addStretch()

        # Run / Stop
        self.run_btn = QPushButton("▶ RUN EVALUATION")
        self.run_btn.setStyleSheet(self._RUN_STYLE)
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QPushButton("⏹ STOP")
        self.stop_btn.setStyleSheet(self._STOP_STYLE)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        ctrl_layout.addWidget(self.run_btn)
        ctrl_layout.addWidget(self.stop_btn)

        root.addWidget(ctrl)

        # ── PROGRESS ─────────────────────────────────────────────────────────
        prog_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar { background:#1a1a1a; border:1px solid #444; border-radius:4px;
                           color:#e0e0e0; text-align:center; height:16px; font-size:9px; }
            QProgressBar::chunk { background:#00b4d8; border-radius:3px; }
        """)
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet("color:#888; font-size:10px; min-width:200px;")
        prog_row.addWidget(self.progress_bar)
        prog_row.addWidget(self.status_lbl)
        root.addLayout(prog_row)

        # ── MAIN CONTENT: left=frame, right=metrics ───────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet("QSplitter::handle { background:#444; }")

        # ── LEFT: Frame comparison ────────────────────────────────────────────
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        # Title bar + download
        frame_title_row = QHBoxLayout()
        frame_title_lbl = QLabel("📷 Frame Comparison")
        frame_title_lbl.setStyleSheet("color:#00b4d8; font-weight:bold; font-size:10px;")
        frame_title_row.addWidget(frame_title_lbl)
        frame_title_row.addStretch()
        self.dl_frame_btn = QPushButton("⬇ Save Comparison")
        self.dl_frame_btn.setStyleSheet(self._BTN_STYLE)
        self.dl_frame_btn.setEnabled(False)
        self.dl_frame_btn.clicked.connect(self._download_frame)
        frame_title_row.addWidget(self.dl_frame_btn)
        left_layout.addLayout(frame_title_row)

        # ── Three panels side by side ─────────────────────────────────────────
        panels_widget = QWidget()
        panels_layout = QHBoxLayout(panels_widget)
        panels_layout.setContentsMargins(0, 0, 0, 0)
        panels_layout.setSpacing(4)

        def _make_panel(title, border_color, title_bg, img_key):
            pane = QWidget()
            pane_layout = QVBoxLayout(pane)
            pane_layout.setContentsMargins(0, 0, 0, 0)
            pane_layout.setSpacing(0)

            # Title bar with download button
            header = QWidget()
            header.setStyleSheet(
                f"background:{title_bg}; border:1px solid {border_color}; border-bottom:none;")
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(6, 2, 4, 2)
            header_layout.setSpacing(4)
            title_lbl = QLabel(title)
            title_lbl.setStyleSheet(
                f"color:{border_color}; font-size:9px; font-weight:bold; "
                f"background:transparent; border:none;")
            header_layout.addWidget(title_lbl)
            header_layout.addStretch()
            hint_lbl = QLabel("🔍 click to zoom")
            hint_lbl.setStyleSheet("color:#555; font-size:8px; background:transparent; border:none;")
            header_layout.addWidget(hint_lbl)

            dl_btn = QPushButton("⬇")
            dl_btn.setFixedSize(22, 18)
            dl_btn.setToolTip("Save this image")
            dl_btn.setStyleSheet(
                f"QPushButton {{ background:transparent; color:{border_color}; "
                f"border:1px solid {border_color}; border-radius:3px; font-size:9px; font-weight:bold; }}"
                f"QPushButton:hover {{ background:{title_bg}; }}")
            header_layout.addWidget(dl_btn)

            img_lbl = ClickableImageLabel(title=title, border_color=border_color)
            dl_btn.clicked.connect(img_lbl._save_image)

            # Wrap image in a scroll area so it never overflows the panel
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(img_lbl)
            scroll.setStyleSheet(
                f"QScrollArea {{ background:#0a0a0a; border:1px solid {border_color}; }}"
                "QScrollBar:vertical   { background:#1a1a1a; width:8px; border-radius:4px; }"
                "QScrollBar::handle:vertical   { background:#444; border-radius:4px; min-height:20px; }"
                "QScrollBar::handle:vertical:hover   { background:#00b4d8; }"
                "QScrollBar:horizontal { background:#1a1a1a; height:8px; border-radius:4px; }"
                "QScrollBar::handle:horizontal { background:#444; border-radius:4px; min-width:20px; }"
                "QScrollBar::handle:horizontal:hover { background:#00b4d8; }"
                "QScrollBar::add-line, QScrollBar::sub-line { width:0; height:0; }"
            )

            pane_layout.addWidget(header)
            pane_layout.addWidget(scroll, stretch=1)
            return pane, img_lbl

        panel_gt,   self.gt_display   = _make_panel(
            "🟢 Ground Truth (COCO JSON)", "#00cc44", "#0a2a14", "gt_image")
        panel_pred, self.pred_display = _make_panel(
            "🔵 Predicted (.pt model)",    "#4da6ff", "#0a1a2a", "pred_image")
        panel_comp, self.frame_display = _make_panel(
            "🟡 Comparison (GT+Pred+Match)", "#ffd700", "#1a1a00", "vis_image")
        panel_pipe, self.pipe_display = _make_panel(
            "🟣 Full Pipeline (2-Stage+ROI)", "#cc66ff", "#1a0a2a", "pipe_image")

        panels_layout.addWidget(panel_gt,   stretch=1)
        panels_layout.addWidget(panel_pred, stretch=1)
        panels_layout.addWidget(panel_comp, stretch=1)
        panels_layout.addWidget(panel_pipe, stretch=1)

        left_layout.addWidget(panels_widget, stretch=1)

        # Nav bar
        nav_row = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.setStyleSheet(self._BTN_STYLE); self.prev_btn.setEnabled(False)
        self.prev_btn.clicked.connect(self._prev_frame)
        self.frame_counter_lbl = QLabel("— / —")
        self.frame_counter_lbl.setStyleSheet("color:#888; font-size:10px;")
        self.frame_counter_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.setStyleSheet(self._BTN_STYLE); self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self._next_frame)
        nav_row.addWidget(self.prev_btn)
        nav_row.addWidget(self.frame_counter_lbl, stretch=1)
        nav_row.addWidget(self.next_btn)
        left_layout.addLayout(nav_row)

        splitter.addWidget(left_widget)

        # ── RIGHT: Metrics & charts ───────────────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        metrics_tabs = QTabWidget()
        metrics_tabs.setStyleSheet("""
            QTabBar::tab { padding:5px 10px; font-size:9px; font-weight:bold; }
        """)

        _SCROLL_STYLE = (
            "QScrollArea { background:#1a1a1a; border:none; }"
            "QScrollBar:vertical { background:#2a2a2a; width:8px; border-radius:4px; }"
            "QScrollBar::handle:vertical { background:#444; border-radius:4px; }"
            "QScrollBar::handle:vertical:hover { background:#00b4d8; }"
            "QScrollBar:horizontal { background:#2a2a2a; height:8px; border-radius:4px; }"
            "QScrollBar::handle:horizontal { background:#444; border-radius:4px; }"
            "QScrollBar::handle:horizontal:hover { background:#00b4d8; }"
        )

        _TBL_YOLO = """
            QTableWidget { background:#0a1a2a; color:#e0e0e0; gridline-color:#2a3a4a;
                           font-size:10px; border:1px solid #4da6ff; }
            QHeaderView::section { background:#0d2540; color:#4da6ff; border:1px solid #4da6ff;
                                   padding:3px; font-weight:bold; font-size:9px; }
            QTableWidget::item:selected { background:#1a3a5c; }
        """
        _TBL_PIPE = """
            QTableWidget { background:#1a0a2a; color:#e0e0e0; gridline-color:#3a2a4a;
                           font-size:10px; border:1px solid #cc66ff; }
            QHeaderView::section { background:#250d40; color:#cc66ff; border:1px solid #cc66ff;
                                   padding:3px; font-weight:bold; font-size:9px; }
            QTableWidget::item:selected { background:#3a1a5c; }
        """

        def _make_dual_tab(yolo_tbl_attr, pipe_tbl_attr, yolo_cols, pipe_cols, title_yolo, title_pipe):
            """Build a tab with YOLO table on top and Pipeline table below."""
            tab = QWidget()
            layout = QVBoxLayout(tab)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(6)

            def _make_section(hdr_text, hdr_color, hdr_bg, border, cols, tbl_style, attr_name):
                grp = QWidget()
                grp.setStyleSheet(f"QWidget{{border:1px solid {border}; border-radius:3px;}}")
                vb = QVBoxLayout(grp)
                vb.setContentsMargins(0, 0, 0, 0)
                vb.setSpacing(0)
                hdr = QLabel(hdr_text)
                hdr.setStyleSheet(
                    f"background:{hdr_bg}; color:{hdr_color}; font-size:9px; "
                    f"font-weight:bold; padding:3px 6px; border:none; border-bottom:1px solid {border};")
                vb.addWidget(hdr)
                tbl = QTableWidget()
                tbl.setColumnCount(len(cols))
                tbl.setHorizontalHeaderLabels(cols)
                tbl.setRowCount(9)
                tbl.verticalHeader().setVisible(False)
                tbl.setStyleSheet(tbl_style)
                tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
                for i in range(9):
                    tbl.setItem(i, 0, QTableWidgetItem(str(i)))
                vb.addWidget(tbl)
                setattr(self, attr_name, tbl)
                return grp

            layout.addWidget(_make_section(
                title_yolo, "#4da6ff", "#0d2540", "#4da6ff",
                yolo_cols, _TBL_YOLO, yolo_tbl_attr), stretch=1)
            layout.addWidget(_make_section(
                title_pipe, "#cc66ff", "#250d40", "#cc66ff",
                pipe_cols, _TBL_PIPE, pipe_tbl_attr), stretch=1)
            return tab

        # ── Tab A: Cumulative ─────────────────────────────────────────────────
        tab_cum = QWidget()
        tab_cum_layout = QVBoxLayout(tab_cum)
        tab_cum_layout.setContentsMargins(4, 4, 4, 4)
        tab_cum_layout.setSpacing(4)

        cum_hdr_row = QHBoxLayout()
        cum_hdr_lbl = QLabel("📊 Cumulative Metrics — YOLO vs Full Pipeline")
        cum_hdr_lbl.setStyleSheet("color:#00b4d8; font-weight:bold; font-size:10px;")
        cum_hdr_row.addWidget(cum_hdr_lbl)
        cum_hdr_row.addStretch()
        self.dl_cum_btn = QPushButton("⬇ Export CSV")
        self.dl_cum_btn.setStyleSheet(self._BTN_STYLE)
        self.dl_cum_btn.setEnabled(False)
        self.dl_cum_btn.clicked.connect(self._download_cum_csv)
        cum_hdr_row.addWidget(self.dl_cum_btn)
        tab_cum_layout.addLayout(cum_hdr_row)

        _CUM_COLS = ['KP','MeanErr','Std','TP','FP','FN','Prec%','Rec%','F1%']

        # YOLO section
        yolo_cum_grp = QWidget()
        yolo_cum_grp.setStyleSheet("QWidget{border:1px solid #4da6ff; border-radius:3px;}")
        yolo_cum_vb = QVBoxLayout(yolo_cum_grp)
        yolo_cum_vb.setContentsMargins(0,0,0,0); yolo_cum_vb.setSpacing(0)
        yolo_cum_hdr = QLabel("🔵 YOLO Raw (.pt) — Stage 1 only")
        yolo_cum_hdr.setStyleSheet("background:#0d2540; color:#4da6ff; font-size:9px; font-weight:bold; padding:3px 6px; border:none; border-bottom:1px solid #4da6ff;")
        yolo_cum_vb.addWidget(yolo_cum_hdr)
        self.cum_table = QTableWidget()
        self.cum_table.setColumnCount(9)
        self.cum_table.setHorizontalHeaderLabels(_CUM_COLS)
        self.cum_table.setRowCount(9)
        self.cum_table.verticalHeader().setVisible(False)
        self.cum_table.setStyleSheet(_TBL_YOLO)
        self.cum_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for i in range(9): self.cum_table.setItem(i, 0, QTableWidgetItem(str(i)))
        yolo_cum_vb.addWidget(self.cum_table)
        self.cum_summary_lbl = QLabel("")
        self.cum_summary_lbl.setStyleSheet("color:#4da6ff; font-size:9px; padding:2px 6px; background:#0a1a2a;")
        self.cum_summary_lbl.setWordWrap(True)
        yolo_cum_vb.addWidget(self.cum_summary_lbl)
        tab_cum_layout.addWidget(yolo_cum_grp, stretch=1)

        # Pipeline section
        pipe_cum_grp = QWidget()
        pipe_cum_grp.setStyleSheet("QWidget{border:1px solid #cc66ff; border-radius:3px;}")
        pipe_cum_vb = QVBoxLayout(pipe_cum_grp)
        pipe_cum_vb.setContentsMargins(0,0,0,0); pipe_cum_vb.setSpacing(0)
        pipe_cum_hdr = QLabel("🟣 Full Pipeline (2-Stage+ROI) — after morphological refinement")
        pipe_cum_hdr.setStyleSheet("background:#250d40; color:#cc66ff; font-size:9px; font-weight:bold; padding:3px 6px; border:none; border-bottom:1px solid #cc66ff;")
        pipe_cum_vb.addWidget(pipe_cum_hdr)
        self.pipe_cum_table = QTableWidget()
        self.pipe_cum_table.setColumnCount(9)
        self.pipe_cum_table.setHorizontalHeaderLabels(_CUM_COLS)
        self.pipe_cum_table.setRowCount(9)
        self.pipe_cum_table.verticalHeader().setVisible(False)
        self.pipe_cum_table.setStyleSheet(_TBL_PIPE)
        self.pipe_cum_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for i in range(9): self.pipe_cum_table.setItem(i, 0, QTableWidgetItem(str(i)))
        pipe_cum_vb.addWidget(self.pipe_cum_table)
        self.pipe_summary_lbl = QLabel("")
        self.pipe_summary_lbl.setStyleSheet("color:#cc66ff; font-size:9px; padding:2px 6px; background:#1a0a2a;")
        self.pipe_summary_lbl.setWordWrap(True)
        pipe_cum_vb.addWidget(self.pipe_summary_lbl)
        tab_cum_layout.addWidget(pipe_cum_grp, stretch=1)

        metrics_tabs.addTab(tab_cum, "📊 Cumulative")

        # ── Tab B: Frame metrics ──────────────────────────────────────────────
        tab_frame = QWidget()
        tab_frame_layout = QVBoxLayout(tab_frame)
        tab_frame_layout.setContentsMargins(4, 4, 4, 4)
        tab_frame_layout.setSpacing(4)

        fr_hdr_row = QHBoxLayout()
        fr_hdr_lbl = QLabel("🎯 Current Frame — YOLO vs Pipeline")
        fr_hdr_lbl.setStyleSheet("color:#00b4d8; font-weight:bold; font-size:10px;")
        fr_hdr_row.addWidget(fr_hdr_lbl)
        fr_hdr_row.addStretch()
        self.dl_frame_csv_btn = QPushButton("⬇ Export CSV")
        self.dl_frame_csv_btn.setStyleSheet(self._BTN_STYLE)
        self.dl_frame_csv_btn.setEnabled(False)
        self.dl_frame_csv_btn.clicked.connect(self._download_frame_csv)
        fr_hdr_row.addWidget(self.dl_frame_csv_btn)
        tab_frame_layout.addLayout(fr_hdr_row)

        _FR_COLS = ['KP','TP','FP','FN','Avg Err(px)']

        yolo_fr_grp = QWidget()
        yolo_fr_grp.setStyleSheet("QWidget{border:1px solid #4da6ff; border-radius:3px;}")
        yolo_fr_vb = QVBoxLayout(yolo_fr_grp)
        yolo_fr_vb.setContentsMargins(0,0,0,0); yolo_fr_vb.setSpacing(0)
        yolo_fr_hdr = QLabel("🔵 YOLO Raw — per frame")
        yolo_fr_hdr.setStyleSheet("background:#0d2540; color:#4da6ff; font-size:9px; font-weight:bold; padding:3px 6px; border:none; border-bottom:1px solid #4da6ff;")
        yolo_fr_vb.addWidget(yolo_fr_hdr)
        self.frame_table = QTableWidget()
        self.frame_table.setColumnCount(5)
        self.frame_table.setHorizontalHeaderLabels(_FR_COLS)
        self.frame_table.setRowCount(9)
        self.frame_table.verticalHeader().setVisible(False)
        self.frame_table.setStyleSheet(_TBL_YOLO)
        self.frame_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for i in range(9): self.frame_table.setItem(i, 0, QTableWidgetItem(str(i)))
        yolo_fr_vb.addWidget(self.frame_table)
        tab_frame_layout.addWidget(yolo_fr_grp, stretch=1)

        pipe_fr_grp = QWidget()
        pipe_fr_grp.setStyleSheet("QWidget{border:1px solid #cc66ff; border-radius:3px;}")
        pipe_fr_vb = QVBoxLayout(pipe_fr_grp)
        pipe_fr_vb.setContentsMargins(0,0,0,0); pipe_fr_vb.setSpacing(0)
        pipe_fr_hdr = QLabel("🟣 Full Pipeline — per frame")
        pipe_fr_hdr.setStyleSheet("background:#250d40; color:#cc66ff; font-size:9px; font-weight:bold; padding:3px 6px; border:none; border-bottom:1px solid #cc66ff;")
        pipe_fr_vb.addWidget(pipe_fr_hdr)
        self.pipe_frame_table = QTableWidget()
        self.pipe_frame_table.setColumnCount(5)
        self.pipe_frame_table.setHorizontalHeaderLabels(_FR_COLS)
        self.pipe_frame_table.setRowCount(9)
        self.pipe_frame_table.verticalHeader().setVisible(False)
        self.pipe_frame_table.setStyleSheet(_TBL_PIPE)
        self.pipe_frame_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for i in range(9): self.pipe_frame_table.setItem(i, 0, QTableWidgetItem(str(i)))
        pipe_fr_vb.addWidget(self.pipe_frame_table)
        tab_frame_layout.addWidget(pipe_fr_grp, stretch=1)

        metrics_tabs.addTab(tab_frame, "🎯 Frame")

        def _make_dual_chart_tab(yolo_attr, pipe_attr, tab_title):
            """Chart tab with YOLO chart on left, Pipeline chart on right."""
            tab = QWidget()
            tl = QVBoxLayout(tab)
            tl.setContentsMargins(0, 0, 0, 0)
            tl.setSpacing(0)
            hdr = QLabel(f"  🔵 YOLO Raw  (left)     🟣 Full Pipeline  (right)  \u2014  {tab_title}")
            hdr.setStyleSheet("background:#1a1a1a; color:#666; font-size:9px; padding:2px 6px; border-bottom:1px solid #333;")
            tl.addWidget(hdr)
            split = QSplitter(Qt.Orientation.Horizontal)
            split.setHandleWidth(4)
            split.setStyleSheet("QSplitter::handle { background:#333; }")

            def _half(attr, border, bg):
                half = QWidget()
                vb = QVBoxLayout(half)
                vb.setContentsMargins(0, 0, 0, 0)
                sc = QScrollArea()
                sc.setWidgetResizable(True)
                sc.setStyleSheet(
                    f"QScrollArea{{background:{bg};border:1px solid {border};}}"
                    "QScrollBar:vertical{background:#1a1a1a;width:8px;border-radius:4px;}"
                    "QScrollBar::handle:vertical{background:#444;border-radius:4px;}"
                    "QScrollBar::handle:vertical:hover{background:#00b4d8;}"
                    "QScrollBar:horizontal{background:#1a1a1a;height:8px;border-radius:4px;}"
                    "QScrollBar::handle:horizontal{background:#444;border-radius:4px;}"
                )
                inner = QWidget()
                inner_vb = QVBoxLayout(inner)
                inner_vb.setContentsMargins(0, 0, 0, 0)
                lbl = QLabel("Run evaluation to generate chart")
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setStyleSheet(f"background:{bg}; color:#555; border:none;")
                lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                inner_vb.addWidget(lbl)
                sc.setWidget(inner)
                vb.addWidget(sc)
                setattr(self, attr, lbl)
                return half

            split.addWidget(_half(yolo_attr, "#4da6ff", "#050d14"))
            split.addWidget(_half(pipe_attr, "#cc66ff", "#0d0514"))
            tl.addWidget(split, stretch=1)
            return tab

        metrics_tabs.addTab(_make_dual_chart_tab("errbar_lbl", "pipe_errbar_lbl", "Error Bar"), "\U0001f4c9 Error Bar")
        metrics_tabs.addTab(_make_dual_chart_tab("pck_lbl",    "pipe_pck_lbl",    "PCK Curve"), "\U0001f4c8 PCK Curve")
        metrics_tabs.addTab(_make_dual_chart_tab("det_lbl",    "pipe_det_lbl",    "Detection"), "\U0001f3af Detection")
        metrics_tabs.addTab(_make_dual_chart_tab("corr_lbl",   "pipe_corr_lbl",   "Correlation"), "\U0001f517 Correlation")
        metrics_tabs.addTab(_make_dual_chart_tab("sev_lbl",    "pipe_sev_lbl",    "Severity"), "\U0001f321 Severity")

        # ── Tab H: Thresholds ─────────────────────────────────────────────────
        tab_thresh = QWidget()
        tab_thresh_outer = QVBoxLayout(tab_thresh)
        tab_thresh_outer.setContentsMargins(0,0,0,0)

        # Wrap content in scroll area so nothing gets cut off
        thresh_scroll = QScrollArea()
        thresh_scroll.setWidgetResizable(True)
        thresh_scroll.setStyleSheet("QScrollArea { background:#1a1a1a; border:none; }"
                                    "QScrollBar:vertical { background:#2a2a2a; width:8px; border-radius:4px; }"
                                    "QScrollBar::handle:vertical { background:#444; border-radius:4px; }"
                                    "QScrollBar::handle:vertical:hover { background:#00b4d8; }")

        thresh_content = QWidget()
        thresh_content.setStyleSheet("background:#1a1a1a;")
        tab_thresh_layout = QVBoxLayout(thresh_content)
        tab_thresh_layout.setSpacing(6)
        tab_thresh_layout.setContentsMargins(12, 10, 12, 10)

        _SLIDER_STYLE = """
            QSlider::groove:horizontal {
                background: #3d3d3d;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00b4d8;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #48cae4;
            }
            QSlider::sub-page:horizontal {
                background: #005f8a;
                border-radius: 3px;
            }
        """

        def _thresh_header(icon, title):
            lbl = QLabel(f"{icon} {title}")
            lbl.setStyleSheet(
                "color:#00b4d8; font-weight:bold; font-size:10px; "
                "border-bottom:1px solid #00b4d8; padding-bottom:3px; "
                "margin-top:6px;")
            return lbl

        def _make_slider(lo, hi, default, label_text, fmt_fn, desc_text, connect_fn):
            """Build a single slider row matching the image style."""
            container = QWidget()
            container.setStyleSheet("background:#1a1a1a;")
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(0, 2, 0, 6)
            vbox.setSpacing(1)

            # Value label (updates live)
            val_lbl = QLabel(f"{label_text}: {fmt_fn(default)}")
            val_lbl.setStyleSheet("color:#ccc; font-size:10px;")
            vbox.addWidget(val_lbl)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(lo, hi)
            slider.setValue(default)
            slider.setStyleSheet(_SLIDER_STYLE)

            def _on_change(v, lbl=val_lbl, lt=label_text, ff=fmt_fn):
                lbl.setText(f"{lt}: {ff(v)}")
                connect_fn(v)

            slider.valueChanged.connect(_on_change)
            vbox.addWidget(slider)

            # Description
            desc = QLabel(desc_text)
            desc.setStyleSheet("color:#666; font-size:9px;")
            desc.setWordWrap(True)
            vbox.addWidget(desc)

            return container, slider

        # ── Section 1: YOLO Inference ──────────────────────────────────────────
        tab_thresh_layout.addWidget(_thresh_header("⚙", "YOLO INFERENCE THRESHOLDS"))

        def _set_conf(v):
            self._conf_thresh = v / 100.0
            self._update_thresh_status()

        conf_widget, self.conf_slider = _make_slider(
            1, 100, 25,
            "Confidence Threshold",
            lambda v: f"{v/100:.2f}",
            "Min YOLO confidence to accept a detection. Lower = more detections, more false positives. "
            "Recommended: 0.20–0.35 for body keypoints. Antenna tips (k5–k8) may need ~0.15.",
            _set_conf)
        tab_thresh_layout.addWidget(conf_widget)

        def _set_iou(v):
            self._iou_thresh = v / 100.0
            self._update_thresh_status()

        iou_widget, self.iou_slider = _make_slider(
            1, 100, 45,
            "IoU Threshold (NMS)",
            lambda v: f"{v/100:.2f}",
            "Non-Maximum Suppression overlap threshold. Lower = fewer duplicate boxes. "
            "Recommended: 0.40–0.50 for single-bee frames; 0.60+ for multi-bee.",
            _set_iou)
        tab_thresh_layout.addWidget(iou_widget)

        # ── Section 2: Matching ────────────────────────────────────────────────
        tab_thresh_layout.addWidget(_thresh_header("🎯", "GT–PREDICTION MATCHING THRESHOLDS"))

        def _set_match(v):
            self._match_thresh = v
            self._update_thresh_status()

        match_widget, self.match_slider = _make_slider(
            5, 200, 50,
            "Spatial Match Radius",
            lambda v: f"{v} px",
            "Max pixel distance for a prediction to count as a True Positive vs a GT keypoint. "
            "Body (k0–k4): 30–50px. Antenna tips (k5–k8): 60–100px (higher localization error).",
            _set_match)
        tab_thresh_layout.addWidget(match_widget)

        # ── Section 3: PCK ────────────────────────────────────────────────────
        tab_thresh_layout.addWidget(_thresh_header("📈", "PCK DISTANCE THRESHOLDS"))

        def _set_pck1(v):
            self._pck_thresholds = sorted([v, self.pck2_slider.value(), self.pck3_slider.value()])
            self._update_thresh_status()

        def _set_pck2(v):
            self._pck_thresholds = sorted([self.pck1_slider.value(), v, self.pck3_slider.value()])
            self._update_thresh_status()

        def _set_pck3(v):
            self._pck_thresholds = sorted([self.pck1_slider.value(), self.pck2_slider.value(), v])
            self._update_thresh_status()

        pck1_widget, self.pck1_slider = _make_slider(
            1, 100, 5,
            "PCK Threshold 1",
            lambda v: f"{v} px  (strict)",
            "Strict localization quality — typically represents fine motor task accuracy.",
            _set_pck1)
        tab_thresh_layout.addWidget(pck1_widget)

        pck2_widget, self.pck2_slider = _make_slider(
            1, 100, 10,
            "PCK Threshold 2",
            lambda v: f"{v} px  (standard)",
            "Standard biological analysis tolerance for body keypoints.",
            _set_pck2)
        tab_thresh_layout.addWidget(pck2_widget)

        pck3_widget, self.pck3_slider = _make_slider(
            1, 100, 15,
            "PCK Threshold 3",
            lambda v: f"{v} px  (loose)",
            "Loose threshold appropriate for antenna tip uncertainty in thin-structure detection.",
            _set_pck3)
        tab_thresh_layout.addWidget(pck3_widget)

        # ── Section 4: Reference table ────────────────────────────────────────
        tab_thresh_layout.addWidget(_thresh_header("📋", "PER-KEYPOINT REFERENCE"))

        notes_table = QTableWidget()
        notes_table.setColumnCount(4)
        notes_table.setHorizontalHeaderLabels(["Keypoint","Group","Typical Error","Rec. Match px"])
        ref_rows = [
            ("k0","Body","~5 px","30 px"),("k1","Body","~4 px","30 px"),
            ("k2","Body","~8 px","35 px"),("k3","Body","~9 px","35 px"),
            ("k4","Body","~14 px","40 px"),("k5","Antenna","~40 px","80 px"),
            ("k6","Antenna","~36 px","70 px"),("k7","Antenna","~23 px","60 px"),
            ("k8","Antenna","~28 px","65 px"),
        ]
        notes_table.setRowCount(len(ref_rows))
        notes_table.verticalHeader().setVisible(False)
        notes_table.setStyleSheet("""
            QTableWidget { background:#1a1a1a; color:#e0e0e0; gridline-color:#2a2a2a;
                           font-size:10px; border:1px solid #333; }
            QHeaderView::section { background:#2d2d2d; color:#00b4d8; border:1px solid #333;
                                   padding:4px; font-weight:bold; font-size:9px; }
            QTableWidget::item:selected { background:#1a3a5c; }
        """)
        body_color = QColor("#1a3a2a")
        ant_color  = QColor("#3a1a1a")
        for r, (kp, grp, err, rec) in enumerate(ref_rows):
            bg = body_color if grp == "Body" else ant_color
            for c, val in enumerate([kp, grp, err, rec]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setBackground(bg)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                notes_table.setItem(r, c, item)
        notes_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        notes_table.setFixedHeight(220)
        tab_thresh_layout.addWidget(notes_table)

        tab_thresh_layout.addStretch()

        thresh_scroll.setWidget(thresh_content)
        tab_thresh_outer.addWidget(thresh_scroll)

        metrics_tabs.addTab(tab_thresh, "⚙ Thresholds")

        right_layout.addWidget(metrics_tabs)
        splitter.addWidget(right_widget)

        splitter.setSizes([620, 580])
        root.addWidget(splitter, stretch=1)

        # Store chart labels for resize updates
        self._chart_labels = {
            'errbar': self.errbar_lbl,
            'pck':    self.pck_lbl,
            'det':    self.det_lbl,
            'corr':   self.corr_lbl,
            'sev':    self.sev_lbl,
        }
        self._pipe_chart_labels = {
            'errbar': self.pipe_errbar_lbl,
            'pck':    self.pipe_pck_lbl,
            'det':    self.pipe_det_lbl,
            'corr':   self.pipe_corr_lbl,
            'sev':    self.pipe_sev_lbl,
        }
        self._chart_figs      = {}
        self._pipe_chart_figs = {}

    # ------------------------------------------------------------------ thresholds
    def _apply_thresholds(self):
        self._conf_thresh    = self.conf_slider.value() / 100.0
        self._iou_thresh     = self.iou_slider.value() / 100.0
        self._match_thresh   = self.match_slider.value()
        self._pck_thresholds = sorted([
            self.pck1_slider.value(),
            self.pck2_slider.value(),
            self.pck3_slider.value(),
        ])
        self._update_thresh_status()

    def _update_thresh_status(self):
        conf  = self.conf_slider.value() / 100.0
        iou   = self.iou_slider.value() / 100.0
        match = self.match_slider.value()
        p1    = self.pck1_slider.value()
        p2    = self.pck2_slider.value()
        p3    = self.pck3_slider.value()
        self.thresh_status_lbl.setText(
            f"⚙ Conf:{conf:.2f} | IoU:{iou:.2f} | Match:{match}px | PCK:{p1},{p2},{p3}px")
        self._conf_thresh    = conf
        self._iou_thresh     = iou
        self._match_thresh   = match
        self._pck_thresholds = sorted([p1, p2, p3])

    # ------------------------------------------------------------------ browse
    def _browse_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "YOLO Models (*.pt)")
        if f:
            self.eval_model_path = f
            self.eval_model_lbl.setText(Path(f).name)
            self.eval_model_lbl.setStyleSheet("color:#00ff00; font-size:10px; border:none;")
            self._check_ready()

    def _browse_coco(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select COCO JSON", "", "JSON Files (*.json)")
        if f:
            try:
                with open(f) as fp:
                    self.coco_data = json.load(fp)
                self.coco_lbl.setText(f"✅ {Path(f).name} ({len(self.coco_data['images'])} imgs)")
                self.coco_lbl.setStyleSheet("color:#00ff00; font-size:10px; border:none;")
                self._check_ready()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load COCO JSON:\n{e}")

    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if d:
            self.image_folder = d
            self.folder_lbl.setText(f"✅ {Path(d).name}")
            self.folder_lbl.setStyleSheet("color:#00ff00; font-size:10px; border:none;")
            self._check_ready()

    def _check_ready(self):
        ok = bool(self.eval_model_path and self.coco_data and self.image_folder)
        self.run_btn.setEnabled(ok)

    # ------------------------------------------------------------------ run/stop
    def _run(self):
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.dl_cum_btn.setEnabled(False)
        self.dl_frame_csv_btn.setEnabled(False)
        self.dl_frame_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.frame_predictions        = {}
        self._pipe_frame_predictions  = {}
        self._final_metrics           = {}
        self._all_errors_by_kp        = {}
        self._pipe_metrics            = {}
        self._pipe_errors_by_kp       = {}
        self._clear_frame_table()
        self.status_lbl.setText("▶ Stage 1/2: YOLO evaluation...")
        self.status_lbl.setStyleSheet("color:#4da6ff; font-size:10px;")

        self.worker = EvaluationWorker(
            self.eval_model_path, self.image_folder, self.coco_data,
            self._conf_thresh, self._iou_thresh, self._match_thresh,
            self._pck_thresholds)
        self.worker.progress.connect(self._on_progress)
        self.worker.frame_processed.connect(self._on_frame)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _start_pipeline_worker(self):
        """Called after YOLO worker finishes — runs the full pipeline sequentially."""
        self.status_lbl.setText("▶ Stage 2/2: Full pipeline evaluation...")
        self.status_lbl.setStyleSheet("color:#cc66ff; font-size:10px;")
        self.progress_bar.setValue(0)

        # Read LIVE threshold values from Implementation tab sliders
        det_thresh = {
            'motion_threshold': 3.0,
            'min_area':         5,
            'max_area':         400,
            'min_aspect_ratio': 2.5,
            'kernel_size':      5,
            'smoothing_frames': 3,
            'binary_threshold': 0.0,
        }
        if self.parent_gui is not None and hasattr(self.parent_gui, 'get_detection_thresholds'):
            try:
                det_thresh = self.parent_gui.get_detection_thresholds()
                print(f"[EVAL] Using Implementation tab thresholds: {det_thresh}")
            except Exception as e:
                print(f"[EVAL] Failed to read thresholds, using defaults: {e}")

        self.pipe_worker = PipelineEvaluationWorker(
            self.eval_model_path, self.image_folder, self.coco_data,
            self._conf_thresh, self._iou_thresh, self._match_thresh,
            self._pck_thresholds, det_thresh)
        self.pipe_worker.progress.connect(self._on_pipe_progress)
        self.pipe_worker.frame_processed.connect(self._on_pipe_frame)
        self.pipe_worker.finished.connect(self._on_pipe_finished)
        self.pipe_worker.error.connect(self._on_error)
        self.pipe_worker.start()

    def _stop(self):
        if self.worker:
            self.worker.stop()
        if self.pipe_worker:
            self.pipe_worker.stop()
        self.stop_btn.setEnabled(False)
        self.status_lbl.setText("Stopping...")

    # ------------------------------------------------------------------ slots
    def _on_progress(self, current, total, fname):
        pct = int(current / total * 100)
        self.progress_bar.setValue(pct)
        self.status_lbl.setText(f"[YOLO] Frame {current}/{total}: {fname}")

    def _on_frame(self, idx, frame_data):
        self.frame_predictions[idx] = frame_data
        if idx == 0:
            self.current_frame_idx = 0
            self._show_frame(frame_data)
        self._update_frame_table(frame_data['frame_metrics'])

    def _on_finished(self, result):
        self._final_metrics    = result['metrics']
        self._all_errors_by_kp = result['all_errors_by_kp']
        self.frame_predictions = result['frame_predictions']
        self.dl_cum_btn.setEnabled(True)
        self.dl_frame_csv_btn.setEnabled(True)
        self.dl_frame_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self._populate_cum_table(self._final_metrics)
        self._update_charts(self._final_metrics, self._all_errors_by_kp)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.current_frame_idx = 0
        self._show_frame_by_idx(0)
        # Chain into pipeline worker (sequential — no GPU collision)
        self._start_pipeline_worker()

    def _on_pipe_progress(self, current, total, fname):
        pct = int(current / total * 100)
        self.progress_bar.setValue(pct)
        self.status_lbl.setText(f"[2/2 Pipeline] {current}/{total}: {fname}")

    def _on_pipe_frame(self, idx, frame_data):
        self._pipe_frame_predictions[idx] = frame_data
        # Show pipeline image in the purple panel
        pipe_img = frame_data.get('pipe_image')
        if pipe_img is not None:
            h, w = pipe_img.shape[:2]
            rgb  = cv2.cvtColor(pipe_img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
            pix  = QPixmap.fromImage(qimg)
            self.pipe_display.set_full_pixmap(pix)
        # Update pipeline per-frame table for the current navigation index
        if idx == self.current_frame_idx:
            self._update_pipe_frame_table(frame_data.get('frame_metrics', {}))

    def _on_pipe_finished(self, result):
        self._pipe_metrics        = result['metrics']
        self._pipe_errors_by_kp   = result['all_errors_by_kp']
        self._pipe_frame_predictions = result['frame_predictions']
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_lbl.setText(
            f"✅ Complete — YOLO + Full Pipeline evaluated on {result['total_frames']} frames")
        self.status_lbl.setStyleSheet("color:#00ff00; font-size:10px;")
        self._populate_pipe_cum_table(self._pipe_metrics)
        self._update_pipe_charts(self._pipe_metrics, self._pipe_errors_by_kp)
        # Refresh current frame's pipeline panel
        self._show_frame_by_idx(self.current_frame_idx)

    def _on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_lbl.setText("❌ Error")
        self.status_lbl.setStyleSheet("color:#ff5555; font-size:10px;")
        QMessageBox.critical(self, "Evaluation Error", msg[:600])

    # ------------------------------------------------------------------ frame display
    def _show_frame(self, frame_data):
        def _bgr_to_pix(img):
            if img is None:
                return None
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(qimg)

        gt_pix   = _bgr_to_pix(frame_data.get('gt_image'))
        pred_pix = _bgr_to_pix(frame_data.get('pred_image'))
        vis_pix  = _bgr_to_pix(frame_data.get('vis_image'))

        if gt_pix:
            self.gt_display.set_full_pixmap(gt_pix)
        if pred_pix:
            self.pred_display.set_full_pixmap(pred_pix)
        if vis_pix:
            self.frame_display.set_full_pixmap(vis_pix)

        # Build composite for "Save Comparison"
        self._current_vis_pixmap = vis_pix  # fallback
        gt_img   = frame_data.get('gt_image')
        pred_img = frame_data.get('pred_image')
        vis_img  = frame_data.get('vis_image')
        if gt_img is not None and pred_img is not None and vis_img is not None:
            target_h = max(gt_img.shape[0], pred_img.shape[0], vis_img.shape[0])
            def _resize_h(img, h):
                scale = h / img.shape[0]
                return cv2.resize(img, (int(img.shape[1] * scale), h))
            gap = np.zeros((target_h, 4, 3), dtype=np.uint8)
            composite = np.hstack([
                _resize_h(gt_img,   target_h), gap,
                _resize_h(pred_img, target_h), gap,
                _resize_h(vis_img,  target_h),
            ])
            h, w = composite.shape[:2]
            rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
            self._current_vis_pixmap = QPixmap.fromImage(qimg)

        fname = frame_data.get('file_name', '')
        fi    = frame_data.get('frame_idx', 0)
        total = len(self.frame_predictions)
        self.frame_counter_lbl.setText(f"Frame {fi+1}/{total}  —  {fname}")
        self.dl_frame_btn.setEnabled(True)
        self._update_frame_table(frame_data.get('frame_metrics', {}))
        self.dl_frame_csv_btn.setEnabled(True)

    def _show_frame_by_idx(self, idx):
        keys = sorted(self.frame_predictions.keys())
        if not keys or idx >= len(keys):
            return
        self.current_frame_idx = idx
        self._show_frame(self.frame_predictions[keys[idx]])
        # Also update pipeline panel if data available
        pipe_keys = sorted(self._pipe_frame_predictions.keys())
        if pipe_keys and idx < len(pipe_keys):
            pipe_data = self._pipe_frame_predictions[pipe_keys[idx]]
            pipe_img = pipe_data.get('pipe_image')
            if pipe_img is not None:
                h, w = pipe_img.shape[:2]
                rgb  = cv2.cvtColor(pipe_img, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb.data.tobytes(), w, h, w*3, QImage.Format.Format_RGB888)
                self.pipe_display.set_full_pixmap(QPixmap.fromImage(qimg))
            self._update_pipe_frame_table(pipe_data.get('frame_metrics', {}))

    def _prev_frame(self):
        if self.current_frame_idx > 0:
            self._show_frame_by_idx(self.current_frame_idx - 1)

    def _next_frame(self):
        if self.current_frame_idx < len(self.frame_predictions) - 1:
            self._show_frame_by_idx(self.current_frame_idx + 1)

    # ------------------------------------------------------------------ tables
    def _clear_frame_table(self):
        for i in range(9):
            for j in range(1, 5):
                self.frame_table.setItem(i, j, QTableWidgetItem(""))
                self.pipe_frame_table.setItem(i, j, QTableWidgetItem(""))

    def _update_frame_table(self, frame_metrics):
        for k in range(9):
            m = frame_metrics.get(k, {'tp':0,'fp':0,'fn':0,'errors':[]})
            avg = f"{np.mean(m['errors']):.1f}" if m['errors'] else "—"
            vals = [str(m['tp']), str(m['fp']), str(m['fn']), avg]
            for j, v in enumerate(vals, 1):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.frame_table.setItem(k, j, item)

    def _populate_cum_table(self, metrics):
        for k in range(9):
            m = metrics.get(k, {})
            vals = [
                f"{m.get('mean_error',0):.2f}",
                f"{m.get('std_error',0):.2f}",
                str(m.get('tp',0)),
                str(m.get('fp',0)),
                str(m.get('fn',0)),
                f"{m.get('precision',0)*100:.1f}",
                f"{m.get('recall',0)*100:.1f}",
                f"{m.get('f1',0)*100:.1f}",
            ]
            for j, v in enumerate(vals, 1):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                # Color code F1 column
                if j == 8:
                    f1v = m.get('f1', 0)
                    if f1v >= 0.8:   item.setForeground(QColor('#00ff00'))
                    elif f1v >= 0.5: item.setForeground(QColor('#ffa500'))
                    else:            item.setForeground(QColor('#ff5555'))
                self.cum_table.setItem(k, j, item)

        # Summary
        all_tp  = sum(metrics.get(k,{}).get('tp',0) for k in range(9))
        all_fp  = sum(metrics.get(k,{}).get('fp',0) for k in range(9))
        all_fn  = sum(metrics.get(k,{}).get('fn',0) for k in range(9))
        avg_f1  = np.mean([metrics.get(k,{}).get('f1',0) for k in range(9)])
        avg_err = np.mean([metrics.get(k,{}).get('mean_error',0) for k in range(9)])
        body_f1 = np.mean([metrics.get(k,{}).get('f1',0) for k in range(5)])
        ant_f1  = np.mean([metrics.get(k,{}).get('f1',0) for k in range(5,9)])
        self.cum_summary_lbl.setText(
            f"Overall mF1: {avg_f1*100:.1f}%  |  Body F1: {body_f1*100:.1f}%  |  "
            f"Antenna F1: {ant_f1*100:.1f}%  |  "
            f"Mean Err: {avg_err:.1f}px  |  TP:{all_tp}  FP:{all_fp}  FN:{all_fn}"
        )

    # ------------------------------------------------------------------ charts
    def _update_charts(self, metrics, all_errors_by_kp):
        if not MATPLOTLIB_AVAILABLE:
            return
        figs = {
            'errbar': _make_error_bar_chart(metrics),
            'pck':    _make_pck_chart(metrics, all_errors_by_kp, self._pck_thresholds),
            'det':    _make_detection_chart(metrics),
            'corr':   _make_correlation_chart(all_errors_by_kp),
            'sev':    _make_severity_chart(all_errors_by_kp),
        }
        self._chart_figs = figs
        for key, fig in figs.items():
            pix = _fig_to_qpixmap(fig, dpi=95)
            lbl = self._chart_labels[key]
            lbl.setPixmap(pix.scaled(lbl.size(),
                                     Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation))
            plt.close(fig)

    def _update_pipe_charts(self, metrics, all_errors_by_kp):
        if not MATPLOTLIB_AVAILABLE:
            return
        figs = {
            'errbar': _make_error_bar_chart(metrics),
            'pck':    _make_pck_chart(metrics, all_errors_by_kp, self._pck_thresholds),
            'det':    _make_detection_chart(metrics),
            'corr':   _make_correlation_chart(all_errors_by_kp),
            'sev':    _make_severity_chart(all_errors_by_kp),
        }
        self._pipe_chart_figs = figs
        for key, fig in figs.items():
            pix = _fig_to_qpixmap(fig, dpi=95)
            lbl = self._pipe_chart_labels[key]
            lbl.setPixmap(pix.scaled(lbl.size(),
                                     Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation))
            plt.close(fig)

    def _populate_pipe_cum_table(self, metrics):
        for k in range(9):
            m = metrics.get(k, {})
            vals = [
                f"{m.get('mean_error',0):.2f}",
                f"{m.get('std_error',0):.2f}",
                str(m.get('tp',0)), str(m.get('fp',0)), str(m.get('fn',0)),
                f"{m.get('precision',0)*100:.1f}",
                f"{m.get('recall',0)*100:.1f}",
                f"{m.get('f1',0)*100:.1f}",
            ]
            for j, v in enumerate(vals, 1):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if j == 8:
                    f1v = m.get('f1', 0)
                    if f1v >= 0.8:   item.setForeground(QColor('#cc66ff'))
                    elif f1v >= 0.5: item.setForeground(QColor('#ff99cc'))
                    else:            item.setForeground(QColor('#ff5555'))
                self.pipe_cum_table.setItem(k, j, item)
        all_tp  = sum(metrics.get(k,{}).get('tp',0) for k in range(9))
        all_fp  = sum(metrics.get(k,{}).get('fp',0) for k in range(9))
        all_fn  = sum(metrics.get(k,{}).get('fn',0) for k in range(9))
        avg_f1  = np.mean([metrics.get(k,{}).get('f1',0) for k in range(9)])
        avg_err = np.mean([metrics.get(k,{}).get('mean_error',0) for k in range(9)])
        body_f1 = np.mean([metrics.get(k,{}).get('f1',0) for k in range(5)])
        ant_f1  = np.mean([metrics.get(k,{}).get('f1',0) for k in range(5,9)])
        self.pipe_summary_lbl.setText(
            f"Pipeline mF1: {avg_f1*100:.1f}%  |  Body F1: {body_f1*100:.1f}%  |  "
            f"Antenna F1: {ant_f1*100:.1f}%  |  "
            f"Mean Err: {avg_err:.1f}px  |  TP:{all_tp}  FP:{all_fp}  FN:{all_fn}"
        )

    def _update_pipe_frame_table(self, frame_metrics):
        for k in range(9):
            m = frame_metrics.get(k, {'tp':0,'fp':0,'fn':0,'errors':[]})
            avg = f"{np.mean(m['errors']):.1f}" if m['errors'] else "—"
            vals = [str(m['tp']), str(m['fp']), str(m['fn']), avg]
            for j, v in enumerate(vals, 1):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.pipe_frame_table.setItem(k, j, item)

    # ------------------------------------------------------------------ download
    def _download_frame(self):
        if self._current_vis_pixmap is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Frame Image", "frame_comparison.png",
            "PNG Image (*.png);;JPEG (*.jpg)")
        if path:
            self._current_vis_pixmap.save(path)

    def _download_cum_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Cumulative Metrics CSV", "cumulative_metrics.csv",
            "CSV Files (*.csv)")
        if not path:
            return
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Keypoint','Mean_Error_px','Std_Error','TP','FP','FN','Precision','Recall','F1'])
            for k in range(9):
                m = self._final_metrics.get(k, {})
                w.writerow([k,
                    f"{m.get('mean_error',0):.4f}",
                    f"{m.get('std_error',0):.4f}",
                    m.get('tp',0), m.get('fp',0), m.get('fn',0),
                    f"{m.get('precision',0):.4f}",
                    f"{m.get('recall',0):.4f}",
                    f"{m.get('f1',0):.4f}"])

    def _download_frame_csv(self):
        keys = sorted(self.frame_predictions.keys())
        if not keys:
            return
        frame_data = self.frame_predictions[keys[self.current_frame_idx]]
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Frame Metrics CSV", "frame_metrics.csv",
            "CSV Files (*.csv)")
        if not path:
            return
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Frame', frame_data.get('file_name', '')])
            w.writerow(['Keypoint','TP','FP','FN','Avg_Error_px'])
            for k in range(9):
                m = frame_data.get('frame_metrics', {}).get(k, {'tp':0,'fp':0,'fn':0,'errors':[]})
                avg = f"{np.mean(m['errors']):.4f}" if m['errors'] else "0"
                w.writerow([k, m['tp'], m['fp'], m['fn'], avg])


class KeypointViewerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"BeeVision — {TORCH_DEVICE.upper()}")
        self.setGeometry(100, 100, 1600, 850)

        self.central = QWidget()
        self.setCentralWidget(self.central)
        root_layout = QVBoxLayout(self.central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ══════════ TOP-LEVEL TAB WIDGET ══════════
        self.top_tabs = QTabWidget()
        self.top_tabs.setStyleSheet("""
            QTabWidget::pane { border: none; }
            QTabBar::tab {
                padding: 10px 22px;
                font-weight: bold;
                font-size: 11px;
                background: #2d2d2d;
                color: #888;
                border: 1px solid #444;
                border-bottom: none;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #1a1a1a;
                color: #00b4d8;
                border-bottom: 2px solid #00b4d8;
            }
            QTabBar::tab:hover { color: #00b4d8; }
        """)

        # ─────────────────────────── TAB 1: IMPLEMENTATION ───────────────────
        impl_widget = QWidget()
        main_layout  = QHBoxLayout(impl_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # ==================== LEFT SIDE: VIDEO TABS ====================
        self.tab_widget = QTabWidget()
        self.tab_widget.setMinimumWidth(900)
        
        self.video_display = QLabel("Load Video to Begin")
        self.video_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display.setStyleSheet("background: #1e1e1e; border: 2px solid #333; color: #555;")
        self.video_display.setMinimumSize(800, 600)
        self.tab_widget.addTab(self.video_display, "Pose Estimation")
        
        self.darkest_display = QLabel("Darkest pixels visualization will appear here")
        self.darkest_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.darkest_display.setStyleSheet("background: #1e1e1e; border: 2px solid #333; color: #555;")
        self.darkest_display.setMinimumSize(800, 600)
        self.tab_widget.addTab(self.darkest_display, "Darkest Pixels & ROI")
        
        self.mask_display = QLabel("Darkest mask visualization will appear here")
        self.mask_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mask_display.setStyleSheet("background: #000000; border: 2px solid #333; color: #555;")
        self.mask_display.setMinimumSize(800, 600)
        self.tab_widget.addTab(self.mask_display, "Darkest Mask (Viridis)")
        
        self.mask_no_keypoints_display = QLabel("Darkest mask without keypoints will appear here")
        self.mask_no_keypoints_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mask_no_keypoints_display.setStyleSheet("background: #000000; border: 2px solid #333; color: #555;")
        self.mask_no_keypoints_display.setMinimumSize(800, 600)
        self.tab_widget.addTab(self.mask_no_keypoints_display, "Darkest Map (No Keypoints)")
        
        self.body_keypoints_roi_display = QLabel("Body keypoints and ROI direction vectors will appear here")
        self.body_keypoints_roi_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.body_keypoints_roi_display.setStyleSheet("background: #000000; border: 2px solid #333; color: #555;")
        self.body_keypoints_roi_display.setMinimumSize(800, 600)
        self.tab_widget.addTab(self.body_keypoints_roi_display, "Body Keypoints + ROI Vectors")
        
        self.roi_bbox_body_display = QLabel("ROI triangles, bounding box, and body keypoints will appear here")
        self.roi_bbox_body_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.roi_bbox_body_display.setStyleSheet("background: #000000; border: 2px solid #333; color: #555;")
        self.roi_bbox_body_display.setMinimumSize(800, 600)
        self.tab_widget.addTab(self.roi_bbox_body_display, "ROI + BBox + Body")
        
        self.grayscale_body_roi_display = QLabel("Grayscale with body keypoints and ROI angles will appear here")
        self.grayscale_body_roi_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.grayscale_body_roi_display.setStyleSheet("background: #000000; border: 2px solid #333; color: #555;")
        self.grayscale_body_roi_display.setMinimumSize(800, 600)
        self.tab_widget.addTab(self.grayscale_body_roi_display, "Grayscale + Body + ROI")
        
        self.full_frame_darkest_display = QLabel("Full frame darkest visualization will appear here")
        self.full_frame_darkest_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.full_frame_darkest_display.setStyleSheet("background: #000000; border: 2px solid #333; color: #555;")
        self.full_frame_darkest_display.setMinimumSize(800, 600)
        self.tab_widget.addTab(self.full_frame_darkest_display, "Full Frame Darkest")
        
        # ========== NEW: FOR RESEARCH TAB (Morphological Pipeline Visualization) ==========
        self.research_tab, self.research_widget = create_research_visualization_tab()
        self.tab_widget.addTab(self.research_tab, "🔬 For Research")
        
        main_layout.addWidget(self.tab_widget, stretch=3)
        
        # ==================== RIGHT SIDE: SIDEBAR WITH TABS ====================
        self.sidebar = QWidget()
        self.sidebar.setMinimumWidth(650)
        self.side_layout = QVBoxLayout(self.sidebar)
        self.side_layout.setSpacing(0)
        self.side_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create tab widget for sidebar (Configuration, Analysis, System Info)
        self.sidebar_tab_widget = QTabWidget()
        self.sidebar_tab_widget.setStyleSheet("""
            QTabBar::tab {
                padding: 8px 15px;
                font-weight: bold;
            }
        """)
        
        # ========== TAB 1: CONFIGURATION ==========
        config_widget = QWidget()
        config_outer_layout = QVBoxLayout(config_widget)
        config_outer_layout.setContentsMargins(0,0,0,0)
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setStyleSheet(
            "QScrollArea { background:#1a1a1a; border:none; }"
            "QScrollBar:vertical { background:#2a2a2a; width:8px; border-radius:4px; }"
            "QScrollBar::handle:vertical { background:#444; border-radius:4px; }"
            "QScrollBar::handle:vertical:hover { background:#00b4d8; }")
        config_inner = QWidget()
        config_layout = QVBoxLayout(config_inner)
        config_scroll.setWidget(config_inner)
        config_outer_layout.addWidget(config_scroll)
        
        self.mdl_btn = QPushButton("Select Model (.pt)")
        self.mdl_btn.clicked.connect(self._get_model)
        self.mdl_lbl = QLabel("No Model Selected")
        self.mdl_lbl.setStyleSheet("color: #888; font-size: 11px;")
        
        self.vid_btn = QPushButton("Select Video Source")
        self.vid_btn.clicked.connect(self._get_video)
        self.vid_path_lbl = QLabel("No Video Selected")
        self.vid_path_lbl.setStyleSheet("color: #888; font-size: 11px;")

        self.img_folder_btn = QPushButton("📁 Select Image Folder (60 FPS)")
        self.img_folder_btn.clicked.connect(self._get_image_folder)
        self.img_folder_btn.setStyleSheet("QPushButton { background: #1a3a4a; color: #00b4d8; border: 1px solid #00b4d8; padding: 6px; border-radius: 4px; font-weight: bold; } QPushButton:hover { background: #1e4a60; }")
        self.img_folder_lbl = QLabel("No Image Folder Selected")
        self.img_folder_lbl.setStyleSheet("color: #888; font-size: 11px;")
        
        config_layout.addWidget(QLabel("Model Selection:"))
        config_layout.addWidget(self.mdl_btn)
        config_layout.addWidget(self.mdl_lbl)
        config_layout.addSpacing(15)
        
        config_layout.addWidget(QLabel("Video Source:"))
        config_layout.addWidget(self.vid_btn)
        config_layout.addWidget(self.vid_path_lbl)
        config_layout.addSpacing(8)

        config_layout.addWidget(QLabel("— OR —"))
        config_layout.addSpacing(4)
        config_layout.addWidget(QLabel("Image Folder Source (treated as 60 FPS):"))
        config_layout.addWidget(self.img_folder_btn)
        config_layout.addWidget(self.img_folder_lbl)
        config_layout.addSpacing(20)
        
        self.btn_run = QPushButton("▶ START VISUALIZATION")
        self.btn_run.setStyleSheet("QPushButton { background: #2e7d32; color: white; border: none; padding: 12px; font-weight: bold; font-size: 12px; }")
        self.btn_run.clicked.connect(self._start)
        self.btn_run.setEnabled(False)
        
        self.btn_stop = QPushButton("⏹ STOP")
        self.btn_stop.setStyleSheet("QPushButton { background: #c62828; color: white; border: none; padding: 12px; font-weight: bold; font-size: 12px; }")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)
        
        self.lbl_stat = QLabel("System Ready")
        self.lbl_stat.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_stat.setStyleSheet("color: #aaa; font-size: 11px; margin-top: 10px;")
        
        config_layout.addWidget(self.btn_run)
        config_layout.addWidget(self.btn_stop)
        config_layout.addWidget(self.lbl_stat)
        
        # ========== THRESHOLD SLIDERS (Real-time Adjustments) ==========
        config_layout.addSpacing(30)
        
        # Threshold section title
        threshold_title = QLabel("⚙️ DETECTION THRESHOLDS")
        threshold_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        threshold_title.setStyleSheet("color: #00b4d8;")
        config_layout.addWidget(threshold_title)
        
        # 1. Motion Threshold (default 3.0, range 0.5 - 10.0)
        motion_label = QLabel(f"Motion Threshold: 3.0 pixels")
        motion_label.setStyleSheet("color: #ccc; font-size: 10px;")
        self.motion_slider = QSlider(Qt.Orientation.Horizontal)
        self.motion_slider.setRange(5, 100)  # 0.5 to 10.0 (x10)
        self.motion_slider.setValue(30)  # default 3.0
        self.motion_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #444;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00b4d8;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        self.motion_slider.valueChanged.connect(
            lambda val: motion_label.setText(f"Motion Threshold: {val/10:.1f} pixels")
        )
        config_layout.addWidget(motion_label)
        config_layout.addWidget(self.motion_slider)
        
        # 2. Min Area (default 5 for thin worker bee antennae, range 1 - 50)
        # TIP: Use 5-10 for worker bees, 15-25 for yellowjackets
        area_min_label = QLabel(f"Min Antenna Area: 5 pixels")
        area_min_label.setStyleSheet("color: #ccc; font-size: 10px;")
        self.area_min_slider = QSlider(Qt.Orientation.Horizontal)
        self.area_min_slider.setRange(1, 50)
        self.area_min_slider.setValue(5)  # Default for thin worker bee antennae
        self.area_min_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #444;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00b4d8;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        self.area_min_slider.valueChanged.connect(
            lambda val: area_min_label.setText(f"Min Antenna Area: {val} pixels")
        )
        config_layout.addWidget(area_min_label)
        config_layout.addWidget(self.area_min_slider)
        
        # 3. Max Area (default 400 for worker bees, range 100 - 1200)
        # TIP: Use 300-500 for worker bees, 600-1000 for yellowjackets
        area_max_label = QLabel(f"Max Antenna Area: 400 pixels")
        area_max_label.setStyleSheet("color: #ccc; font-size: 10px;")
        self.area_max_slider = QSlider(Qt.Orientation.Horizontal)
        self.area_max_slider.setRange(100, 1200)
        self.area_max_slider.setValue(400)  # Default for thin worker bee antennae
        self.area_max_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #444;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00b4d8;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        self.area_max_slider.valueChanged.connect(
            lambda val: area_max_label.setText(f"Max Antenna Area: {val} pixels")
        )
        config_layout.addWidget(area_max_label)
        config_layout.addWidget(self.area_max_slider)
        
        # 4. Aspect Ratio (default 2.5 for curved worker bee antennae, range 1.5 - 10.0)
        # TIP: Use 2.0-3.0 for worker bees (more curved), 4.0-6.0 for yellowjackets (straighter)
        aspect_label = QLabel(f"Min Aspect Ratio: 2.5")
        aspect_label.setStyleSheet("color: #ccc; font-size: 10px;")
        self.aspect_slider = QSlider(Qt.Orientation.Horizontal)
        self.aspect_slider.setRange(15, 100)  # 1.5 to 10.0 (x10)
        self.aspect_slider.setValue(25)  # default 2.5 for worker bees
        self.aspect_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #444;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00b4d8;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        self.aspect_slider.valueChanged.connect(
            lambda val: aspect_label.setText(f"Min Aspect Ratio: {val/10:.1f}")
        )
        config_layout.addWidget(aspect_label)
        config_layout.addWidget(self.aspect_slider)
        
        # 5. Kernel Size Mode (Thin vs Thick antennae)
        # TIP: Use 3-7 for thin worker bee antennae, 11-21 for thick yellowjacket antennae
        kernel_mode_label = QLabel(f"Kernel Size: 5px (Thin antennae)")
        kernel_mode_label.setStyleSheet("color: #ccc; font-size: 10px;")
        self.kernel_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.kernel_size_slider.setRange(3, 21)  # 3 to 21 pixels
        self.kernel_size_slider.setValue(5)  # default 5 for thin worker bee antennae
        self.kernel_size_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #444;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00b4d8;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        def update_kernel_label(val):
            if val <= 7:
                mode = "THIN antennae"
            elif val <= 12:
                mode = "Medium"
            else:
                mode = "THICK antennae"
            kernel_mode_label.setText(f"Kernel Size: {mode} ({val}px)")
        
        self.kernel_size_slider.valueChanged.connect(update_kernel_label)
        config_layout.addWidget(kernel_mode_label)
        config_layout.addWidget(self.kernel_size_slider)
        
        # 6. Smoothing/Responsiveness (default 3, range 1-10)
        smoothing_label = QLabel(f"Smoothing: Fast (3 frames)")
        smoothing_label.setStyleSheet("color: #ccc; font-size: 10px;")
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setRange(1, 10)
        self.smoothing_slider.setValue(3)  # default 3
        self.smoothing_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #444;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00b4d8;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        def update_smoothing_label(val):
            if val <= 2:
                mode = "Very Fast (less smooth)"
            elif val <= 4:
                mode = "Fast"
            elif val <= 7:
                mode = "Balanced"
            else:
                mode = "Smooth (slower)"
            smoothing_label.setText(f"Smoothing: {mode} ({val} frames)")
        
        self.smoothing_slider.valueChanged.connect(update_smoothing_label)
        config_layout.addWidget(smoothing_label)
        config_layout.addWidget(self.smoothing_slider)
        
        # 5. ROI Triangle Thickness (default 1, range 0-5)
        config_layout.addSpacing(10)
        roi_thickness_label = QLabel(f"ROI Triangle Thickness: 1 pixel")
        roi_thickness_label.setStyleSheet("color: #ccc; font-size: 10px;")
        self.roi_thickness_slider = QSlider(Qt.Orientation.Horizontal)
        self.roi_thickness_slider.setRange(0, 5)
        self.roi_thickness_slider.setValue(1)  # Default thickness
        self.roi_thickness_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #444;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00b4d8;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        
        def update_roi_thickness_label(val):
            if val == 0:
                roi_thickness_label.setText(f"ROI Triangle Thickness: {val} pixels (Hidden)")
            elif val == 1:
                roi_thickness_label.setText(f"ROI Triangle Thickness: {val} pixel")
            else:
                roi_thickness_label.setText(f"ROI Triangle Thickness: {val} pixels")
            # Real-time update to worker if running
            if hasattr(self, 'worker') and self.worker is not None:
                self.worker.roi_thickness = val
        
        self.roi_thickness_slider.valueChanged.connect(update_roi_thickness_label)
        config_layout.addWidget(roi_thickness_label)
        config_layout.addWidget(self.roi_thickness_slider)
        
        # 8. ROI Percentile (default 0 = Auto/Otsu, range 0%-50% with 0.1% steps)
        #    Keeps top N% brightest tophat pixels within each antenna ROI.
        #    Antennae are always the strongest local response, so they survive
        #    regardless of absolute brightness — unlike a global threshold.
        config_layout.addSpacing(10)
        self.binary_thresh_label = QLabel("ROI Percentile: Auto (Otsu)")
        self.binary_thresh_label.setStyleSheet("color: #ccc; font-size: 10px;")
        self.binary_thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.binary_thresh_slider.setRange(0, 500)  # 0-500 → 0.0%-50.0%
        self.binary_thresh_slider.setValue(0)  # 0 = Auto/Otsu
        self.binary_thresh_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #444;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ff6b6b;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        
        def update_binary_thresh_label(val):
            float_val = val / 10.0  # 0-500 → 0.0-50.0%
            if val == 0:
                self.binary_thresh_label.setText("ROI Percentile: Auto (Otsu)")
            else:
                self.binary_thresh_label.setText(f"ROI Percentile: Top {float_val:.1f}%")
            # Real-time update to worker if running
            if hasattr(self, 'worker') and self.worker is not None:
                self.worker.binary_threshold = float_val
        
        self.binary_thresh_slider.valueChanged.connect(update_binary_thresh_label)
        config_layout.addWidget(self.binary_thresh_label)
        config_layout.addWidget(self.binary_thresh_slider)

        # 9. Pixel Darkness (0 = original, 100 = fully black)
        #    Applied to each raw frame BEFORE any detection/processing method runs.
        config_layout.addSpacing(10)
        self.pixel_darkness_label = QLabel("Frame Darkness: 0% (Original)")
        self.pixel_darkness_label.setStyleSheet("color: #ccc; font-size: 10px;")
        self.pixel_darkness_slider = QSlider(Qt.Orientation.Horizontal)
        self.pixel_darkness_slider.setRange(0, 100)   # 0 = no change, 100 = fully black
        self.pixel_darkness_slider.setValue(0)
        self.pixel_darkness_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #444;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #a29bfe;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #6c5ce7;
                border-radius: 3px;
            }
        """)

        def update_pixel_darkness_label(val):
            if val == 0:
                self.pixel_darkness_label.setText("Frame Darkness: 0% (Original)")
            elif val == 100:
                self.pixel_darkness_label.setText("Frame Darkness: 100% (Fully Black)")
            else:
                self.pixel_darkness_label.setText(f"Frame Darkness: {val}%")
            # Real-time update to worker if running
            if hasattr(self, 'worker') and self.worker is not None:
                self.worker.pixel_darkness = val / 100.0

        self.pixel_darkness_slider.valueChanged.connect(update_pixel_darkness_label)
        config_layout.addWidget(self.pixel_darkness_label)
        config_layout.addWidget(self.pixel_darkness_slider)

        # 10. Antenna EMA Smoothing Alpha (Layer 2 temporal stabilisation)
        #     α = 1.0  → no smoothing (raw per-frame detection)
        #     α = 0.1  → heavy smoothing (slow response, very stable)
        #     Default = 0.6 (responsive with mild jitter reduction)
        config_layout.addSpacing(10)
        self.ema_alpha_label = QLabel("Antenna EMA α: 0.60 (Balanced)")
        self.ema_alpha_label.setStyleSheet("color: #ccc; font-size: 10px;")
        self.ema_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.ema_alpha_slider.setRange(10, 90)   # 10-90 → 0.10-0.90
        self.ema_alpha_slider.setValue(60)        # default α = 0.60
        self.ema_alpha_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #444;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ffd166;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #e6a817;
                border-radius: 3px;
            }
        """)

        def update_ema_alpha_label(val):
            alpha = val / 100.0
            if val >= 80:
                mode = "Responsive (more jitter)"
            elif val >= 50:
                mode = "Balanced"
            elif val >= 30:
                mode = "Smooth (slight lag)"
            else:
                mode = "Heavy smooth (laggy)"
            self.ema_alpha_label.setText(f"Antenna EMA α: {alpha:.2f} ({mode})")
            # Real-time update to worker if running
            if hasattr(self, 'worker') and self.worker is not None:
                self.worker.ema_alpha = alpha

        self.ema_alpha_slider.valueChanged.connect(update_ema_alpha_label)
        config_layout.addWidget(self.ema_alpha_label)
        config_layout.addWidget(self.ema_alpha_slider)

        # Apply Thresholds Button
        config_layout.addSpacing(15)
        self.btn_apply_thresholds = QPushButton("✓ APPLY THRESHOLDS")
        self.btn_apply_thresholds.setStyleSheet("""
            QPushButton { 
                background: #0077b6; 
                color: white; 
                border: none; 
                padding: 10px; 
                font-weight: bold; 
                font-size: 11px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #00b4d8;
            }
            QPushButton:pressed {
                background: #005f8a;
            }
            QPushButton:disabled {
                background: #444;
                color: #888;
            }
        """)
        self.btn_apply_thresholds.clicked.connect(self._apply_thresholds)
        self.btn_apply_thresholds.setEnabled(False)  # Disabled until video is running
        config_layout.addWidget(self.btn_apply_thresholds)
        
        config_layout.addSpacing(10)
        config_layout.addStretch()
        
        self.sidebar_tab_widget.addTab(config_widget, "⚙️ Configuration")
        
        # ========== TAB 2: UNIFIED ANALYSIS DASHBOARD ==========
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        analysis_layout.setSpacing(0)
        analysis_layout.setContentsMargins(0, 0, 0, 0)
        
        # Single unified display for all metrics
        self.unified_analysis_display = QLabel("Analysis dashboard loading...")
        self.unified_analysis_display.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.unified_analysis_display.setStyleSheet(
            "background: #0a0a0a; color: #00ff00; font-family: 'Courier New', monospace; "
            "font-size: 10px; padding: 10px; border: 1px solid #333;"
        )
        self.unified_analysis_display.setMinimumHeight(400)
        self.unified_analysis_display.setWordWrap(True)
        
        analysis_scroll = QScrollArea()
        analysis_scroll.setWidget(self.unified_analysis_display)
        analysis_scroll.setWidgetResizable(True)
        analysis_scroll.setStyleSheet("QScrollArea { background: #1e1e1e; border: none; }")
        
        analysis_layout.addWidget(analysis_scroll)
        self.sidebar_tab_widget.addTab(analysis_widget, "📊 Legacy Analysis")
        
        # ========== NEW: MODERN ANTENNA DASHBOARD ==========
        self.antenna_dashboard_tab, self.antenna_cards = create_antenna_dashboard_tab()
        self.sidebar_tab_widget.addTab(self.antenna_dashboard_tab, "🐝 Antenna Dashboard")
        
        # ========== TAB 3: SYSTEM INFORMATION ==========
        system_widget = QWidget()
        system_layout = QVBoxLayout(system_widget)
        
        cuda_status = 'Yes' if CUDA_AVAILABLE else 'No'
        numba_status = 'Yes' if NUMBA_AVAILABLE else 'No'
        
        info_text = QLabel(
            f"<b>🖥️ HARDWARE INFORMATION</b><br>"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
            f"<b>Device:</b> {TORCH_DEVICE.upper()}<br>"
            f"<b>CUDA Available:</b> {TORCH_CUDA_AVAILABLE}<br>"
            f"<b>GPU Devices:</b> {CUDA_DEVICE_COUNT}<br>"
            f"<b>GPU Name:</b> {CUDA_DEVICE_NAME}<br>"
            f"<b>CuPy:</b> {cuda_status}<br>"
            f"<b>Numba:</b> {numba_status}<br><br>"
            
            f"<b>🎯 DETECTION FEATURES</b><br>"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
            f"✓ Multi-directional tophat<br>"
            f"✓ Convex hull endpoints<br>"
            f"✓ Composite scoring<br>"
            f"✓ Aspect ratio filtering<br>"
            f"✓ Direction-constrained tracking<br>"
            f"✓ Enhanced visualization<br>"
            f"✓ Trophallaxis detection<br>"
            f"✓ 4-Region anatomical tracking<br><br>"
            
            f"<b>📺 VISUALIZATION TABS</b><br>"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
            f"<b>Tab 1:</b> Pose Estimation<br>"
            f"  • Keypoints with skeleton<br>"
            f"  • Real-time tracking<br><br>"
            
            f"<b>Tab 2:</b> Darkest Pixels<br>"
            f"  • Illumination heatmap<br>"
            f"  • ROI triangles<br><br>"
            
            f"<b>Tab 3:</b> Darkest Mask<br>"
            f"  • Viridis colormap<br>"
            f"  • Antenna detection<br><br>"
            
            f"<b>Tab 4:</b> Full Frame<br>"
            f"  • Complete analysis<br>"
            f"  • Pose overlay<br><br>"
            
            f"<b>📊 SIDEBAR ANALYSIS TABS</b><br>"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
            f"<b>Metrics:</b> Trophallaxis events<br>"
            f"<b>Antenna Analysis:</b> RR/RL/LR/LL types<br>"
            f"<b>Antenna Dominance:</b> 5 health metrics<br>"
            f"<b>4-Region Contacts:</b> Anatomical mapping<br><br>"
            
            f"<b>⚠️ Note:</b> Check console for detailed GPU info!"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet(
            "color: #ccc; font-size: 11px; background: #0a0a0a; "
            "padding: 10px; border: 1px solid #333; border-radius: 4px;"
        )
        
        system_scroll = QScrollArea()
        system_scroll.setWidget(info_text)
        system_scroll.setWidgetResizable(True)
        system_scroll.setStyleSheet("QScrollArea { background: #1e1e1e; border: none; }")
        
        system_layout.addWidget(system_scroll)
        self.sidebar_tab_widget.addTab(system_widget, "ℹ️ System Info")
        
        # ==================== ADD SIDEBAR TAB WIDGET TO MAIN LAYOUT ====================
        self.side_layout.addWidget(self.sidebar_tab_widget)
        main_layout.addWidget(self.sidebar, stretch=2)

        # ─────────────────────────── TAB 2: EVALUATION ───────────────────────
        self.evaluation_tab = EvaluationTab(parent=self)

        # ══════════ ADD TABS TO TOP-LEVEL ══════════
        self.top_tabs.addTab(impl_widget,           "🔬 Implementation")
        self.top_tabs.addTab(self.evaluation_tab,   "🎯 Evaluation")
        root_layout.addWidget(self.top_tabs)

        self.model_path        = ""
        self.video_path        = ""
        self.image_folder_path = None    # set when user picks an image folder
        self.input_mode        = 'video' # 'video' | 'images'
        self.worker            = None
        
        # Load saved thresholds from disk (persists across restarts)
        self._thresholds_config_path = os.path.join(
            os.path.expanduser('~'), '.config', 'beevision', 'thresholds.json'
        )
        self._load_thresholds()
    def _save_thresholds(self):
        """Save all threshold slider values to disk for persistence across restarts"""
        config = {
            'motion_threshold': self.motion_slider.value(),
            'min_area': self.area_min_slider.value(),
            'max_area': self.area_max_slider.value(),
            'min_aspect_ratio': self.aspect_slider.value(),
            'kernel_size': self.kernel_size_slider.value(),
            'smoothing': self.smoothing_slider.value(),
            'roi_thickness': self.roi_thickness_slider.value(),
            'binary_threshold': self.binary_thresh_slider.value(),
            'ema_alpha': self.ema_alpha_slider.value(),
        }
        try:
            os.makedirs(os.path.dirname(self._thresholds_config_path), exist_ok=True)
            with open(self._thresholds_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"[CONFIG SAVED] {self._thresholds_config_path}")
        except Exception as e:
            print(f"[CONFIG SAVE ERROR] {e}")
    
    def _load_thresholds(self):
        """Load saved threshold slider values from disk"""
        try:
            if not os.path.exists(self._thresholds_config_path):
                return
            with open(self._thresholds_config_path, 'r') as f:
                config = json.load(f)
            
            if 'motion_threshold' in config:
                self.motion_slider.setValue(config['motion_threshold'])
            if 'min_area' in config:
                self.area_min_slider.setValue(config['min_area'])
            if 'max_area' in config:
                self.area_max_slider.setValue(config['max_area'])
            if 'min_aspect_ratio' in config:
                self.aspect_slider.setValue(config['min_aspect_ratio'])
            if 'kernel_size' in config:
                self.kernel_size_slider.setValue(config['kernel_size'])
            if 'smoothing' in config:
                self.smoothing_slider.setValue(config['smoothing'])
            if 'roi_thickness' in config:
                self.roi_thickness_slider.setValue(config['roi_thickness'])
            if 'binary_threshold' in config:
                # Migration: old configs used 0-1500 (intensity), new uses 0-500 (percentile %)
                # Old values > 500 get reset to 0 (Auto)
                val = config['binary_threshold']
                if val > 500:
                    val = 0
                self.binary_thresh_slider.setValue(val)
            if 'ema_alpha' in config:
                self.ema_alpha_slider.setValue(config['ema_alpha'])
            
            print(f"[CONFIG LOADED] {self._thresholds_config_path}")
        except Exception as e:
            print(f"[CONFIG LOAD ERROR] {e}")
    
    def _get_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Model", ".", "YOLO Models (*.pt)")
        if f:
            self.model_path = f
            self.mdl_lbl.setText(Path(f).name)
            self._check_ready()
  
    def _get_video(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Video", ".", "Video Files (*.mp4 *.avi *.mov)")
        if f:
            self.video_path = f
            self.input_mode = 'video'
            self.vid_path_lbl.setText(Path(f).name)
            # Clear image folder selection when video is chosen
            self.image_folder_path = None
            self.img_folder_lbl.setText("No Image Folder Selected")
            self._check_ready()

    def _get_image_folder(self):
        """Let the user pick a folder of images — treated as 60 FPS virtual video."""
        d = QFileDialog.getExistingDirectory(self, "Select Image Folder (treated as 60 FPS video)")
        if d:
            self.image_folder_path = d
            self.input_mode = 'images'
            self.img_folder_lbl.setText(Path(d).name)
            # Clear video selection when image folder is chosen
            self.video_path = None
            self.vid_path_lbl.setText("No Video Selected")
            self._check_ready()
    
    def _check_ready(self):
        source_ready = bool(
            (self.input_mode == 'video' and self.video_path) or
            (self.input_mode == 'images' and getattr(self, 'image_folder_path', None))
        )
        if self.model_path and source_ready:
            self.btn_run.setEnabled(True)
    
    def get_detection_thresholds(self):
        """Get current threshold values from sliders"""
        return {
            'motion_threshold': self.motion_slider.value() / 10.0,
            'min_area': self.area_min_slider.value(),
            'max_area': self.area_max_slider.value(),
            'min_aspect_ratio': self.aspect_slider.value() / 10.0,
            'kernel_size': self.kernel_size_slider.value(),
            'smoothing_frames': self.smoothing_slider.value(),
            'binary_threshold': self.binary_thresh_slider.value() / 10.0,  # 0-500 → 0.0-50.0%
            'pixel_darkness': self.pixel_darkness_slider.value() / 100.0,  # 0-100 → 0.0-1.0
            'ema_alpha': self.ema_alpha_slider.value() / 100.0             # 10-90 → 0.10-0.90
        }
    
    def _apply_thresholds(self):
        """Apply current slider values to running worker"""
        # Always save to disk, even if worker isn't running
        self._save_thresholds()
        
        if self.worker is not None:
            new_thresholds = self.get_detection_thresholds()
            self.worker.detection_thresholds = new_thresholds
            # Update smoothing responsiveness in real-time
            self.worker.history_size = new_thresholds['smoothing_frames']
            # Update ROI triangle thickness in real-time
            self.worker.roi_thickness = self.roi_thickness_slider.value()
            # Update ROI percentile in real-time
            self.worker.binary_threshold = self.binary_thresh_slider.value() / 10.0
            # Update pixel darkness in real-time
            self.worker.pixel_darkness = self.pixel_darkness_slider.value() / 100.0
            # Update antenna EMA alpha in real-time
            self.worker.ema_alpha = self.ema_alpha_slider.value() / 100.0
            pct_val = self.binary_thresh_slider.value() / 10.0
            pct_str = "Auto (Otsu)" if pct_val == 0 else f"Top {pct_val:.1f}%"
            ema_val = self.ema_alpha_slider.value() / 100.0
            self.lbl_stat.setText(f"✓ Thresholds Applied & Saved! (ROI: {pct_str}, EMA α: {ema_val:.2f})")
            print(f"[THRESHOLDS UPDATED] {new_thresholds}")
            print(f"[ROI THICKNESS UPDATED] {self.worker.roi_thickness} pixels")
            print(f"[ROI PERCENTILE UPDATED] {pct_str}")
            print(f"[EMA ALPHA UPDATED] {ema_val:.2f}")
    
    def _start(self):
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_apply_thresholds.setEnabled(True)  # Enable Apply button when video starts

        # Determine source path and mode
        if self.input_mode == 'images':
            source_path = getattr(self, 'image_folder_path', None)
            mode_label = "Image Folder @ 60 FPS"
        else:
            source_path = self.video_path
            mode_label = "Video File"

        self.lbl_stat.setText(f"Starting — {mode_label}...")
        print(f"[PIPELINE START] Mode: {self.input_mode}  |  Source: {source_path}")

        thresholds = self.get_detection_thresholds()
        self.thread = QThread()
        self.worker = InferenceWorker(
            self.model_path, source_path,
            thresholds=thresholds,
            input_mode=self.input_mode        # ← pass mode into the worker
        )
        self.worker.roi_thickness = self.roi_thickness_slider.value()  # Set initial ROI thickness
        self.worker.pixel_darkness = self.pixel_darkness_slider.value() / 100.0  # Set initial pixel darkness
        self.worker.ema_alpha = self.ema_alpha_slider.value() / 100.0  # Set initial EMA alpha
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.frame_processed.connect(self._update_img)
        self.worker.darkest_visualization.connect(self._update_darkest)
        self.worker.darkest_mask_visualization.connect(self._update_mask)
        self.worker.darkest_mask_no_keypoints.connect(self._update_mask_no_keypoints)
        self.worker.body_keypoints_roi_vectors.connect(self._update_body_keypoints_roi)
        self.worker.roi_bbox_body.connect(self._update_roi_bbox_body)
        self.worker.grayscale_body_roi.connect(self._update_grayscale_body_roi)
        self.worker.full_frame_darkest.connect(self._update_full_frame_darkest)
        self.worker.research_visualization.connect(self._update_research_visualization)  # NEW
        self.worker.metrics_updated.connect(self._update_unified_analysis)
        self.worker.metrics_updated.connect(self._update_antenna_dashboard)
        self.worker.progress.connect(self.lbl_stat.setText)
        self.worker.finished.connect(self._on_finish)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.thread.start()
    
    def _stop(self):
        if self.worker:
            self.worker.stop()
        self.btn_apply_thresholds.setEnabled(False)  # Disable Apply button when stopped
        self.thread.quit()
        self.thread.wait(1000)
    
    def _update_img(self, q_img):
        self.video_display.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.video_display.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
    
    def _update_darkest(self, q_img):
        self.darkest_display.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.darkest_display.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
    
    def _update_mask(self, q_img):
        self.mask_display.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.mask_display.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
    
    def _update_mask_no_keypoints(self, q_img):
        self.mask_no_keypoints_display.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.mask_no_keypoints_display.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
    
    def _update_body_keypoints_roi(self, q_img):
        self.body_keypoints_roi_display.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.body_keypoints_roi_display.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
    
    def _update_roi_bbox_body(self, q_img):
        self.roi_bbox_body_display.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.roi_bbox_body_display.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
    
    def _update_grayscale_body_roi(self, q_img):
        self.grayscale_body_roi_display.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.grayscale_body_roi_display.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
    
    def _update_full_frame_darkest(self, q_img):
        self.full_frame_darkest_display.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.full_frame_darkest_display.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
    
    def _update_research_visualization(self, research_data):
        """Update the For Research tab with morphological pipeline visualization"""
        self.research_widget.update_visualization(research_data)
    
    def _on_finish(self, path):
        self.lbl_stat.setText("✓ Analysis Complete")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_apply_thresholds.setEnabled(False)  # Disable Apply button
    
    def _on_error(self, error_msg):
        self.lbl_stat.setText(f"✗ Error: {error_msg[:30]}...")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_apply_thresholds.setEnabled(False)  # Disable Apply button
    

    
    def _get_progress_bar(self, percentage, width=20):
        """Generate ASCII progress bar"""
        filled = int(percentage / 5)
        empty = width - filled
        bar = '█' * filled + '░' * empty
        return f"[{bar}]"
    
    def closeEvent(self, e):
        self._save_thresholds()
        self._stop()
        super().closeEvent(e)

    def _toggle_trophallaxis_metrics(self):
        """Toggle trophallaxis metrics on/off"""
        self.trophallaxis_enabled = self.trophallaxis_toggle.isChecked()
        
        if self.trophallaxis_enabled:
            self.trophallaxis_toggle.setText("ON")
            self.trophallaxis_toggle.setStyleSheet("""
                QPushButton {
                    background: #2e7d32;
                    color: white;
                    border: 2px solid #4caf50;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QPushButton:pressed {
                    background: #1b5e20;
                }
            """)
        else:
            self.trophallaxis_toggle.setText("OFF")
            self.trophallaxis_toggle.setStyleSheet("""
                QPushButton {
                    background: #c62828;
                    color: white;
                    border: 2px solid #e53935;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QPushButton:pressed {
                    background: #6a1b1a;
                }
            """)
    
    def _toggle_antenna_analysis(self):
        """Toggle antenna analysis on/off"""
        self.antenna_analysis_enabled = self.antenna_toggle.isChecked()
        
        if self.antenna_analysis_enabled:
            self.antenna_toggle.setText("ON")
            self.antenna_toggle.setStyleSheet("""
                QPushButton {
                    background: #2e7d32;
                    color: white;
                    border: 2px solid #4caf50;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QPushButton:pressed {
                    background: #1b5e20;
                }
            """)
            self.antenna_analysis_display.setText(
                "╔════════════════════════════════════════╗\n"
                "║   ANTENNA CONTACT RATIO ANALYSIS       ║\n"
                "╚════════════════════════════════════════╝\n\n"
                "✓ ANTENNA ANALYSIS: ENABLED\n\n"
                "Waiting for trophallaxis events to analyze...\n\n"
                "This analysis will show:\n"
                "  • Right-Right (RR) contact patterns\n"
                "  • Cross-pattern (RL/LR) contacts\n"
                "  • Rare Left-Left (LL) interactions\n"
                "  • Per-event antenna dominance\n"
                "  • Colony health indicators\n"
                "  • Contact pattern statistics\n"
            )
        else:
            self.antenna_toggle.setText("OFF")
            self.antenna_toggle.setStyleSheet("""
                QPushButton {
                    background: #c62828;
                    color: white;
                    border: 2px solid #e53935;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QPushButton:pressed {
                    background: #6a1b1a;
                }
            """)
            self.antenna_analysis_display.setText(
                "╔════════════════════════════════════════╗\n"
                "║   ANTENNA CONTACT RATIO ANALYSIS       ║\n"
                "╚════════════════════════════════════════╝\n\n"
                "⏸️  ANTENNA ANALYSIS: DISABLED\n\n"
                "Toggle the ON/OFF switch above to enable\n"
                "antenna contact analysis.\n\n"
                "When enabled, you will see:\n"
                "  • Real-time antenna contact tracking\n"
                "  • RR/RL/LR/LL pattern breakdown\n"
                "  • Health indicators based on patterns\n"
                "  • Per-event antenna dominance ratios\n"
                "  • Contact type statistics\n"
            )
    def _toggle_antenna_dominance(self):
        """Toggle antenna dominance analysis on/off"""
        self.antenna_dominance_enabled = self.antenna_dominance_toggle.isChecked()
        
        if self.antenna_dominance_enabled:
            self.antenna_dominance_toggle.setText("ON")
            self.antenna_dominance_toggle.setStyleSheet("""
                QPushButton {
                    background: #2e7d32;
                    color: white;
                    border: 2px solid #4caf50;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QPushButton:pressed {
                    background: #1b5e20;
                }
            """)
        else:
            self.antenna_dominance_toggle.setText("OFF")
            self.antenna_dominance_toggle.setStyleSheet("""
                QPushButton {
                    background: #c62828;
                    color: white;
                    border: 2px solid #e53935;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QPushButton:pressed {
                    background: #6a1b1a;
                }
            """)    
    def _toggle_region_analysis(self):
        """Toggle 4-region analysis on/off"""
        self.region_analysis_enabled = self.region_toggle.isChecked()
        
        if self.region_analysis_enabled:
            self.region_toggle.setText("ON")
            self.region_toggle.setStyleSheet("""
                QPushButton {
                    background: #2e7d32;
                    color: white;
                    border: 2px solid #4caf50;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QPushButton:pressed {
                    background: #1b5e20;
                }
            """)
        else:
            self.region_toggle.setText("OFF")
            self.region_toggle.setStyleSheet("""
                QPushButton {
                    background: #c62828;
                    color: white;
                    border: 2px solid #e53935;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QPushButton:pressed {
                    background: #6a1b1a;
                }
            """)
    def _update_antenna_dashboard(self, metrics):
        """
        Update the modern antenna dashboard with latest metrics.
        Called from worker thread via signal.
        """
        frame_number = metrics.get('frame_number', 0)
        antenna_data = metrics.get('antenna_contacts', {})
        region_stats = metrics.get('region_stats', {})
        region_anomalies = metrics.get('region_anomalies', [])
        
        # Extract counts from antenna_data
        rr = antenna_data.get('rr_count', 0)
        rl = antenna_data.get('rl_count', 0)
        lr = antenna_data.get('lr_count', 0)
        ll = antenna_data.get('ll_count', 0)
        total_contacts = antenna_data.get('total_contacts', 0)
        frames_with_contact = antenna_data.get('frames_with_contact', 0)
        
        # =========================
        # KEY METRICS (Always Visible)
        # =========================
        window_coverage = (frames_with_contact / 300 * 100) if frames_with_contact > 0 else 0.0
        
        # R/L ratio
        right_usage = rr + rl
        left_usage = lr + ll
        rl_ratio = right_usage / left_usage if left_usage > 0 else 0.0
        
        self.antenna_cards['contact_activity'].setText(f"{total_contacts}")
        self.antenna_cards['window_coverage'].setText(f"{window_coverage:.1f}%")
        self.antenna_cards['antenna_balance'].setText(f"{rl_ratio:.2f}:1")
        
        # Color coding for balance
        if rl_ratio > 1.5 or rl_ratio < 0.67:
            self.antenna_cards['antenna_balance'].setStyleSheet(
                f"color: {BeeVisionTheme.WARNING}; border: none; font-weight: bold;"
            )
        else:
            self.antenna_cards['antenna_balance'].setStyleSheet(
                f"color: {BeeVisionTheme.TEXT_PRIMARY}; border: none; font-weight: bold;"
            )
        
        # =========================
        # SECTION 1: Directional Patterns
        # =========================
        cross_pattern = rl + lr
        same_side = rr + ll
        
        # Summary cards
        self.antenna_cards['cross_pattern_summary'].setText(f"{cross_pattern}")
        self.antenna_cards['same_side_summary'].setText(f"{same_side}")
        
        # Detailed cards
        self.antenna_cards['rr_frames'].setText(f"{rr}")
        self.antenna_cards['rl_frames'].setText(f"{rl}")
        self.antenna_cards['lr_frames'].setText(f"{lr}")
        self.antenna_cards['ll_frames'].setText(f"{ll}")
        
        self.antenna_cards['cross_pattern_detail'].setText(f"{cross_pattern}")
        self.antenna_cards['same_side_detail'].setText(f"{same_side}")
        
        # Progress bars
        total = max(rr + rl + lr + ll, 1)
        self.antenna_cards['pattern_distribution']['RR'].setValue(int(rr / total * 100))
        self.antenna_cards['pattern_distribution']['RR'].setFormat(f"{rr / total * 100:.1f}%")
        self.antenna_cards['pattern_distribution']['RL'].setValue(int(rl / total * 100))
        self.antenna_cards['pattern_distribution']['RL'].setFormat(f"{rl / total * 100:.1f}%")
        self.antenna_cards['pattern_distribution']['LR'].setValue(int(lr / total * 100))
        self.antenna_cards['pattern_distribution']['LR'].setFormat(f"{lr / total * 100:.1f}%")
        self.antenna_cards['pattern_distribution']['LL'].setValue(int(ll / total * 100))
        self.antenna_cards['pattern_distribution']['LL'].setFormat(f"{ll / total * 100:.1f}%")
        
        # =========================
        # SECTION 2: Symmetry & Lateralization
        # =========================
        right_pct = (right_usage / total * 100) if total > 0 else 0
        left_pct = (left_usage / total * 100) if total > 0 else 0
        
        # Lateralization index (-1 to +1)
        li = (right_usage - left_usage) / (right_usage + left_usage) if (right_usage + left_usage) > 0 else 0
        
        self.antenna_cards['lateralization_summary'].setText(f"{li:+.2f}")
        self.antenna_cards['usage_balance_summary'].setText(f"{rl_ratio:.2f}:1")
        
        self.antenna_cards['right_usage'].setText(f"{right_usage}")
        self.antenna_cards['left_usage'].setText(f"{left_usage}")
        self.antenna_cards['rl_ratio'].setText(f"{rl_ratio:.2f}:1")
        
        # Lateralization interpretation
        if abs(li) < 0.2:
            interp = "Balanced"
        elif li > 0:
            interp = f"Right bias ({abs(li):.2f})"
        else:
            interp = f"Left bias ({abs(li):.2f})"
        
        self.antenna_cards['lateralization_detail'].setText(f"{li:+.2f}")
        
        # Directional symmetry
        symmetry = rl / lr if lr > 0 else 0
        self.antenna_cards['directional_symmetry'].setText(f"{symmetry:.2f}:1")
        
        # =========================
        # SECTION 3: Regional Distribution
        # =========================
        if region_stats:
            # Find most/least contacted
            region_totals = {
                'PROTHORAX': region_stats.get('PROTHORAX', {}).get('total', 0),
                'MESOTHORAX': region_stats.get('MESOTHORAX', {}).get('total', 0),
                'METATHORAX': region_stats.get('METATHORAX', {}).get('total', 0),
                'ABDOMEN': region_stats.get('ABDOMEN', {}).get('total', 0)
            }
            
            most_region = max(region_totals, key=region_totals.get)
            least_region = min(region_totals, key=region_totals.get)
            
            self.antenna_cards['most_contacted'].setText(
                f"{most_region}\n{region_totals[most_region]} frames"
            )
            self.antenna_cards['least_contacted'].setText(
                f"{least_region}\n{region_totals[least_region]} frames"
            )
            self.antenna_cards['contact_rate'].setText(f"{window_coverage:.1f}%")
            
            # Regional distribution
            self.antenna_cards['region_distribution']['REGION 1: PROTHORAX'].setValue(
                int(region_totals['PROTHORAX'] / 300 * 100)
            )
            self.antenna_cards['region_distribution']['REGION 1: PROTHORAX'].setFormat(
                f"{region_totals['PROTHORAX']} ({region_totals['PROTHORAX'] / 300 * 100:.1f}%)"
            )
            
            self.antenna_cards['region_distribution']['REGION 2: MESOTHORAX'].setValue(
                int(region_totals['MESOTHORAX'] / 300 * 100)
            )
            self.antenna_cards['region_distribution']['REGION 2: MESOTHORAX'].setFormat(
                f"{region_totals['MESOTHORAX']} ({region_totals['MESOTHORAX'] / 300 * 100:.1f}%)"
            )
            
            self.antenna_cards['region_distribution']['REGION 3: METATHORAX'].setValue(
                int(region_totals['METATHORAX'] / 300 * 100)
            )
            self.antenna_cards['region_distribution']['REGION 3: METATHORAX'].setFormat(
                f"{region_totals['METATHORAX']} ({region_totals['METATHORAX'] / 300 * 100:.1f}%)"
            )
            
            self.antenna_cards['region_distribution']['REGION 4: ABDOMEN'].setValue(
                int(region_totals['ABDOMEN'] / 300 * 100)
            )
            self.antenna_cards['region_distribution']['REGION 4: ABDOMEN'].setFormat(
                f"{region_totals['ABDOMEN']} ({region_totals['ABDOMEN'] / 300 * 100:.1f}%)"
            )
            
            # Hierarchy (sorted)
            sorted_regions = sorted(region_totals.items(), key=lambda x: x[1], reverse=True)
            for i, (region, total) in enumerate(sorted_regions):
                label = f"Region {i+1}"
                if label in self.antenna_cards['regional_hierarchy']:
                    self.antenna_cards['regional_hierarchy'][label].setValue(
                        int(total / 300 * 100)
                    )
                    self.antenna_cards['regional_hierarchy'][label].setFormat(
                        f"{region}: {total / 300 * 100:.1f}%"
                    )
        
        # =========================
        # ANOMALY ALERTS
        # =========================
        # Clear existing anomaly widgets (except the "no anomaly" label)
        for i in reversed(range(self.antenna_cards['anomaly_container'].count())):
            widget = self.antenna_cards['anomaly_container'].itemAt(i).widget()
            if widget and widget != self.antenna_cards['no_anomaly_label']:
                widget.deleteLater()
        
        if region_anomalies and len(region_anomalies) > 0:
            # Hide "no anomalies" label
            self.antenna_cards['no_anomaly_label'].setVisible(False)
            
            # Add anomaly items
            for anomaly in region_anomalies[:3]:  # Show top 3
                region = anomaly['region']
                baseline = anomaly['baseline']
                current = anomaly['current']
                change = anomaly['change_pct']
                severity = anomaly['severity']
                
                icon = "🔴" if "SEVERE" in severity else "🟡"
                
                anomaly_label = QLabel(
                    f"{icon} {severity}: {region} Contact {'Spike' if change > 0 else 'Drop'}\n"
                    f"    Baseline: {baseline:.1f}%  │  Current: {current}  │  Change: {change:+.0f}%"
                )
                anomaly_label.setFont(QFont(BeeVisionTheme.FONT_FAMILY, 8))
                anomaly_label.setStyleSheet(f"color: {BeeVisionTheme.TEXT_PRIMARY}; border: none;")
                anomaly_label.setWordWrap(True)
                self.antenna_cards['anomaly_container'].addWidget(anomaly_label)
        else:
            # Show "no anomalies" label
            self.antenna_cards['no_anomaly_label'].setVisible(True)
    def _update_unified_analysis(self, metrics_data):
        """Update the unified analysis dashboard with all metrics"""
        
        # Extract all data
        total_events = metrics_data.get('total_events', 0)
        pending_events = metrics_data.get('pending_events', [])
        completed_events = metrics_data.get('completed_events', [])
        
        # Antenna data from independent tracker
        if self.worker and hasattr(self.worker, 'antenna_tracker'):
            antenna_data = self.worker.antenna_tracker.get_aggregate_antenna_metrics()
        else:
            antenna_data = {'total_contacts': 0}
        
        # Regional data
        region_stats = metrics_data.get('region_stats', {})
        region_anomalies = metrics_data.get('region_anomalies', [])
        
        frame_number = metrics_data.get('frame_number', 0)
        active_pairs = metrics_data.get('active_pairs', 0)
        
        # Build the unified dashboard
        lines = []
        
        # HEADER
        lines.append("╔══════════════════════════════════════════════════════════════════════════════╗")
        lines.append("║                      BEEVISION BEHAVIORAL ANALYSIS v2.0                      ║")
        lines.append("║                         Real-Time Tracking & Metrics                         ║")
        lines.append("╚══════════════════════════════════════════════════════════════════════════════╝")
        lines.append("")
        
        # SYSTEM STATUS
        fps_val = getattr(self.worker, 'frames_processed', 0) / (frame_number / 30.0) if frame_number > 0 else 0
        progress_pct = (frame_number / 36000) * 100 if frame_number < 36000 else 100
        progress_bar = self._get_progress_bar(progress_pct, width=10)
        
        lines.append("┌──────────────────────────────────────────────────────────────────────────────┐")
        lines.append("│ SYSTEM STATUS                                                                │")
        lines.append("├──────────────────────────────────────────────────────────────────────────────┤")
        lines.append(f"│ Frame: {frame_number:,} / 36,000  │  Time: {frame_number/30.0:.0f}s / 1200s  │  Progress: {progress_bar} {progress_pct:.0f}%  │")
        lines.append(f"│ Active Bees: {len(self.worker.keypoint_history) if self.worker else 0}          │  Total Pairs: {active_pairs}       │  FPS: {fps_val:.1f}                 │")
        lines.append(f"│ Device: {TORCH_DEVICE.upper()}         │  Status: ✓ RECORDING      │")
        lines.append("└──────────────────────────────────────────────────────────────────────────────┘")
        lines.append("")
        lines.append("")
        
        # SECTION 1: TROPHALLAXIS
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(" SECTION 1: TROPHALLAXIS DETECTION (Face-to-Face Food Sharing Events)")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("")
        
        avg_duration = np.mean([e.get('duration_seconds', 0) for e in completed_events]) if completed_events else 0
        longest_duration = max([e.get('duration_seconds', 0) for e in completed_events]) if completed_events else 0
        longest_pair = ""
        if completed_events:
            longest_event = max(completed_events, key=lambda e: e.get('duration_seconds', 0))
            longest_pair = f"(Bee {longest_event['bee_a_id']} ↔ Bee {longest_event['bee_b_id']})"
        
        lines.append("┌──────────────────────────────────────────────────────────────────────────────┐")
        lines.append("│ SUMMARY                                                                      │")
        lines.append("├──────────────────────────────────────────────────────────────────────────────┤")
        lines.append(f"│ Confirmed Events (≥10s):    {total_events:<50}│")
        lines.append(f"│ Pending Events (<10s):      {len(pending_events):<50}│")
        lines.append(f"│ Average Duration:           {avg_duration:.1f} seconds{' '*37}│")
        lines.append(f"│ Longest Event:              {longest_duration:.1f} seconds {longest_pair:<30}│")
        lines.append("└──────────────────────────────────────────────────────────────────────────────┘")
        lines.append("")
        
        # PENDING EVENTS
        if pending_events:
            lines.append("┌──────────────────────────────────────────────────────────────────────────────┐")
            lines.append("│ PENDING EVENTS (Monitoring for confirmation)                                │")
            lines.append("├──────────────────────────────────────────────────────────────────────────────┤")
            lines.append("│                                                                              │")
            
            for event in pending_events[:2]:  # Show top 2
                bee_a = event['bee_a_id']
                bee_b = event['bee_b_id']
                duration = event.get('duration_seconds', 0)
                progress = int(duration * 10)
                progress_bar = '█' * progress + '░' * (10 - progress)
                
                straightness_a = event.get('straightness_a', 0)
                straightness_b = event.get('straightness_b', 0)
                angle_diff = event.get('angle_diff', 0)
                head_distance = event.get('head_distance', 0)
                
                contacts = event.get('antenna_contacts', {})
                rr = contacts.get('RR', 0)
                rl = contacts.get('RL', 0)
                lr = contacts.get('LR', 0)
                ll = contacts.get('LL', 0)
                
                distances = event.get('antenna_distances', {})
                closest = min(distances.values()) if distances else 999
                
                lines.append(f"│ ⏱️  Bee {bee_a} ↔ Bee {bee_b}{' '*58}│")
                lines.append(f"│ ├─ Duration:          {duration:.1f}s / 10.0s        {progress_bar} {progress*10}%{' '*11}│")
                lines.append(f"│ ├─ Body Straightness: {straightness_a:.2f}, {straightness_b:.2f}          {'✓' if straightness_a >= 0.70 and straightness_b >= 0.70 else '✗'} Pass (≥0.70){' '*10}│")
                lines.append(f"│ ├─ Angular Offset:    {angle_diff:.1f}°              {'✓' if 120 <= angle_diff <= 240 else '✗'} Pass (120-240°){' '*7}│")
                lines.append(f"│ ├─ Head Distance:     {head_distance:.1f}px              {'✓' if head_distance <= 150 else '✗'} Pass (0-150px){' '*8}│")
                lines.append(f"│ ├─ Antenna Contact:   {'Yes' if closest < 15 else 'No'} ({closest:.1f}px)         {'✓' if closest < 15 else '✗'} Active{' '*17}│")
                lines.append(f"│ └─ Contact Types:     RR: {rr}, RL: {rl}, LR: {lr}, LL: {ll}{' '*25}│")
                lines.append("│                                                                              │")
            
            lines.append("└──────────────────────────────────────────────────────────────────────────────┘")
        else:
            lines.append("┌──────────────────────────────────────────────────────────────────────────────┐")
            lines.append("│ PENDING EVENTS: None                                                         │")
            lines.append("└──────────────────────────────────────────────────────────────────────────────┘")
        
        lines.append("")
        
        # CONFIRMED EVENTS
        if completed_events:
            lines.append("┌──────────────────────────────────────────────────────────────────────────────┐")
            lines.append("│ CONFIRMED EVENTS (Last 3)                                                   │")
            lines.append("├──────────────────────────────────────────────────────────────────────────────┤")
            lines.append("│                                                                              │")
            
            for event in completed_events[-3:]:
                bee_a = event['bee_a_id']
                bee_b = event['bee_b_id']
                duration = event.get('duration_seconds', 0)
                straightness = (event.get('straightness_a', 0) + event.get('straightness_b', 0)) / 2
                angle = event.get('angle_diff', 0)
                
                contacts = event.get('antenna_contacts', {})
                total = sum(contacts.values()) if contacts else 1
                rr_pct = (contacts.get('RR', 0) / total * 100) if total > 0 else 0
                rl_pct = (contacts.get('RL', 0) / total * 100) if total > 0 else 0
                lr_pct = (contacts.get('LR', 0) / total * 100) if total > 0 else 0
                ll_pct = (contacts.get('LL', 0) / total * 100) if total > 0 else 0
                
                end_frame = event.get('end_frame', 0)
                
                lines.append(f"│ ✓ Bee {bee_a} ↔ Bee {bee_b}  │  {duration:.1f}s  │  Straightness: {straightness:.2f}  │  Angle: {angle:.1f}°{' '*10}│")
                lines.append(f"│   Contacts: RR: {rr_pct:.0f}%, RL: {rl_pct:.0f}%, LR: {lr_pct:.0f}%, LL: {ll_pct:.0f}%  │  Ended: Frame {end_frame:,}{' '*9}│")
                lines.append("│                                                                              │")
            
            lines.append("└──────────────────────────────────────────────────────────────────────────────┘")
        else:
            lines.append("┌──────────────────────────────────────────────────────────────────────────────┐")
            lines.append("│ CONFIRMED EVENTS: None yet                                                   │")
            lines.append("└──────────────────────────────────────────────────────────────────────────────┘")
        
        lines.append("")
        lines.append("")
        
        # SECTION 2: ANTENNA CONTACTS
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(" SECTION 2: ANTENNA CONTACTS - LIFETIME AGGREGATE (All Frames, All Pairs)")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("")
        
        if antenna_data['total_contacts'] > 0:
            total_contacts = antenna_data['total_contacts']
            contact_rate = (total_contacts / frame_number * 100) if frame_number > 0 else 0
            avg_rate = total_contacts / (frame_number / 30.0) if frame_number > 0 else 0
            
            lines.append("┌──────────────────────────────────────────────────────────────────────────────┐")
            lines.append("│ OVERALL STATISTICS                                                           │")
            lines.append("├──────────────────────────────────────────────────────────────────────────────┤")
            lines.append("│                                                                              │")
            lines.append(f"│  Total Frames with Contact:     {total_contacts:,} frames ({contact_rate:.1f}% of {frame_number:,} total){' '*8}│")
            lines.append(f"│  Average Contact Rate:          {avg_rate:.1f} contacts/second{' '*30}│")
            lines.append("│                                                                              │")
            lines.append("└──────────────────────────────────────────────────────────────────────────────┘")
            lines.append("")
            
            # Contact type breakdown
            rr_count = antenna_data.get('rr_count', 0)
            rl_count = antenna_data.get('rl_count', 0)
            lr_count = antenna_data.get('lr_count', 0)
            ll_count = antenna_data.get('ll_count', 0)
            
            rr_pct = (rr_count / total_contacts * 100) if total_contacts > 0 else 0
            rl_pct = (rl_count / total_contacts * 100) if total_contacts > 0 else 0
            lr_pct = (lr_count / total_contacts * 100) if total_contacts > 0 else 0
            ll_pct = (ll_count / total_contacts * 100) if total_contacts > 0 else 0
            
            rr_bar = self._get_progress_bar(rr_pct, width=20)
            rl_bar = self._get_progress_bar(rl_pct, width=20)
            lr_bar = self._get_progress_bar(lr_pct, width=20)
            ll_bar = self._get_progress_bar(ll_pct, width=20)
            
            lines.append("┌──────────────────────────────────────────────────────────────────────────────┐")
            lines.append("│ CONTACT TYPE BREAKDOWN                                                       │")
            lines.append("├──────────────────────────────────────────────────────────────────────────────┤")
            lines.append("│                                                                              │")
            lines.append(f"│  Right-Right (RR):   {rr_count:,} frames  {rr_bar}  {rr_pct:5.1f}%{' '*8}│")
            lines.append(f"│  Right-Left  (RL):   {rl_count:,} frames  {rl_bar}  {rl_pct:5.1f}%{' '*8}│")
            lines.append(f"│  Left-Right  (LR):   {lr_count:,} frames  {lr_bar}  {lr_pct:5.1f}%{' '*8}│")
            lines.append(f"│  Left-Left   (LL):   {ll_count:,} frames  {ll_bar}  {ll_pct:5.1f}%{' '*8}│")
            lines.append("│                                                                              │")
            lines.append("└──────────────────────────────────────────────────────────────────────────────┘")
            lines.append("")
            
            # Lateralization metrics
            right_usage = rr_count + rl_count
            left_usage = ll_count + lr_count
            right_pct = (right_usage / total_contacts * 100) if total_contacts > 0 else 0
            left_pct = (left_usage / total_contacts * 100) if total_contacts > 0 else 0
            rl_ratio = right_usage / left_usage if left_usage > 0 else 0
            
            cross_pattern = rl_count + lr_count
            same_side = rr_count + ll_count
            cross_pct = (cross_pattern / total_contacts * 100) if total_contacts > 0 else 0
            same_pct = (same_side / total_contacts * 100) if total_contacts > 0 else 0
            
            symmetry = rl_count / lr_count if lr_count > 0 else (rl_count if rl_count > 0 else 1.0)
            li = (rr_count - ll_count) / total_contacts if total_contacts > 0 else 0
            
            lines.append("┌──────────────────────────────────────────────────────────────────────────────┐")
            lines.append("│ LATERALIZATION METRICS                                                       │")
            lines.append("├──────────────────────────────────────────────────────────────────────────────┤")
            lines.append("│                                                                              │")
            lines.append(f"│  Right Antenna Usage:     {right_usage:,} frames ({right_pct:.1f}%)  [RR + RL]{' '*17}│")
            lines.append(f"│  Left Antenna Usage:      {left_usage:,} frames ({left_pct:.1f}%)  [LL + LR]{' '*18}│")
            lines.append(f"│  Right/Left Ratio:        {rl_ratio:.2f} : 1{' '*46}│")
            lines.append("│                                                                              │")
            lines.append(f"│  Cross-Pattern Usage:     {cross_pattern:,} frames ({cross_pct:.1f}%)  [RL + LR]{' '*17}│")
            lines.append(f"│  Same-Side Usage:         {same_side:,} frames ({same_pct:.1f}%)  [RR + LL]{' '*18}│")
            lines.append("│                                                                              │")
            lines.append(f"│  Directional Symmetry:    {symmetry:.2f}  [RL/LR ratio]{' '*32}│")
            lines.append(f"│  Lateralization Index:    {li:+.2f}  [Range: -1.0 to +1.0]{' '*25}│")
            lines.append("│                                                                              │")
            lines.append("└──────────────────────────────────────────────────────────────────────────────┘")
        else:
            lines.append("┌──────────────────────────────────────────────────────────────────────────────┐")
            lines.append("│ Waiting for antenna contact data...                                         │")
            lines.append("└──────────────────────────────────────────────────────────────────────────────┘")
        
        lines.append("")
        lines.append("")
        
        # SECTION 3: REGIONAL ANALYSIS
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(" SECTION 3: REGIONAL ANALYSIS (4-Region Anatomical Distribution)")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("")
        
        if region_stats:
            # Window summary
            total_region_contacts = sum(stats['total'] for stats in region_stats.values())
            region_right = sum(stats['right'] for stats in region_stats.values())
            region_left = sum(stats['left'] for stats in region_stats.values())
            region_ratio = region_right / region_left if region_left > 0 else 0
            contact_rate = (total_region_contacts / 300 * 100)
            
            lines.append("┌──────────────────────────────────────────────────────────────────────────────┐")
            lines.append("│ CURRENT WINDOW (Last 10 seconds = 300 frames @ 30fps)                       │")
            lines.append("├──────────────────────────────────────────────────────────────────────────────┤")
            lines.append("│                                                                              │")
            lines.append(f"│  Total Contact Frames:    {total_region_contacts} / 300  ({contact_rate:.1f}% contact rate){' '*21}│")
            lines.append(f"│  Right Antenna:           {region_right} frames  ({region_right/total_region_contacts*100 if total_region_contacts > 0 else 0:.1f}%){' '*30}│")
            lines.append(f"│  Left Antenna:            {region_left} frames   ({region_left/total_region_contacts*100 if total_region_contacts > 0 else 0:.1f}%){' '*31}│")
            lines.append(f"│  Right/Left Ratio:        {region_ratio:.2f} : 1{' '*46}│")
            lines.append("│                                                                              │")
            lines.append("└──────────────────────────────────────────────────────────────────────────────┘")
            lines.append("")
            
            # By region
            lines.append("┌──────────────────────────────────────────────────────────────────────────────┐")
            lines.append("│ BY BODY REGION (Antenna-to-Body Contact Distribution)                       │")
            lines.append("├──────────────────────────────────────────────────────────────────────────────┤")
            lines.append("│                                                                              │")
            
            for region_name in ['PROTHORAX', 'MESOTHORAX', 'METATHORAX', 'ABDOMEN']:
                stats = region_stats.get(region_name, {'right': 0, 'left': 0, 'total': 0})
                right = stats['right']
                left = stats['left']
                total = stats['total']
                right_pct = (right / 300 * 100)
                left_pct = (left / 300 * 100)
                total_pct = (total / 300 * 100)
                
                region_label = {
                    'PROTHORAX': 'REGION 1: PROTHORAX (Head → Thorax Line 1)',
                    'MESOTHORAX': 'REGION 2: MESOTHORAX (Thorax Line 1 → Line 2)',
                    'METATHORAX': 'REGION 3: METATHORAX (Thorax Line 2 → Line 3)',
                    'ABDOMEN': 'REGION 4: ABDOMEN (Thorax Line 3 → Abdomen Tip)'
                }[region_name]
                
                lines.append("│  ┌──────────────────────────────────────────────────────────────────────┐   │")
                lines.append(f"│  │ {region_label:<68}│   │")
                lines.append("│  ├──────────────────────────────────────────────────────────────────────┤   │")
                lines.append(f"│  │  Right: {right} frames ({right_pct:.1f}%)   │  Left: {left} frames ({left_pct:.1f}%){' '*15}│   │")
                lines.append(f"│  │  Total: {total} frames ({total_pct:.1f}%){' '*39}│   │")
                lines.append("│  └──────────────────────────────────────────────────────────────────────┘   │")
                lines.append("│                                                                              │")
            
            lines.append("└──────────────────────────────────────────────────────────────────────────────┘")
            lines.append("")
            
            # Hierarchy
            region_totals = [(name, region_stats.get(name, {'total': 0})['total']) for name in ['PROTHORAX', 'MESOTHORAX', 'METATHORAX', 'ABDOMEN']]
            region_totals.sort(key=lambda x: x[1], reverse=True)
            
            lines.append("┌──────────────────────────────────────────────────────────────────────────────┐")
            lines.append("│ REGIONAL HIERARCHY                                                           │")
            lines.append("├──────────────────────────────────────────────────────────────────────────────┤")
            lines.append("│                                                                              │")
            
            if region_totals[0][1] > 0:
                lines.append(f"│  Most Contacted:  {region_totals[0][0]} ({region_totals[0][1]/300*100:.1f}% of all contacts){' '*20}│")
                lines.append(f"│  Least Contacted: {region_totals[-1][0]} ({region_totals[-1][1]/300*100:.1f}% of all contacts){' '*21}│")
            lines.append("│                                                                              │")
            
            for i, (name, total) in enumerate(region_totals, 1):
                pct = total / 300 * 100
                bar = self._get_progress_bar(pct, width=20)
                lines.append(f"│    {i}. {name:<12} {bar}  {pct:5.1f}%  ({total}/300 frames){' '*10}│")
            
            lines.append("│                                                                              │")
            lines.append("└──────────────────────────────────────────────────────────────────────────────┘")
            lines.append("")
            
            # Anomalies
            lines.append("┌──────────────────────────────────────────────────────────────────────────────┐")
            lines.append("│ ANOMALY DETECTION (vs Baseline)                                             │")
            lines.append("├──────────────────────────────────────────────────────────────────────────────┤")
            lines.append("│                                                                              │")
            
            if region_anomalies:
                for anomaly in region_anomalies[:3]:
                    region = anomaly['region']
                    baseline = anomaly['baseline']
                    current = anomaly['current']
                    change = anomaly['change_pct']
                    severity = anomaly['severity']
                    
                    lines.append(f"│  {severity} {region}:{' '*58}│")
                    lines.append(f"│      Baseline: {baseline:.1f}  │  Current: {current}  │  Change: {change:+.0f}%{' '*20}│")
                    lines.append("│                                                                              │")
            else:
                lines.append("│  ✓ No anomalies detected in current window                                  │")
                lines.append("│                                                                              │")
            
            lines.append("└──────────────────────────────────────────────────────────────────────────────┘")
        else:
            lines.append("┌──────────────────────────────────────────────────────────────────────────────┐")
            lines.append("│ Waiting for regional contact data...                                        │")
            lines.append("└──────────────────────────────────────────────────────────────────────────────┘")
        
        lines.append("")
        lines.append("")
        
        # FOOTER
        lines.append("╔══════════════════════════════════════════════════════════════════════════════╗")
        lines.append("║                           End of Analysis Dashboard                          ║")
        lines.append(f"║                     Last Updated: Frame {frame_number:,} @ {frame_number/30.0:.1f}s{' '*21}║")
        lines.append("╚══════════════════════════════════════════════════════════════════════════════╝")
        
        # Display
        self.unified_analysis_display.setText('\n'.join(lines))
    def closeEvent(self, e):
        self._save_thresholds()
        self._stop()
        super().closeEvent(e)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    p = app.palette()
    p.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    p.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    app.setPalette(p)
    
    win = KeypointViewerGUI()
    win.show()
    sys.exit(app.exec())
