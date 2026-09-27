"""
main_window.py
Fetal Ultrasound Facial Measurement Tool — Main Window Logic
"""

import os
import sys
import json
import datetime
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from PyQt5 import uic
from PyQt5.QtWidgets import (
    QDialog, QFileDialog, QMessageBox, QApplication, QGraphicsScene
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

from inference_engine import (
    FetalMeasurementEngine,
    build_labelme_json,
    build_info_image,
    KPT_NAMES, CLI_NAMES, NORMAL_RANGE,
)

# ─────────────────────────────────────────────
# Supported image file extensions
# ─────────────────────────────────────────────
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

# ─────────────────────────────────────────────
# Model paths (relative to the tool/ directory)
# ─────────────────────────────────────────────
# Place model weights in the paths below as described in README.md.
BASE_DIR = Path(__file__).parent

YOLO_MODEL_PATHS = [
    BASE_DIR / "models" / "detect" / f"yolo_fold{i}_best.pt"
    for i in range(1, 6)
]

VITPOSE_MODEL_PATHS = [
    BASE_DIR / "models" / "keypoint" / f"vitpose_fold{i}_best.pth"
    for i in range(1, 6)
]


# ─────────────────────────────────────────────
# Background worker thread for prediction
# ─────────────────────────────────────────────
class PredictWorker(QThread):
    """Runs inference in a background thread to keep the UI responsive."""

    progress     = pyqtSignal(int, int, str)  # (current_idx, total, image_name)
    result_ready = pyqtSignal(int, dict)       # (image_idx, result_dict)
    finished     = pyqtSignal()
    error        = pyqtSignal(str)

    def __init__(self, engine: FetalMeasurementEngine, img_paths: list):
        super().__init__()
        self.engine    = engine
        self.img_paths = img_paths

    def run(self):
        total = len(self.img_paths)
        for idx, path in enumerate(self.img_paths):
            try:
                self.progress.emit(idx + 1, total, Path(path).name)
                img_bgr = cv2.imread(str(path))
                if img_bgr is None:
                    self.error.emit(f"Failed to read image: {path}")
                    continue
                result = self.engine.predict(img_bgr)
                result['img_path'] = str(path)
                result['img_bgr']  = img_bgr
                # Re-render vis_img with correct filename in overlay
                from inference_engine import (draw_roi, draw_keypoints_and_geometry,
                                               draw_keypoint_circles,
                                               draw_measurements_on_image,
                                               COLOR_PRED)
                vis = img_bgr.copy()
                if result.get('roi_box') is not None:
                    draw_roi(vis, result['roi_box'])
                if result.get('kpt_vis') is not None and result['kpt_vis'].sum() > 0:
                    draw_keypoints_and_geometry(vis, result['kpt_coords'],
                                                result['kpt_vis'], color=COLOR_PRED)
                    draw_keypoint_circles(vis, result['kpt_coords'],
                                          result['kpt_vis'], color=COLOR_PRED, radius=3)
                draw_measurements_on_image(vis, result.get('clinical', {}),
                                           result.get('kpt_coords'),
                                           result.get('kpt_vis'),
                                           img_name=Path(path).name)
                result['vis_img'] = vis
                self.result_ready.emit(idx, result)
            except Exception:
                self.error.emit(
                    f"Error predicting {Path(path).name}:\n{traceback.format_exc()}")
        self.finished.emit()


# ─────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────
class MainWindow(QDialog):

    def __init__(self):
        super().__init__()

        # Load UI layout from file
        uic.loadUi(str(BASE_DIR / "main_window.ui"), self)
        self.setWindowTitle("Fetal Ultrasound Facial Measurement Tool")

        # State variables
        self.img_paths:   list = []   # all loaded image paths
        self.results:     list = []   # prediction result for each image (or None)
        self.current_idx: int  = -1   # index of the currently displayed image
        self.engine: FetalMeasurementEngine = None
        self._worker: PredictWorker = None

        # Connect buttons to handlers
        self.openfile.clicked.connect(self.on_openfile)
        self.predict.clicked.connect(self.on_predict)
        self.output_labelme.clicked.connect(self.on_output_labelme)
        self.output_info_image.clicked.connect(self.on_output_info_image)

        # Enable keyboard focus for A/D navigation
        self.setFocusPolicy(Qt.StrongFocus)

        # Set up the graphics scene for image display
        self._scene = QGraphicsScene()
        self.graphicsView.setScene(self._scene)

        # Welcome message
        self._log("Welcome to the Fetal Ultrasound Facial Measurement Tool.\n"
                  "Click 'Open file/folder' to load images, then click 'Predict' to run inference.")

        # Attempt to load models on startup
        self._try_load_engine()

    # ─────────────────────────────────────────
    # Engine Initialisation
    # ─────────────────────────────────────────
    def _try_load_engine(self):
        """Load model weights. Warns if any weight file is missing."""
        yolo_exist = all(p.exists() for p in YOLO_MODEL_PATHS)
        vit_exist  = all(p.exists() for p in VITPOSE_MODEL_PATHS)

        if not yolo_exist or not vit_exist:
            missing = []
            if not yolo_exist:
                missing.append("YOLO detection models  (models/detect/yolo_fold1~5_best.pt)")
            if not vit_exist:
                missing.append("ViTPose keypoint models (models/keypoint/vitpose_fold1~5_best.pth)")
            self._log("[WARNING] The following model files are missing:\n"
                      + "\n".join(f"  • {m}" for m in missing))
            self._log("Please follow README.md to place the model weights, then restart the application.")
            return

        try:
            self._log("Loading models, please wait...")
            QApplication.processEvents()
            self.engine = FetalMeasurementEngine(
                yolo_model_paths=[str(p) for p in YOLO_MODEL_PATHS],
                vitpose_model_paths=[str(p) for p in VITPOSE_MODEL_PATHS],
            )
            self._log("Models loaded successfully. Ready to predict.")
        except Exception:
            self._log(f"[ERROR] Failed to load models:\n{traceback.format_exc()}")

    # ─────────────────────────────────────────
    # Open File / Folder
    # ─────────────────────────────────────────
    def on_openfile(self):
        while True:
            # Ask the user whether to open a single file or a folder
            msg = QMessageBox(self)
            msg.setWindowTitle("Open Images")
            msg.setText(
                "How would you like to load images?\n\n"
                "Click 'Image'   -> Select a single image file\n"
                "Click 'Folder'  -> Select a folder of images"
            )

            yes_button = msg.addButton(QMessageBox.Yes)
            no_button = msg.addButton(QMessageBox.No)
            cancel_button = msg.addButton(QMessageBox.Cancel)

            yes_button.setText("Image")
            no_button.setText("Folder")

            msg.exec_()
            reply = msg.clickedButton()
        
            if reply == cancel_button:
                return

            if reply == yes_button:
                # Single file mode
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "Select Image File", "",
                    "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)"
                )
                if not file_path:
                    return

                if Path(file_path).suffix.lower() not in IMAGE_EXTENSIONS:
                    QMessageBox.warning(self, "Load Failed",
                                        "File/folder load in fail, please select again.")
                    continue  # retry

                self.img_paths = [file_path]
                break

            else:
                # Folder mode — collect all supported images in the directory
                folder_path = QFileDialog.getExistingDirectory(
                    self, "Select Image Folder", "")
                if not folder_path:
                    return

                found = sorted([
                    str(p) for p in Path(folder_path).iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                ])

                if not found:
                    QMessageBox.warning(self, "Load Failed",
                                        "File/folder load in fail, please select again.")
                    continue  # retry

                self.img_paths = found
                break

        # Reset state for the newly loaded images
        self.results     = [None] * len(self.img_paths)
        self.current_idx = 0

        self._log(f"Loaded {len(self.img_paths)} image(s).")
        self._display_current()

    # ─────────────────────────────────────────
    # Predict
    # ─────────────────────────────────────────
    def on_predict(self):
        if not self.img_paths:
            QMessageBox.information(self, "Notice", "Please load images first.")
            return

        if self.engine is None:
            QMessageBox.warning(self, "Model Not Loaded",
                                "Models are not loaded. "
                                "Please place the weight files in the correct paths and restart.")
            return

        # Disable buttons while prediction is running
        self._set_buttons_enabled(False)
        self._log(f"Starting prediction for {len(self.img_paths)} image(s)...")

        # Clear previous results
        self.results = [None] * len(self.img_paths)

        self._worker = PredictWorker(self.engine, self.img_paths)
        self._worker.progress.connect(self._on_predict_progress)
        self._worker.result_ready.connect(self._on_result_ready)
        self._worker.finished.connect(self._on_predict_finished)
        self._worker.error.connect(self._on_predict_error)
        self._worker.start()

    def _on_predict_progress(self, current, total, name):
        self._log(f"  [{current}/{total}] Predicting: {name}")

    def _on_result_ready(self, idx, result):
        self.results[idx] = result
        # Refresh display if this is the currently shown image
        if idx == self.current_idx:
            self._display_current()

    def _on_predict_finished(self):
        self._set_buttons_enabled(True)
        self._save_excel()
        QMessageBox.information(self, "Done", "All images are predicted.")
        self._log("All images predicted.")
        # Show the first result
        if self.results and self.results[0] is not None:
            self.current_idx = 0
            self._display_current()

    def _on_predict_error(self, msg):
        self._log(f"[ERROR] {msg}")

    # ─────────────────────────────────────────
    # Save Excel Report
    # ─────────────────────────────────────────
    def _save_excel(self):
        """Save prediction results to a timestamped Excel file in output/."""
        output_dir = BASE_DIR / "output"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        xlsx_path = output_dir / f"prediction_{timestamp}.xlsx"

        records = []
        for i, (path, result) in enumerate(zip(self.img_paths, self.results)):
            row = {
                'Index':       i + 1,
                'Image_Name':  Path(path).name,
                'Image_Path':  path,
                'ROI_x1': np.nan, 'ROI_y1': np.nan,
                'ROI_x2': np.nan, 'ROI_y2': np.nan,
                'Det_Success': False,
            }
            if result is not None:
                row['Det_Success'] = result.get('det_success', False)
                roi = result.get('roi_box')
                if roi is not None:
                    row['ROI_x1'], row['ROI_y1'] = roi[0], roi[1]
                    row['ROI_x2'], row['ROI_y2'] = roi[2], roi[3]

                # Keypoint coordinates
                kpt = result.get('kpt_coords')
                vis = result.get('kpt_vis')
                if kpt is not None and vis is not None:
                    for j, name in enumerate(KPT_NAMES):
                        row[f'{name}_x'] = float(kpt[j, 0]) if vis[j] > 0 else np.nan
                        row[f'{name}_y'] = float(kpt[j, 1]) if vis[j] > 0 else np.nan

                # Clinical indicators
                clinical = result.get('clinical', {})
                for k in CLI_NAMES:
                    row[k] = clinical.get(k, np.nan)

                row['Diagnosis'] = result.get('diagnosis', '')

            records.append(row)

        df = pd.DataFrame(records)
        try:
            with pd.ExcelWriter(str(xlsx_path), engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Prediction_Results', index=False)
            self._log(f"Results saved to: {xlsx_path}")
        except Exception as e:
            self._log(f"[ERROR] Failed to save Excel: {e}")

    # ─────────────────────────────────────────
    # Output LabelMe JSON
    # ─────────────────────────────────────────
    def on_output_labelme(self):
        if not any(r is not None for r in self.results):
            QMessageBox.information(self, "Notice", "Please run prediction first.")
            return

        output_dir = BASE_DIR / "labelme_output"
        output_dir.mkdir(exist_ok=True)

        count = 0
        for path, result in zip(self.img_paths, self.results):
            if result is None:
                continue
            img_bgr = result.get('img_bgr')

            if img_bgr is None:
                img_bgr = cv2.imread(str(path))
            h, w = img_bgr.shape[:2] if img_bgr is not None else (0, 0)

            lm_json = build_labelme_json(
                img_path=path,
                img_shape=(h, w),
                roi_box=result.get('roi_box'),
                kpt_coords=result.get('kpt_coords'),
                vis_mask=result.get('kpt_vis'),
                clinical=result.get('clinical', {}),
            )

            json_path = output_dir / (Path(path).stem + ".json")
            with open(str(json_path), 'w', encoding='utf-8') as f:
                json.dump(lm_json, f, ensure_ascii=False, indent=2)
            count += 1

        self._log(f"LabelMe JSON exported: {count} file(s) -> {output_dir}")
        QMessageBox.information(self, "Done",
                                f"Exported {count} LabelMe JSON file(s) to\n{output_dir}")

    # ─────────────────────────────────────────
    # Output Info Image
    # ─────────────────────────────────────────
    def on_output_info_image(self):
        if not any(r is not None for r in self.results):
            QMessageBox.information(self, "Notice", "Please run prediction first.")
            return

        output_dir = BASE_DIR / "info_image"
        output_dir.mkdir(exist_ok=True)

        count = 0
        for path, result in zip(self.img_paths, self.results):
            if result is None:
                continue
            img_bgr = result.get('img_bgr')

            if img_bgr is None:
                img_bgr = cv2.imread(str(path))
            if img_bgr is None:
                continue

            combined = build_info_image(
                img_bgr=img_bgr,
                vis_img=result.get('vis_img', img_bgr),
                clinical=result.get('clinical', {}),
                diagnosis=result.get('diagnosis', ''),
                img_name=Path(path).name,
            )

            out_path = output_dir / (Path(path).stem + "_info.png")
            cv2.imwrite(str(out_path), combined)
            count += 1

        self._log(f"Info images exported: {count} file(s) -> {output_dir}")
        QMessageBox.information(self, "Done",
                                f"Exported {count} info image(s) to\n{output_dir}")

    # ─────────────────────────────────────────
    # Display Current Image
    # ─────────────────────────────────────────
    def _display_current(self):
        """Refresh the graphicsView and textBrowser for the current image index."""
        if not self.img_paths or self.current_idx < 0:
            return

        idx = self.current_idx
        path = self.img_paths[idx]
        result = self.results[idx] if idx < len(self.results) else None

        # ─────────────────────────────────────
        # Select image to display
        # ─────────────────────────────────────
        display_img = None

        if result is not None:
            # Prefer annotated image
            display_img = result.get('vis_img')

            # If annotated image is unavailable, use original image
            if display_img is None:
                display_img = result.get('img_bgr')

        # If no image exists in result, read it from disk
        if display_img is None:
            display_img = cv2.imread(str(path))

        # ─────────────────────────────────────
        # Display image
        # ─────────────────────────────────────
        if display_img is not None:
            self._show_image(display_img)

        # ─────────────────────────────────────
        # Update text browser
        # ─────────────────────────────────────
        if result is not None:
            text = self._format_result_text(idx, path, result)
        else:
            text = (
                f"[{idx+1}/{len(self.img_paths)}] "
                f"{Path(path).name}\n"
                "Not yet predicted."
            )

        self.textBrowser.setPlainText(text)

    def _show_image(self, img_bgr: np.ndarray):
        """Display an OpenCV BGR image in the QGraphicsView."""
        h, w = img_bgr.shape[:2]
        rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        qimg  = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self._scene.clear()
        self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(0, 0, w, h)
        self.graphicsView.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def _format_result_text(self, idx: int, path: str, result: dict) -> str:
        """Format measurement values and diagnostic suggestions as plain text."""
        lines = [
            f"[{idx+1}/{len(self.img_paths)}] {Path(path).name}",
            f"Detection: {'Success' if result.get('det_success') else 'ROI not detected'}",
            "",
            "── Clinical Measurements ──",
        ]

        clinical = result.get('clinical', {})
        for k in CLI_NAMES:
            val  = clinical.get(k, float('nan'))
            unit = "deg" if k != 'PL_Dist' else "px"
            nr   = NORMAL_RANGE.get(k, '')
            if val != val:  # isnan
                lines.append(f"  {k}: N/A")
            elif k == 'PL_Dist':
                lines.append(f"  {k}: {val:.2f} {unit}")
            else:
                lines.append(f"  {k}: {val:.2f} {unit}  (normal: {nr})")

        lines += ["", "── Diagnosis ──", result.get('diagnosis', '')]
        return "\n".join(lines)

    # ─────────────────────────────────────────
    # Log Output
    # ─────────────────────────────────────────
    def _log(self, msg: str):
        """Append a message to the text browser and scroll to the bottom."""
        current = self.textBrowser.toPlainText()
        self.textBrowser.setPlainText((current + "\n" + msg) if current else msg)
        sb = self.textBrowser.verticalScrollBar()
        sb.setValue(sb.maximum())
        QApplication.processEvents()

    # ─────────────────────────────────────────
    # Keyboard Navigation (A / D)
    # ─────────────────────────────────────────
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_A:
            self._switch_image(-1)
        elif key == Qt.Key_D:
            self._switch_image(1)
        else:
            super().keyPressEvent(event)

    def _switch_image(self, direction: int):
        """Navigate to the previous (-1) or next (+1) image."""
        if not self.img_paths:
            return
        new_idx = self.current_idx + direction
        if new_idx < 0:
            QMessageBox.information(self, "Notice", "Already at the first image.")
            return
        if new_idx >= len(self.img_paths):
            QMessageBox.information(self, "Notice", "Already at the last image.")
            return
        self.current_idx = new_idx
        self._display_current()

    # ─────────────────────────────────────────
    # Button Enable / Disable
    # ─────────────────────────────────────────
    def _set_buttons_enabled(self, enabled: bool):
        for btn in (self.openfile, self.predict,
                    self.output_labelme, self.output_info_image):
            btn.setEnabled(enabled)

    # ─────────────────────────────────────────
    # Resize Event — re-fit image to view
    # ─────────────────────────────────────────
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._scene.items():
            self.graphicsView.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
