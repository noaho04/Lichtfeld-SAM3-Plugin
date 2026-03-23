import os
os.environ["YOLO_AUTOINSTALL"] = "false"  # prevent ultralytics runtime pip calls

import threading

from pathlib import Path

import lichtfeld as lf

from .masks import download_weights, extract_masks

_WEIGHTS_PATH = Path(__file__).parent / "sam3.pt"


class SAM3MaskPanel(lf.ui.Panel):
    id = "sam3_masks.main"
    label = "SAM3 Masks"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 300

    def __init__(self):
        self._prompts = ""
        self._fill_holes = False
        self._confidence = 0.25
        self._imgsz = 644
        self._dilate_px = 0
        self._status = ""
        self._running = False
        self._masks_ready = False

    def draw(self, ui):
        ui.heading("SAM3 Mask Extraction")
        ui.separator()

        # -- Setup --
        try:
            import torch
            build = "CUDA" if torch.cuda.is_available() else "CPU"
            ui.label(f"PyTorch: {torch.__version__} ({build})")
        except ImportError:
            ui.text_disabled("PyTorch not yet installed")

        if not _WEIGHTS_PATH.exists():
            if ui.button_styled("Get SAM3 Weights", "secondary") and not self._running:
                threading.Thread(target=download_weights, args=(self,), daemon=True).start()
        else:
            _, self._prompts = ui.input_text("Prompts (x, y)", self._prompts)
            _, self._fill_holes = ui.checkbox("Fill holes in masks", self._fill_holes)
            _, self._dilate_px = ui.slider_int("Dilate (px)", self._dilate_px, 0, 50)
            _, self._confidence = ui.slider_float(
                "Confidence", self._confidence, 0.05, 1.0
            )
            _, self._imgsz = ui.slider_int(
                "Inference Resolution", self._imgsz, 224, 1792
            )
            self._imgsz = (self._imgsz // 14) * 14
            ui.text_disabled(f"(snapped to {self._imgsz}, must be divisible by 14)")

            ui.separator()

            can_run = lf.has_scene() and self._prompts.strip() and not self._running
            if not can_run and not self._running:
                ui.text_disabled("Load a scene and enter prompts to run.")
            if ui.button_styled("Extract Masks", "primary") and can_run:
                prompts = [p.strip() for p in self._prompts.split(",") if p.strip()]
                self._masks_ready = False
                threading.Thread(
                    target=extract_masks,
                    args=(self, prompts, self._fill_holes, self._confidence, self._imgsz, self._dilate_px),
                    daemon=True,
                ).start()

            if self._masks_ready:
                ui.separator()
                if ui.button_styled("Reload Dataset", "success"):
                    data_path = lf.dataset_params().data_path
                    lf.load_file(data_path, is_dataset=True)
                    self._masks_ready = False

            if self._status:
                ui.separator()
                ui.label(self._status)
