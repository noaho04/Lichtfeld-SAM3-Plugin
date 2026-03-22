import time
from pathlib import Path

import lichtfeld as lf

PLUGIN_DIR = Path(__file__).parent


def _get_scene_image_dir() -> Path:
    """Resolve the images directory from the currently loaded dataset."""
    params = lf.dataset_params()
    data_path = Path(params.data_path)
    images_dir = data_path / params.images
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"No images directory found at: {images_dir}"
        )
    return images_dir


def download_weights(panel):
    """Download SAM3 weights from HuggingFace."""
    try:
        panel._running = True
        panel._status = "Downloading SAM3 weights..."

        from huggingface_hub import hf_hub_download
        weights_path = PLUGIN_DIR / "sam3.pt"
        if not weights_path.exists():
            hf_hub_download(
                repo_id="facebook/sam3",
                filename="sam3.pt",
                local_dir=str(PLUGIN_DIR),
                token=True,
            )
            panel._status = "SAM3 weights downloaded."
        else:
            panel._status = "SAM3 weights already present."
        panel._running = False

    except Exception as e:
        panel._status = f"Download failed: {e}"
        panel._running = False


def extract_masks(panel, prompts, fill_holes, confidence, imgsz, dilate_px=0):
    """Run SAM3 mask extraction on the scene's images."""
    try:
        panel._running = True
        panel._status = "Loading SAM3 model..."

        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from scipy.ndimage import binary_dilation, binary_fill_holes

        images_dir = _get_scene_image_dir()
        out_dir = images_dir.parent / "masks"
        out_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = str(PLUGIN_DIR / "sam3.pt")

        if not Path(checkpoint).exists():
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="facebook/sam3",
                filename="sam3.pt",
                local_dir=str(PLUGIN_DIR),
                token=True,
            )

        device = "cuda:0"

        panel._status = "Loading SAM3 on CUDA..."

        from ultralytics.models.sam import SAM3SemanticPredictor

        overrides = dict(
            conf=confidence,
            task="segment",
            mode="predict",
            model=checkpoint,
            device=device,
            save=False,
            imgsz=imgsz,
        )
        predictor = SAM3SemanticPredictor(overrides=overrides)

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        images = sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in exts
        )

        if not images:
            panel._status = f"No images found in {images_dir}"
            panel._running = False
            return

        start = time.time()
        for i, img_path in enumerate(images, 1):
            panel._status = f"Processing {i}/{len(images)}: {img_path.name}"

            with Image.open(img_path) as img:
                orig_w, orig_h = img.size

            predictor.set_image(str(img_path))
            results = predictor(text=prompts)

            masks = []
            for result in results:
                if result.masks is None:
                    continue
                for mask_tensor in result.masks.data:
                    mask = F.interpolate(
                        mask_tensor.unsqueeze(0).unsqueeze(0).float(),
                        size=(orig_h, orig_w),
                        mode="nearest",
                    )[0, 0].bool().cpu().numpy()
                    masks.append(mask)

            if not masks:
                mask = np.zeros((orig_h, orig_w), dtype=bool)
            else:
                mask = np.logical_or.reduce(masks)

            if fill_holes:
                mask = binary_fill_holes(mask)

            if dilate_px > 0:
                struct = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), dtype=bool)
                mask = binary_dilation(mask, structure=struct)

            result_img = (mask.astype(np.uint8)) * 255
            Image.fromarray(result_img, mode="L").save(out_dir / img_path.name)

        # Free VRAM before signaling completion
        del predictor
        torch.cuda.empty_cache()

        elapsed = time.time() - start
        panel._status = f"Done — {len(images)} masks in {elapsed:.1f}s -> {out_dir}"
        panel._running = False
        panel._masks_ready = True

    except Exception as e:
        panel._status = f"Error: {e}"
        panel._running = False
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
