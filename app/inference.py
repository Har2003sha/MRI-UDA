import os
import sys
import json
import numpy as np
import torch
import cv2
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.uda_unet import UDAUNet, dice_coefficient, iou_score  # noqa: E402
from data.synthetic_mri import generate_domain_batch, apply_domain_shift, IMG_SIZE  # noqa: E402

MODELS_DIR = os.path.join(BASE_DIR, "models")
DEVICE = torch.device("cpu")

_loaded_models = {}


def load_model(name):
    """name: 'baseline' or 'uda'"""
    if name in _loaded_models:
        return _loaded_models[name]
    path = os.path.join(MODELS_DIR, f"{name}_unet.pth")
    model = UDAUNet()
    if os.path.exists(path):
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)
    model.eval()
    _loaded_models[name] = model
    return model


def get_metrics():
    path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def preprocess_uploaded_image(file_path):
    """Load any uploaded image (jpg/png/etc), convert to grayscale,
    resize to model input size, normalize to [0,1]."""
    img = Image.open(file_path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


def generate_sample_image(domain="A"):
    """Generate a fresh synthetic MRI sample (used for the 'Try a sample
    scan' feature so users can test the pipeline without owning real MRI
    files)."""
    imgs, masks = generate_domain_batch(1, domain=domain, seed_offset=np.random.randint(0, 1_000_000))
    return imgs[0], masks[0]


@torch.no_grad()
def run_segmentation(img_array, model_name="uda"):
    """img_array: 2D float32 array in [0,1], shape (IMG_SIZE, IMG_SIZE)
    Returns: predicted binary mask (2D array), raw probability map
    """
    model = load_model(model_name)
    x = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).float()
    seg_logits, _ = model(x, return_domain=False)
    probs = torch.sigmoid(seg_logits)[0, 0].numpy()
    pred_mask = (probs > 0.5).astype(np.float32)
    return pred_mask, probs


def make_overlay(img_array, pred_mask, gt_mask=None):
    """Create an RGB overlay: grayscale MRI + predicted tumor region in red
    (+ ground-truth boundary in green, if available, for visual QA)."""
    base = (img_array * 255).astype(np.uint8)
    rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    red_layer = np.zeros_like(rgb)
    red_layer[:, :, 2] = 255  # BGR -> red channel
    mask_bool = pred_mask > 0.5
    overlay = rgb.copy()
    overlay[mask_bool] = cv2.addWeighted(rgb, 0.4, red_layer, 0.6, 0)[mask_bool]

    # predicted contour in bright yellow
    contours, _ = cv2.findContours((pred_mask * 255).astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)

    if gt_mask is not None:
        gt_contours, _ = cv2.findContours((gt_mask * 255).astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 1)  # green = ground truth

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def save_array_as_png(arr_uint8_or_float, path, is_mask=False):
    if is_mask:
        img = (arr_uint8_or_float * 255).astype(np.uint8)
        Image.fromarray(img).save(path)
    else:
        if arr_uint8_or_float.dtype != np.uint8:
            img = (arr_uint8_or_float * 255).astype(np.uint8) if arr_uint8_or_float.max() <= 1.0 else arr_uint8_or_float.astype(np.uint8)
        else:
            img = arr_uint8_or_float
        Image.fromarray(img).save(path)
