"""
Synthetic multi-scanner brain-MRI generator.

Real cross-scanner MRI datasets (BraTS, etc.) are hundreds of GB and require
licensed access, so this module generates *structurally realistic* synthetic
brain-slice images with tumor masks, and applies scanner-specific intensity
transforms to simulate the domain shift that occurs between MRI scanners
(different vendors / field strengths / acquisition protocols).

Domain A ("Scanner-A", e.g. Siemens 1.5T)  -> SOURCE domain, labels available
Domain B ("Scanner-B", e.g. GE 3T)          -> TARGET domain, labels withheld
                                                during training (used only for
                                                evaluation, exactly as in
                                                unsupervised domain adaptation)

Each sample is a 128x128 single-channel "brain slice" (skull ellipse with
soft-tissue texture) containing zero or one synthetic tumor (bright/dark
blob with irregular boundary).
"""
import numpy as np
import cv2


IMG_SIZE = 96


def _make_brain_mask(size=IMG_SIZE):
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size / 2, size / 2
    ry, rx = size * 0.42, size * 0.36
    mask = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
    return mask.astype(np.float32)


def _perlin_like_texture(size=IMG_SIZE, octaves=4, seed=None):
    rng = np.random.default_rng(seed)
    texture = np.zeros((size, size), dtype=np.float32)
    freq = 4
    amp = 1.0
    for _ in range(octaves):
        small = rng.normal(0, 1, (freq, freq)).astype(np.float32)
        up = cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
        texture += amp * up
        freq *= 2
        amp *= 0.5
    texture -= texture.min()
    texture /= (texture.max() + 1e-8)
    return texture


def _random_blob_mask(size, center, radius, irregularity, seed=None):
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, 64)
    r = radius * (1 + irregularity * rng.normal(0, 1, theta.shape))
    r = np.clip(r, radius * 0.4, radius * 1.8)
    pts = np.stack([
        center[0] + r * np.cos(theta),
        center[1] + r * np.sin(theta)
    ], axis=1).astype(np.int32)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(np.float32)


def generate_sample(seed=None, with_tumor=True):
    """Generate one clean (pre-domain-shift) synthetic brain slice + mask."""
    rng = np.random.default_rng(seed)
    size = IMG_SIZE
    brain = _make_brain_mask(size)
    tissue = _perlin_like_texture(size, seed=seed)
    base = 0.35 + 0.35 * tissue
    base *= brain

    mask = np.zeros((size, size), dtype=np.float32)
    if with_tumor:
        cx = rng.integers(int(size * 0.35), int(size * 0.65))
        cy = rng.integers(int(size * 0.35), int(size * 0.65))
        radius = rng.integers(8, 20)
        tumor_mask = _random_blob_mask(size, (cx, cy), radius,
                                        irregularity=0.25, seed=seed)
        tumor_mask *= brain
        core = _random_blob_mask(size, (cx, cy), radius * 0.4,
                                  irregularity=0.3, seed=(seed or 0) + 999)
        tumor_intensity = 0.85 * tumor_mask - 0.3 * core * tumor_mask
        base = base * (1 - tumor_mask) + (base + tumor_intensity) * tumor_mask
        mask = tumor_mask

    noise = rng.normal(0, 0.02, (size, size)).astype(np.float32)
    img = np.clip(base + noise, 0, 1)
    return img.astype(np.float32), mask.astype(np.float32)


def apply_domain_shift(img, domain="A"):
    """Simulate scanner-dependent intensity characteristics.

    Domain A: standard contrast / low noise   (reference scanner)
    Domain B: gamma shift + bias field + higher noise + different contrast
              curve -> mimics a different vendor / field strength
    """
    img = img.copy()
    size = img.shape[0]
    if domain == "A":
        gamma = 1.0
        gain, bias = 1.0, 0.0
        noise_std = 0.015
    else:
        gamma = 1.6
        gain, bias = 1.25, -0.08
        noise_std = 0.045

    yy, xx = np.mgrid[0:size, 0:size]
    bf_strength = 0.05 if domain == "A" else 0.18
    bias_field = 1 + bf_strength * np.sin(xx / size * np.pi) * np.cos(yy / size * np.pi)

    img = np.clip(img * gain + bias, 0, 1)
    img = np.power(img, gamma)
    img = img * bias_field
    img = np.clip(img, 0, 1)
    img = img + np.random.default_rng().normal(0, noise_std, img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


def generate_domain_batch(n, domain="A", with_tumor_ratio=0.85, seed_offset=0):
    imgs, masks = [], []
    for i in range(n):
        seed = seed_offset + i
        with_tumor = np.random.default_rng(seed).random() < with_tumor_ratio
        clean_img, mask = generate_sample(seed=seed, with_tumor=with_tumor)
        shifted = apply_domain_shift(clean_img, domain=domain)
        imgs.append(shifted)
        masks.append(mask)
    return np.stack(imgs), np.stack(masks)
