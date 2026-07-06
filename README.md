# Cross-Scanner MRI Segmentation using Unsupervised Domain Adaptation

A B.Tech AI project: a Flask web app that segments brain tumors in MRI slices
from **different scanners**, using a **U-Net + Gradient-Reversal domain-
adversarial network (DANN)** so the model generalizes to a scanner it never
saw labels for (Unsupervised Domain Adaptation).

> Note: this is delivered as a **Flask** app (server-rendered HTML + Bootstrap
> + Chart.js), as requested, rather than a separate React frontend. The
> architecture is layered so a React/FastAPI split can be added later without
> touching the model or training code (see "Migrating to React + FastAPI" below).

## Features
- Upload any MRI slice (PNG/JPG/BMP/TIFF) **or** generate a synthetic sample
  scan from "Scanner A" (source domain) or "Scanner B" (target domain) to
  test the pipeline instantly, no dataset required.
- Choose between the **Baseline U-Net** (no domain adaptation) and the
  **UDA/DANN U-Net** to compare cross-scanner robustness.
- Segmentation visualization: input slice, predicted tumor mask, and a
  red-overlay + contour visualization (green = ground truth, when available).
- Evaluation dashboard: Dice / IoU bar charts, training-loss curves, domain-
  classifier loss curve, and usage statistics, all via Chart.js.
- Results are persisted to a database (PostgreSQL or SQLite) so past scans
  can be revisited in a gallery view.

## Tech Stack
| Layer | Technology |
|---|---|
| Deep Learning | PyTorch — U-Net encoder/decoder + Gradient Reversal Layer domain classifier (DANN) |
| Classical ML / metrics | Dice coefficient, IoU (NumPy) |
| Backend | Flask |
| Database | PostgreSQL (via SQLAlchemy) / SQLite fallback |
| Frontend | Jinja2 + Bootstrap 5 + Chart.js |
| Image I/O | Pillow, OpenCV |

## Project Structure
```
mri-uda-project/
├── app/
│   ├── app.py            # Flask routes (upload, segment, dashboard, gallery)
│   ├── database.py        # SQLAlchemy models (PostgreSQL/SQLite)
│   ├── inference.py       # Model loading + segmentation + overlay generation
│   ├── static/
│   │   ├── css/style.css
│   │   ├── uploads/        # uploaded/sample input images
│   │   └── results/        # predicted masks + overlays
│   └── templates/          # index, upload, result, gallery, dashboard, about
├── models/
│   ├── uda_unet.py         # U-Net + GRL + domain classifier definition
│   ├── baseline_unet.pth   # trained baseline weights
│   ├── uda_unet.pth        # trained UDA weights
│   └── metrics.json        # evaluation metrics used by the dashboard
├── data/
│   └── synthetic_mri.py    # synthetic multi-scanner MRI + domain-shift generator
├── scripts/
│   └── train.py            # trains baseline + UDA models, writes metrics.json
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# CPU-only PyTorch install (recommended, much smaller download):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 1. Train the models (already trained weights are included, but you can retrain)
```bash
python scripts/train.py
```
This generates a synthetic two-scanner dataset, trains a **baseline** U-Net
(source-domain labels only) and a **UDA** U-Net (DANN with source labels +
target-domain images, no target labels), evaluates both on held-out source
and target data, and saves `models/*.pth` + `models/metrics.json`.

### 2. Run the web app
```bash
python app/app.py
```
Visit **http://127.0.0.1:5000**

By default this uses a local SQLite file (`mri_uda.db`, auto-created). To use
PostgreSQL instead:
```bash
createdb mri_uda
export DATABASE_URL=postgresql://user:password@localhost:5432/mri_uda
python app/app.py
```

## Using real MRI data instead of synthetic data
Replace `data/synthetic_mri.py`'s `generate_domain_batch()` calls in
`scripts/train.py` with loaders for your real dataset:
- **Source domain**: labeled 2D slices + tumor masks from one scanner
  (e.g. one BraTS institution/scanner).
- **Target domain**: unlabeled 2D slices from a different scanner.
Keep images grayscale, resized to 96×96 (or change `IMG_SIZE` in
`data/synthetic_mri.py` and retrain), normalized to [0, 1]. No other code
changes are required — `UDAUNet`, the training loop, and the Flask app are
data-agnostic.

## How the UDA (DANN) method works
1. A shared U-Net encoder produces bottleneck features for both domains.
2. The segmentation decoder is trained only on labeled source-domain data
   (Dice + BCE loss).
3. A small domain classifier tries to predict which domain a bottleneck
   feature came from (source vs target), trained on **both** domains
   (labels are just "which domain", not tumor masks — that's what makes it
   *unsupervised* w.r.t. the target domain).
4. A **Gradient Reversal Layer** sits between the encoder and the domain
   classifier: on the forward pass it's the identity function; on the
   backward pass it multiplies the gradient by `-λ`. This means the encoder
   is trained to make source/target features **indistinguishable** to the
   domain classifier, i.e. to learn scanner-invariant features — which is
   what lets segmentation knowledge learned on Scanner A transfer to
   Scanner B.

Reference: Ganin et al., *"Domain-Adversarial Training of Neural Networks"*, JMLR 2016.

## Migrating to React + FastAPI (optional next step)
The original brief also mentioned a React + FastAPI + PostgreSQL stack. To
split this Flask app into that architecture later:
- Move `app/inference.py` + `models/` + `data/` unchanged into a **FastAPI**
  service exposing `POST /segment` (accepts an image, returns mask/overlay
  URLs + metrics) and `GET /metrics`.
- Keep `app/database.py` (SQLAlchemy models work identically under FastAPI).
- Replace the Jinja2 templates with a **React** SPA that calls the FastAPI
  endpoints and renders the same charts with `chart.js`/`recharts`.
No model or training code needs to change — only the web layer.
