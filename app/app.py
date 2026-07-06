import os
import sys
import uuid
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from database import db, SegmentationResult, get_database_uri  # noqa: E402
from inference import (  # noqa: E402
    run_segmentation, make_overlay, save_array_as_png,
    preprocess_uploaded_image, generate_sample_image, get_metrics
)
from models.uda_unet import dice_coefficient, iou_score  # noqa: E402

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = get_database_uri()
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB uploads

UPLOAD_DIR = os.path.join(BASE_DIR, "app", "static", "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "app", "static", "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

db.init_app(app)
with app.app_context():
    db.create_all()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/")
def index():
    metrics = get_metrics()
    recent = SegmentationResult.query.order_by(SegmentationResult.created_at.desc()).limit(6).all()
    return render_template("index.html", metrics=metrics, recent=[r.to_dict() for r in recent])


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload.html")

    model_choice = request.form.get("model_choice", "uda")
    scanner_domain = request.form.get("scanner_domain", "A")
    use_sample = request.form.get("use_sample") == "1"

    gt_mask = None

    if use_sample:
        img_arr, gt_mask = generate_sample_image(domain=scanner_domain)
        original_filename = f"sample_scanner_{scanner_domain}.png"
        uid = uuid.uuid4().hex[:10]
        input_path = os.path.join(UPLOAD_DIR, f"{uid}_input.png")
        save_array_as_png(img_arr, input_path)
    else:
        file = request.files.get("mri_file")
        if not file or file.filename == "":
            flash("Please choose an MRI image file, or use a sample scan.")
            return redirect(url_for("upload"))
        if not allowed_file(file.filename):
            flash("Unsupported file type. Please upload PNG/JPG/BMP/TIFF.")
            return redirect(url_for("upload"))

        uid = uuid.uuid4().hex[:10]
        original_filename = file.filename
        raw_path = os.path.join(UPLOAD_DIR, f"{uid}_raw_{original_filename}")
        file.save(raw_path)
        img_arr = preprocess_uploaded_image(raw_path)
        input_path = os.path.join(UPLOAD_DIR, f"{uid}_input.png")
        save_array_as_png(img_arr, input_path)

    # --- run inference ---
    pred_mask, probs = run_segmentation(img_arr, model_name=model_choice)

    mask_path = os.path.join(RESULTS_DIR, f"{uid}_mask.png")
    overlay_path = os.path.join(RESULTS_DIR, f"{uid}_overlay.png")
    save_array_as_png(pred_mask, mask_path, is_mask=True)

    overlay_rgb = make_overlay(img_arr, pred_mask, gt_mask=gt_mask)
    save_array_as_png(overlay_rgb, overlay_path)

    tumor_pixels = int(pred_mask.sum())
    tumor_area_pct = 100.0 * tumor_pixels / pred_mask.size

    dice_val, iou_val = None, None
    if gt_mask is not None:
        dice_val = dice_coefficient(pred_mask, gt_mask)
        iou_val = iou_score(pred_mask, gt_mask)

    result = SegmentationResult(
        original_filename=original_filename,
        scanner_domain=scanner_domain,
        model_used=model_choice,
        input_image_path=f"uploads/{os.path.basename(input_path)}",
        mask_image_path=f"results/{os.path.basename(mask_path)}",
        overlay_image_path=f"results/{os.path.basename(overlay_path)}",
        tumor_pixel_count=tumor_pixels,
        tumor_area_percent=tumor_area_pct,
        dice_score=dice_val,
        iou_score=iou_val,
    )
    db.session.add(result)
    db.session.commit()

    return redirect(url_for("view_result", result_id=result.id))


@app.route("/result/<int:result_id>")
def view_result(result_id):
    result = SegmentationResult.query.get_or_404(result_id)
    return render_template("result.html", r=result.to_dict())


@app.route("/results")
def results_gallery():
    all_results = SegmentationResult.query.order_by(SegmentationResult.created_at.desc()).all()
    return render_template("gallery.html", results=[r.to_dict() for r in all_results])


@app.route("/dashboard")
def dashboard():
    metrics = get_metrics()
    all_results = SegmentationResult.query.all()
    domain_counts = {"A": 0, "B": 0}
    model_counts = {"baseline": 0, "uda": 0}
    for r in all_results:
        if r.scanner_domain in domain_counts:
            domain_counts[r.scanner_domain] += 1
        if r.model_used in model_counts:
            model_counts[r.model_used] += 1
    return render_template(
        "dashboard.html",
        metrics=metrics,
        total_scans=len(all_results),
        domain_counts=domain_counts,
        model_counts=model_counts,
    )


@app.route("/api/metrics")
def api_metrics():
    return jsonify(get_metrics() or {})


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5101))
    app.run(host="0.0.0.0", port=port, debug=True)
